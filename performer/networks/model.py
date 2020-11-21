import math
import string
import collections

from tensorflow import multiply, einsum
from tensorflow.python.keras.layers import advanced_activations
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.layers import core
from tensorflow.keras.layers import Layer
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras.layers import einsum_dense
from tensorflow.python.keras import regularizers
from tensorflow.python.ops import array_ops
from tensorflow.python.keras.utils import tf_utils
import numpy as np
import tensorflow as tf

from performer.networks.build_attention import build_linear_attention_equation
from performer.networks.build_attention import build_normalisation_equation
from performer.networks.build_attention import build_quadratic_attention_equation
from performer.networks.random_matrix_sampler import GaussianOrthogonalRandomMatrix as GOR
from performer.networks.random_matrix_sampler import kernel_feature_creator
from performer.networks.multihead_attention import MultiHeadAttention

_CHR_IDX = string.ascii_lowercase

class Performer(Layer):
    """Performer Layer

    This is an implementation of multi-head attention with linear attention via
    positive orthogonal random features approach.

    Examples:
    >>> layer = Performer(num_heads=2, key_dim=2,
                          attention_method='linear', supports=2)
    >>> target = tf.keras.Input(shape=[8, 16])
    >>> source = tf.keras.Input(shape=[4, 16])
    >>> output_tensor, weights = layer(target, source,
      ...                              return_attention_scores=True)
    >>> print(output_tensor.shape)
    (None, 8, 16)
    >>> print(weights.shape)
    (None, 2, 8, 4)
    """
    def __init__(self,
                 num_heads,
                 key_dim,
                 value_dim=None,
                 dropout=0.0,
                 use_bias=True,
                 output_shape=None,
                 attention_axes=None,
                 attention_method='linear',
                 scaling=0,
                 supports=None,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Performer, self).__init__(**kwargs)
        self._num_heads = num_heads
        self._key_dim = key_dim
        self._value_dim = value_dim if value_dim else key_dim
        self._dropout = dropout
        self._use_bias = use_bias
        self._output_shape = output_shape
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._bias_initializer = initializers.get(bias_initializer)
        self._kernel_regularizer = regularizers.get(kernel_regularizer)
        self._bias_regularizer = regularizers.get(bias_regularizer)
        self._kernel_constraint = constraints.get(kernel_constraint)
        self._bias_constraint = constraints.get(bias_constraint)
        self.attention_method = attention_method
        self.scaling = scaling
        self.supports = supports
        self._check_attention_method_is_valid()
        if attention_axes is not None and not isinstance(attention_axes,
                                                         collections.abc.Sized):
            self._attention_axes = (attention_axes,)
        else:
            self._attention_axes = attention_axes
        self._built_from_signature = False
        if self.attention_method == 'quadratic':
            self._compute_attention = self.quadratic_attention
            self._build_attention_equation = build_quadratic_attention_equation
        else:
            self._compute_attention = self.linear_attention
            self._build_attention_equation = build_linear_attention_equation
            self._check_supports_is_not_none()
            self.sampler = GOR(self.supports, key_dim, self.scaling)
            self._frozen_features = self._get_frozen_random_features(kwargs)
            self._build_normalisation_equation = build_normalisation_equation

    def call(self, *inputs, **kwargs):
        query, value, key = inputs
        attention_mask = kwargs.get('attention_mask')
        return_attention_scores = kwargs.get('return_attention_scores')
        training = kwargs.get('training')
        if not self._built_from_signature:
            self._build_from_signature(query=query, value=value, key=key)
        if key is None:
            key = value
        query = self._query_dense(query)
        key = self._key_dense(key)
        value = self._value_dense(value)
        attention_output, attention_scores = self._compute_attention(
            query, key, value, attention_mask, training)
        attention_output = self._output_dense(attention_output)
        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output

    def _get_frozen_random_features(self, kwargs):
        if '_frozen_features' in kwargs:
            frozen_features = kwargs.pop('_frozen_features')
        else:
            frozen_features = self.sampler.sample()
        return tf.constant(frozen_features, name='_frozen_features')

    def _check_supports_is_not_none(self):
        if self.supports is None:
            raise RuntimeError('must have numbers of supports specified')

    def _check_attention_method_is_valid(self):
        message = 'invalid attention method'
        assert self.attention_method in ['linear', 'quadratic'], message

    def _build_attention(self, rank):
        if self._attention_axes is None:
            self._attention_axes = tuple(range(1, rank - 2))
        else:
            self._attention_axes = tuple(self._attention_axes)
        self._add_attention_equation(rank)
        self._add_soft_max_and_dropout_layers()
        if hasattr(self, '_build_normalisation_equation'):
            self._add_normalisation_equation(rank)

    def _add_attention_equation(self, rank):
        result = self._build_attention_equation(rank, self._attention_axes)
        self._dot_product_equation, self._combine_equation, attn_scores_rank = result
        norm_axes = tuple(range(attn_scores_rank - len(self._attention_axes), attn_scores_rank))
        self._norm_axes = norm_axes

    def _add_soft_max_and_dropout_layers(self):
        self._softmax = advanced_activations.Softmax(axis=self._norm_axes)
        self._dropout_layer = core.Dropout(rate=self._dropout)

    def _add_normalisation_equation(self, rank):
        result = self._build_normalisation_equation(rank, self._attention_axes)
        self._k1_equation, self._q_k1_equation, self._qk1_q_equation = result

    def quadratic_attention(self, query, key, value, attention_mask=None, training=None):
        query = multiply(query, 1. / math.sqrt(float(self._key_dim)))
        attention_scores = einsum(self._dot_product_equation, key, query)
        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        attention_scores_dropout = self._dropout_layer(attention_scores, training=training)
        attention_output = einsum(self._combine_equation, attention_scores_dropout, value)
        return attention_output, attention_scores

    def _masked_softmax(self, attention_scores, attention_mask=None):
        if attention_mask is not None:
            mask_expansion_axes = [-len(self._attention_axes) * 2 - 1]
            for _ in range(len(attention_scores.shape) - len(attention_mask.shape)):
                attention_mask = array_ops.expand_dims(attention_mask, axis=mask_expansion_axes)
        return self._softmax(attention_scores, attention_mask)

    def linear_attention(self, query, key, value, attention_mask=None, training=None):
        if attention_mask is not None:
            raise NotImplementedError('masked linear attention not implemented')
        random_features = self._get_random_features(training)
        lifted_query = kernel_feature_creator(query, random_features, True)
        lifted_key = kernel_feature_creator(key, random_features, False)
        kv = einsum(self._dot_product_equation, lifted_key, value)
        qkv = einsum(self._combine_equation, lifted_query, kv)
        normalised_qkv = self._normalise(lifted_key, lifted_query, qkv)
        return normalised_qkv, None

    @tf.function
    def _get_random_features(self, train):
        out = self.sampler.sample() if train is None else self._frozen_features
        return out

    def _normalise(self, lifted_key, lifted_query, qkv):
        ones = tf.ones_like(lifted_key[..., 0])
        k_ones = einsum(self._k1_equation, lifted_key, ones)
        D = einsum(self._q_k1_equation, lifted_query, k_ones)
        D = 1. / (D + 1e-6)
        normalised_qkv = einsum(self._qk1_q_equation, D, qkv)
        return normalised_qkv

    def get_config(self):
        performer_config = {
            "attention_method":
                self.attention_method,
            "supports":
                self.supports,
            "scaling":
                self.scaling,
            "num_heads":
                self._num_heads,
            "key_dim":
                self._key_dim,
            "value_dim":
                self._value_dim,
            "dropout":
                self._dropout,
            "use_bias":
                self._use_bias,
            "output_shape":
                self._output_shape,
            "attention_axes":
                self._attention_axes,
            "kernel_initializer":
                initializers.serialize(self._kernel_initializer),
            "bias_initializer":
                initializers.serialize(self._bias_initializer),
            "kernel_regularizer":
                regularizers.serialize(self._kernel_regularizer),
            "bias_regularizer":
                regularizers.serialize(self._bias_regularizer),
            "activity_regularizer":
                regularizers.serialize(self._activity_regularizer),
            "kernel_constraint":
                constraints.serialize(self._kernel_constraint),
            "bias_constraint":
                constraints.serialize(self._bias_constraint)
        }
        if hasattr(self, '_frozen_features'):
            random_features = self._frozen_features.numpy()
            performer_config['_frozen_features'] = random_features
        base_config = super(Performer, self).get_config()
        base_config.update(performer_config)
        return base_config

    def _build_from_signature(self, query, value, key=None):
        self._built_from_signature = True
        if hasattr(query, "shape"):
            query_shape = tensor_shape.TensorShape(query.shape)
        else:
            query_shape = query
        if hasattr(value, "shape"):
            value_shape = tensor_shape.TensorShape(value.shape)
        else:
            value_shape = value
        if key is None:
            key_shape = value_shape
        elif hasattr(key, "shape"):
            key_shape = tensor_shape.TensorShape(key.shape)
        else:
            key_shape = key

        common_kwargs = dict(
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint)

        with tf_utils.maybe_init_scope(self):
            free_dims = query_shape.rank - 1
            einsum_equation, bias_axes, output_rank = _build_proj_equation(
                free_dims, bound_dims=1, output_dims=2)
            self._query_dense = einsum_dense.EinsumDense(
                einsum_equation,
                output_shape=_get_output_shape(output_rank - 1,
                                             [self._num_heads, self._key_dim]),
                bias_axes=bias_axes if self._use_bias else None,
                name="query",
                **common_kwargs)
            einsum_equation, bias_axes, output_rank = _build_proj_equation(
                    key_shape.rank - 1, bound_dims=1, output_dims=2)
            self._key_dense = einsum_dense.EinsumDense(einsum_equation,
                                                       output_shape=_get_output_shape(output_rank - 1,
                                                                                     [self._num_heads, self._key_dim]),
                                                      bias_axes=bias_axes if self._use_bias else None,
                                                      name="key",
                                                      **common_kwargs)
            einsum_equation, bias_axes, output_rank = _build_proj_equation(
              value_shape.rank - 1, bound_dims=1, output_dims=2)
            self._value_dense = einsum_dense.EinsumDense(
                einsum_equation,
                output_shape=_get_output_shape(output_rank - 1, [self._num_heads, self._value_dim]),
                bias_axes=bias_axes if self._use_bias else None,
                name="value",
                **common_kwargs)

            self._build_attention(output_rank)
            if self._output_shape:
                if not isinstance(self._output_shape, collections.abc.Sized):
                    output_shape = [self._output_shape]
                else:
                    output_shape = self._output_shape
            else:
                output_shape = [query_shape[-1]]
                einsum_equation, bias_axes, output_rank = _build_proj_equation(
                free_dims, bound_dims=2, output_dims=len(output_shape))
                self._output_dense = einsum_dense.EinsumDense(
                  einsum_equation,
                  output_shape=_get_output_shape(output_rank - 1, output_shape),
                  bias_axes=bias_axes if self._use_bias else None,
                  name="attention_output",
                  **common_kwargs)


def _build_proj_equation(free_dims, bound_dims, output_dims):
    input_str = ""
    kernel_str = ""
    output_str = ""
    bias_axes = ""
    letter_offset = 0
    for i in range(free_dims):
        char = _CHR_IDX[i + letter_offset]
        input_str += char
        output_str += char

    letter_offset += free_dims
    for i in range(bound_dims):
        char = _CHR_IDX[i + letter_offset]
        input_str += char
        kernel_str += char

    letter_offset += bound_dims
    for i in range(output_dims):
        char = _CHR_IDX[i + letter_offset]
        kernel_str += char
        output_str += char
        bias_axes += char
    equation = "%s,%s->%s" % (input_str, kernel_str, output_str)
    return equation, bias_axes, len(output_str)


def _get_output_shape(output_rank, known_last_dims):
    return [None] * (output_rank - len(known_last_dims)) + list(known_last_dims)
