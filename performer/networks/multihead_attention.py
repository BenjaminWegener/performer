from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import string

import numpy as np

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers import advanced_activations
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.layers import einsum_dense
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops


_CHR_IDX = string.ascii_lowercase


def _build_attention_equation(rank, attn_axes):
  target_notation = _CHR_IDX[:rank]
  # `batch_dims` includes the head dim.
  batch_dims = tuple(np.delete(range(rank), attn_axes + (rank - 1,)))
  letter_offset = rank
  source_notation = ""
  for i in range(rank):
    if i in batch_dims or i == rank - 1:
      source_notation += target_notation[i]
    else:
      source_notation += _CHR_IDX[letter_offset]
      letter_offset += 1

  product_notation = "".join([target_notation[i] for i in batch_dims] +
                             [target_notation[i] for i in attn_axes] +
                             [source_notation[i] for i in attn_axes])
  dot_product_equation = "%s,%s->%s" % (source_notation, target_notation,
                                        product_notation)
  attn_scores_rank = len(product_notation)
  combine_equation = "%s,%s->%s" % (product_notation, source_notation,
                                    target_notation)
  return dot_product_equation, combine_equation, attn_scores_rank


def _build_proj_equation(free_dims, bound_dims, output_dims):
  """Builds an einsum equation for projections inside multi-head attention."""
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


class MultiHeadAttention(Layer):

  def __init__(self,
               num_heads,
               key_dim,
               value_dim=None,
               dropout=0.0,
               use_bias=True,
               output_shape=None,
               attention_axes=None,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(MultiHeadAttention, self).__init__(**kwargs)
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
    if attention_axes is not None and not isinstance(attention_axes,
                                                     collections.abc.Sized):
      self._attention_axes = (attention_axes,)
    else:
      self._attention_axes = attention_axes
    self._built_from_signature = False

  def get_config(self):
    config = {
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
    base_config = super(MultiHeadAttention, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _build_from_signature(self, query, value, key=None):
    """Builds layers and variables.
    Once the method is called, self._built_from_signature will be set to True.
    Args:
      query: query tensor or TensorShape.
      value: value tensor or TensorShape.
      key: key tensor or TensorShape.
    """
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
    # Any setup work performed only once should happen in an `init_scope`
    # to avoid creating symbolic Tensors that will later pollute any eager
    # operations.
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
      self._key_dense = einsum_dense.EinsumDense(
          einsum_equation,
          output_shape=_get_output_shape(output_rank - 1,
                                         [self._num_heads, self._key_dim]),
          bias_axes=bias_axes if self._use_bias else None,
          name="key",
          **common_kwargs)
      einsum_equation, bias_axes, output_rank = _build_proj_equation(
          value_shape.rank - 1, bound_dims=1, output_dims=2)
      self._value_dense = einsum_dense.EinsumDense(
          einsum_equation,
          output_shape=_get_output_shape(output_rank - 1,
                                         [self._num_heads, self._value_dim]),
          bias_axes=bias_axes if self._use_bias else None,
          name="value",
          **common_kwargs)

      # Builds the attention computations for multi-head dot product attention.
      # These computations could be wrapped into the keras attention layer once
      # it support mult-head einsum computations.
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

  def _build_attention(self, rank):
    if self._attention_axes is None:
      self._attention_axes = tuple(range(1, rank - 2))
    else:
      self._attention_axes = tuple(self._attention_axes)
    self._dot_product_equation, self._combine_equation, attn_scores_rank = (
        _build_attention_equation(rank, attn_axes=self._attention_axes))
    norm_axes = tuple(
        range(attn_scores_rank - len(self._attention_axes), attn_scores_rank))
    self._softmax = advanced_activations.Softmax(axis=norm_axes)
    self._dropout_layer = core.Dropout(rate=self._dropout)

  def _masked_softmax(self, attention_scores, attention_mask=None):
    if attention_mask is not None:
      mask_expansion_axes = [-len(self._attention_axes) * 2 - 1]
      for _ in range(len(attention_scores.shape) - len(attention_mask.shape)):
        attention_mask = array_ops.expand_dims(
            attention_mask, axis=mask_expansion_axes)
    return self._softmax(attention_scores, attention_mask)

  def _compute_attention(self,
                         query,
                         key,
                         value,
                         attention_mask=None,
                         training=None):
    # Note: Applying scalar multiply at the smaller end of einsum improves
    # XLA performance, but may introduce slight numeric differences in
    # the Transformer attention head.
    query = math_ops.multiply(query, 1.0 / math.sqrt(float(self._key_dim)))

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    attention_scores = special_math_ops.einsum(self._dot_product_equation, key,
                                               query)

    attention_scores = self._masked_softmax(attention_scores, attention_mask)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_scores_dropout = self._dropout_layer(
        attention_scores, training=training)

    # `context_layer` = [B, T, N, H]
    attention_output = special_math_ops.einsum(self._combine_equation,
                                               attention_scores_dropout, value)
    return attention_output, attention_scores

  def call(self,
           query,
           value,
           key=None,
           attention_mask=None,
           return_attention_scores=False,
           training=None):
    if not self._built_from_signature:
      self._build_from_signature(query=query, value=value, key=key)
    if key is None:
      key = value

    #   N = `num_attention_heads`
    #   H = `size_per_head`
    # `query` = [B, T, N ,H]
    query = self._query_dense(query)

    # `key` = [B, S, N, H]
    key = self._key_dense(key)

    # `value` = [B, S, N, H]
    value = self._value_dense(value)

    attention_output, attention_scores = self._compute_attention(
        query, key, value, attention_mask, training)
    attention_output = self._output_dense(attention_output)

    if return_attention_scores:
      return attention_output, attention_scores
    return attention_output

if __name__ == '__main__':
    import tensorflow as tf
    layer = MultiHeadAttention(num_heads=2, key_dim=2)
    target = tf.keras.Input(shape=[8, 16])
    source = tf.keras.Input(shape=[4, 16])
    output_tensor, weights = layer(target, source,return_attention_scores=True)
