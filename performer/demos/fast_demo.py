from datetime import datetime
import tensorflow as tf

from performer.networks.model import Performer


if __name__ == '__main__':
    layer = Performer(num_heads=2,
                      key_dim=20,
                      attention_method='quadratic',
                      supports=2)

    x = tf.random.uniform(shape=(2000, 4, 3))
    y = tf.random.uniform(shape=(2000, 4, 3))
    inputs = tf.keras.layers.Input(shape=[4, 3])
    outputs = layer(inputs, inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile("adam", "mean_squared_error")

    logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logs, profile_batch=5)

    model.fit(x, y, epochs=10, callbacks=[tb_callback])
