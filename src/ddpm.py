import tensorflow as tf
import numpy as np

class SinusoidalEmbedding(tf.keras.layers.Layer):
    """Sinusoidal embeddings for current noise rate embeddings"""
    def __init__(self, dim, *args, **kwargs):
        assert dim % 2 == 0
        super().__init__(*args, **kwargs)
        self.dim = dim

    def call(self, inputs):
        cos_embeddings = []
        sin_embeddings = []
        for i in range(0, self.dim // 2, 1):
            freq1 = tf.pow(20.0, 2.0 * i / self.dim)
            freq2 = tf.pow(20.0, 2.0 * i / self.dim)
            sin_embeddings.append(tf.math.sin(2.0 * tf.constant(np.pi, dtype=tf.float32) * inputs / freq1))
            cos_embeddings.append(tf.math.cos(2.0 * tf.constant(np.pi, dtype=tf.float32) * inputs / freq2))

        sin_embeddings = tf.concat(sin_embeddings, axis=-1)
        cos_embeddings = tf.concat(cos_embeddings, axis=-1)

        return tf.concat([sin_embeddings, cos_embeddings], axis=-1)

