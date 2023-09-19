import argparse
import numpy as np
import tensorflow as tf


class SinusoidalEmbedding(tf.keras.layers.Layer):
    """Sinusoidal embeddings for current noise rate embeddings"""
    def __init__(self, dim: int, *args, **kwargs):
        assert dim % 2 == 0
        super().__init__(*args, **kwargs)
        self.dim = dim

    def call(self, inputs, **kwargs):
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


def pre_activated_resnet_block(inputs, width, embeddings):
    """Residual block with noise embeddings"""
    residual = inputs if inputs.shape[-1] == width else tf.keras.layers.Conv2D(width, 1)(inputs)

    hidden = tf.keras.layers.BatchNormalization()(inputs)
    hidden = tf.keras.activations.swish(hidden)
    hidden = tf.keras.layers.Conv2D(width, 3, padding="same", use_bias=False)(hidden)

    noise_layer = tf.keras.layers.Dense(width, activation=tf.keras.activations.swish)(embeddings)
    hidden += noise_layer

    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.activations.swish(hidden)
    hidden = tf.keras.layers.Conv2D(width, 3, padding="same", use_bias=False,
                                    kernel_initializer=tf.keras.initializers.Constant(value=0))(hidden)

    hidden += residual
    return hidden


class DDIM(tf.keras.Model):
    """Denoising Diffusion Probabilistic Model"""
    def __init__(self, args: argparse.Namespace, data):
        super().__init__()

        inputs = tf.keras.layers.Input([args.height, args.width, args.channels])
        low_resolution_image = tf.keras.layers.Input([args.height // args.downscale,
                                                      args.width // args.downscale,
                                                      args.channels])
        noise_rates = tf.keras.layers.Input([1, 1, 1])

        noise_embedding = SinusoidalEmbedding(args.cnn_channels)(noise_rates)

        hidden = tf.keras.layers.UpSampling2D(args.downscale, interpolation="bicubic")(low_resolution_image)

        hidden = tf.concat([inputs, hidden], axis=-1)
        hidden = tf.keras.layers.Conv2D(args.cnn_channels, 3, padding="same")(hidden)

        # Downscale
        outputs = []
        for i in range(args.stages):
            for _ in range(args.stage_blocks):
                hidden = pre_activated_resnet_block(hidden, args.cnn_channels << i, noise_embedding)
                outputs.append(hidden)
            hidden = tf.keras.layers.Conv2D(args.cnn_channels << (i + 1), 3, strides=2, padding="same")(hidden)

        # Middle
        for _ in range(args.stage_blocks):
            hidden = pre_activated_resnet_block(hidden, args.cnn_channels << args.stages, noise_embedding)

        # Upscale
        for i in reversed(range(args.stages)):
            hidden = tf.keras.layers.Conv2DTranspose(args.cnn_channels << i, (4, 4), strides=2, padding="same")(hidden)
            for _ in range(args.stage_blocks):
                hidden = tf.concat([hidden, outputs.pop()], axis=-1)
                hidden = pre_activated_resnet_block(hidden, args.cnn_channels << i, noise_embedding)

        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.activations.swish(hidden)
        hidden = tf.keras.layers.Conv2D(args.channels, 3, padding="same",
                                        kernel_initializer=tf.keras.initializers.Constant(value=0))(hidden)

        self._network = tf.keras.Model(inputs=[inputs, low_resolution_image, noise_rates], outputs=hidden)
        self._ema_network = tf.keras.models.clone_model(self._network)

        self._downscale = args.downscale
        self._ema_momentum = args.ema
        self._seed = args.seed

        self._image_normalization = tf.keras.layers.Normalization()
        self._image_normalization.adapt(data)

    def _image_denormalization(self, images):
        """Denornmalize the images"""
        images = self._image_normalization.mean + images * self._image_normalization.variance ** 0.5
        images = tf.clip_by_value(images, 0, 255)
        images = tf.cast(images, tf.uint8)
        return images

    def _diffusion_rates(self, times):
        """Calculate the diffusion rates"""
        starting_angle, final_angle = 0.2, 1.55
        angle = starting_angle + times * (final_angle - starting_angle)

        signal_rates = tf.reshape(tf.cos(angle), [-1, 1, 1, 1])
        noise_rates = tf.reshape(tf.sin(angle), [-1, 1, 1, 1])

        return signal_rates, noise_rates

    def train_step(self, images):
        """Train step of the whole model with ema network"""
        images = self._image_normalization(images)
        conditioning = tf.keras.layers.AveragePooling2D(self._downscale)(images)

        noises = tf.random.normal(tf.shape(images), seed=self._seed)
        times = tf.random.uniform(tf.shape(images)[:1], seed=self._seed)

        signal_rates, noise_rates = self._diffusion_rates(times)
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            predicted_noises = self._network([noisy_images, conditioning, noise_rates], training=True)
            loss = self.compiled_loss(predicted_noises, noises)

        self.optimizer.minimize(loss, self._network.trainable_variables, tape=tape)

        for ema_variable, variable in zip(self._ema_network.variables, self._network.variables):
            ema_variable.assign(self._ema_momentum * ema_variable + (1 - self._ema_momentum) * variable)


    def generate(self, initial_noise, conditioning, steps):
        """Sample a batch of images"""
        images = initial_noise
        conditioning = self._image_normalization(conditioning)
        steps = tf.linspace(tf.ones(tf.shape(initial_noise)[0]), tf.zeros(tf.shape(initial_noise)[0]), steps + 1)

        for times, next_times in zip(steps[:-1], steps[1:]):
            signal_rates, noise_rates = self._diffusion_rates(times)
            predicted_noises = self._ema_network([images, conditioning, noise_rates], training=False)

            next_signal_rates, next_noise_rates = self._diffusion_rates(next_times)
            denoised_images = (images - noise_rates * predicted_noises) / signal_rates

            images = next_signal_rates * denoised_images + next_noise_rates * predicted_noises

        images = self._image_denormalization(denoised_images)
        return images
