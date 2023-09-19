import os
import tensorflow as tf


class DatasetCreator:
    def __init__(self,
                 folder_path: str = "/projects/SkyGAN/clouds_fisheye/processed",
                 height: int = 128,
                 width: int = 128,
                 channels: int = 128):
        self.H = height
        self.W = width
        self.C = channels
        self._folder_path = folder_path

    def _list_jpg_files(self):
        return [os.path.join(root, file) for root, _, files in os.walk(self._folder_path) for file in files if
                file.endswith('.jpg')]

    def _filter_resolution(self, file_path):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)
        return tf.shape(image)[0] == 1024 and tf.shape(image)[1] == 1024

    def _load(self, file_path):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)
        return image

    def _resize(self, image):
        return tf.image.resize(image, [128, 128])

    @property
    def dataset(self):
        image_dataset = tf.data.Dataset.from_tensor_slices(self._list_jpg_files())
        image_dataset = image_dataset.filter(self._filter_resolution)
        image_dataset = image_dataset.map(self._load, num_parallel_calls=tf.data.AUTOTUNE)
        image_dataset = image_dataset.map(self._resize, num_parallel_calls=tf.data.AUTOTUNE)
        return image_dataset
