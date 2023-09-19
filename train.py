import os
import argparse
import tensorflow as tf

print(f"Avaliable GPUs: {tf.config.list_physical_devices('GPU')}")

from src.dataset import DatasetCreator
from src.ddim import DDIM

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--height", default=512, type=int, help="Up-scaled image height")
parser.add_argument("--width", default=512, type=int, help="Up-scaled image width")
parser.add_argument("--channels", default=3, type=int, help="Up-scaled image channels")
parser.add_argument("--cnn_channels", default=32, type=int, help="CNN channels in the first stage")
parser.add_argument("--downscale", default=4, type=int, help="Downscale factor")
parser.add_argument("--stages", default=4, type=int, help="UNet scaling stages")
parser.add_argument("--stage_blocks", default=2, type=int, help="ResNet blocks per stage")
parser.add_argument("--ema", default=0.999, type=float, help="Exponential moving average momentum")
parser.add_argument("--batch_size", default=64, type=int, help="Size of batch of the train set")
parser.add_argument("--epoch_batches", default=1000, type=int, help="Number of images per one epoch")
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs")
parser.add_argument("--test_size", default=100, type=int, help="Number of the images in the test set")
parser.add_argument("--log_folder", default="training_4x_to_512", type=str, help="Folder to save weights")

args = parser.parse_args([] if "__file__" not in globals() else None)

tf.keras.utils.set_random_seed(args.seed)

dataset = DatasetCreator(folder_path="/projects/SkyGAN/clouds_fisheye/processed",
                         height=args.height,
                         width=args.width,
                         channels=args.channels).dataset

ddim = DDIM(args, dataset)

train = dataset.shuffle(10 * args.batch_size, seed=args.seed)
train = train.skip(100).repeat()
train = train.batch(args.batch_size)
train = train.prefetch(tf.data.AUTOTUNE)

conditioning = dataset.take(80).batch(80).get_single_element()
conditioning = tf.cast(conditioning, dtype=tf.float32)
conditioning = tf.keras.layers.AveragePooling2D(args.downscale)(conditioning)
conditioning = tf.cast(conditioning, dtype=tf.uint8)

ddim.compile(
    optimizer=tf.optimizers.experimental.AdamW(jit_compile=False),
    loss=tf.losses.MeanAbsoluteError(),
)

checkpoint_path = os.path.join(args.log_folder, "cp.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

ddim.fit(
    train,
    epochs=args.epochs,
    steps_per_epoch=args.epoch_batches,
    callbacks=[cp_callback]
)
