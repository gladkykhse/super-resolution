import os
import argparse
import tensorflow as tf
from src.dataset import DatasetCreator
from src.ddim import DDIM

parser = argparse.ArgumentParser()
# Mode to run code
parser.add_argument("--mode", default=None, type=str, choices=["train", "upscale"],
                    help="Mode of using the model")
# Arguments for training
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--height", default=128, type=int, help="Up-scaled image height")
parser.add_argument("--width", default=128, type=int, help="Up-scaled image width")
parser.add_argument("--channels", default=3, type=int, help="Up-scaled image channels")
parser.add_argument("--cnn_channels", default=32, type=int, help="CNN channels in the first stage")
parser.add_argument("--downscale", default=2, type=int, help="Downscale factor")
parser.add_argument("--stages", default=4, type=int, help="UNet scaling stages")
parser.add_argument("--stage_blocks", default=2, type=int, help="ResNet blocks per stage")
parser.add_argument("--ema", default=0.999, type=float, help="Exponential moving average momentum")
parser.add_argument("--batch_size", default=64, type=int, help="Size of batch of the train set")
parser.add_argument("--epoch_batches", default=1000, type=int, help="Number of images per one epoch")
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs")
parser.add_argument("--log_folder", default="training_2x_to_128", type=str, help="Folder to save weights")
parser.add_argument("--dataset_folder", default="/projects/SkyGAN/clouds_fisheye/processed", type=str,
                    help="Folder with training data")

# Arguments for upscaling
parser.add_argument("--src_folder", default="low_res_64", type=str, help="Folder with images to upscale")
parser.add_argument("--dst_folder", default="sr_images_128", type=str, help="Folder to save results")
parser.add_argument("--upscale", default=2, type=int, choices=[4, 8], help="Upscale factor")
parser.add_argument("--weights_path", default="/home/s_gladkykh/super-resolution/training_2x_to_128/cp.ckpt", type=str,
                    help="Upscale factor")
parser.add_argument("--resolution", default=(64, 64), type=tuple, choices=[(64, 64)],
                    help="Starting resolution of the images to upscale")

args = parser.parse_args([] if "__file__" not in globals() else None)

if args.mode == "train":

    # Set random seed
    tf.keras.utils.set_random_seed(args.seed)

    # Create dataset from folder
    dataset = DatasetCreator(folder_path="/projects/SkyGAN/clouds_fisheye/processed",
                             height=args.height,
                             width=args.width,
                             channels=args.channels).dataset
    # Initialize the model
    ddim = DDIM(args, dataset)

    # Prepare training data
    train = dataset.shuffle(10 * args.batch_size, seed=args.seed)
    train = train.skip(100).repeat()
    train = train.batch(args.batch_size)
    train = train.prefetch(tf.data.AUTOTUNE)

    # Compile the model
    ddim.compile(
        optimizer=tf.optimizers.experimental.AdamW(jit_compile=False),
        loss=tf.losses.MeanAbsoluteError(),
    )

    # Create checkpoint after each epoch
    checkpoint_path = os.path.join(args.log_folder, "cp.ckpt")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    # Train the model
    ddim.fit(
        train,
        epochs=args.epochs,
        steps_per_epoch=args.epoch_batches,
        callbacks=[cp_callback]
    )
else:
    # Load the data from the folder to the tensor
    image_tensors = []
    for filename in os.listdir(args.src_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image = tf.io.read_file(os.path.join(args.src_folder, filename))
            image = tf.image.decode_image(image, channels=3)
            image = tf.image.resize(image, args.resolution)
            image_tensors.append(image)
    stacked_image_tensors = tf.stack(image_tensors)

    # Initialize the model
    ddim = DDIM(args, stacked_image_tensors)

    # Load the weights from the folder
    ddim.load_weights(args.weights_path)

    # Create the initial random noise
    noise = tf.random.normal(
        [stacked_image_tensors.shape[0], args.resolution[0] * args.upscale, args.resolution[1] * args.upscale, 3],
        seed=args.seed
    )
    # Generate the results
    results = ddim.generate(noise, stacked_image_tensors, 1000)
    results = tf.cast(results, tf.uint8)

    # Create directory if not exists
    os.makedirs(args.dst_folder, exist_ok=True)

    # Save generated images to a folder
    for i, image_tensor in enumerate(results):
        encoded_image = tf.image.encode_jpeg(image_tensor)
        filename = os.path.join(args.dst_folder, f'{i}.png')
        with tf.io.gfile.GFile(filename, 'wb') as f:
            f.write(encoded_image.numpy())
