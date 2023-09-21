## Image Super-Resolution Documentation

### Purpose
This project can be used to train and generate images with high resolution given low-resolution ones utilizing Denoising Diffusion Implicit Model (DDIM)

### Structure and use
The `src` folder contains two files:
- `ddim.py`: has a class of the model, support functions and layers such as: Timestamp embeddings and ResNet blocks. 
- `dataset.py`: has a class DatasetCreator that creates dataset from the folder and preprocesses it for training

with the model itself (`ddim.py`) and dataset loader (`dataset.py`).

The `main.py` is a starting point of the project. From this file you can run both training and upscaling modes.

To specify what exactly do you want to do, you can pass an argument:
- `--mode`: train/upscale

Train parameters:
- `--seed`: random seed
- `--height`: height of the high-resolution image you want to achieve
- `--width`: width of the high-resolution image you want to achieve
- `--width`: number of channels of the high-resolution image you want to achieve
- `--cnn_channels`: number of CNN channels in the first stage
- `--downscale`: factor of downscaling high-res images
- `--stages`: UNet scaling stages
- `--stage_blocks`: RexNet blocks for each stage
- `--ema`: Exponential moving average momentum
- `--batch_size`: Size of the batch for training
- `--epoch_batches`: Batches per one epoch of training
- `--epochs`: Number of training epochs
- `--log_folder`: Folder to save weights after each epoch
- `--dataset_folder`: Folder with data for training

Upscale parameters:
- `--src_folder`: folder with images that you want to upscale
- `--dst_folder`: folder to save super-resolution results
- `--upscale`: upscaling factor (equals to`--downscale`)
- `--weights_path`: path to weights of the model to use
- `--resolution`: tuple of height and width of images you want to upscale

An output of the `upscale` mode is a newly created folder with upscaled images that you provided.


**Note**: Many paramers are set to reasonable default values. If you don't want to dive deeper into the architecture and simply train and use the model you can set the most important parameters such as: downscale factor, resolution of the image, epochs, wights folder and dataset folder, etc.

**Remark**: even for upscaling most of the train parameters are crucial. They are needed to initialize the model correctly for your data.

