## Image Super-Resolution Project Specification
### Idea:

Explore the latest and the most powerful generative conditional Image Super-Resolution techniques based on the Demonising Diffusion. Implement from scratch the Image Super-Resolution using the before Denoising Diffusion Implicit Model (DDIM). Research the best parameters and approaches to deliver the best possible result for the given data (low resolution sky images). Evaluate models using different metrics for the image quality after super-resolution such as: Peak Signal to Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM).

----

### Project structure:

In the project I will follow the paper: https://arxiv.org/abs/2010.02502

The modules below need to be implemented

1. Data preprocessing and dataset creation module
2. Super-Resolution model (conditional DDIM) with all needed components such as: embeddings, convolution blocks, attention mechanism, etc.
3. Main part that allows to run both training and upscaling with defined parameters

----

### Project from the user's side

**Input:**
- path to a folder with input images
- path to a folder with resulting upscaled images
- upscale factor
- path to weights of the model
- staring resolution

**Output:**
- a folder with the upscaled input images

----

OS: any operational system that supports the dependencies below

Programming Language: Python 3.10

Development Environment: PyCharm Professional 2023.1.3

Libraries:
- TensorFlow 2.13.0
- NumPy 1.24.3
