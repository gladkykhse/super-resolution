## Image Super-Resolution Project Specification
### Idea:

Explore the latest and the most powerful generative conditional Image Super-Resolution techniques based on the Demonising Diffusion Probabilistic Models (DDPM). Implement from scratch the Image Super-Resolution via Iterative Refinement. Research the best parameters and approaches to deliver the best possible result for the given data (low resolution sky images). Evaluate models using different metrics for the image quality after super-resolution such as: Peak Signal to Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM).

----

### Project structure:

In the project I will loosely follow the paper: https://arxiv.org/abs/2104.07636

The modules below need to be implemented

1. Data preprocessing and dataset creation module
2. Super-Resolution model (SR3) with all needed components such as: embeddings, convolution blocks, attention mechanism, etc.
3. Evaluation and visualisation module will consist of metrics and functions for the representation of the upscaled image and the original one
4. Utils module will contain useful functions such as: pano to fisheye and reverse, image rescaling, etc.

----

### Project from the user's side

**Input:**
- path to a folder with either fisheye or panorama images
- type of the input (fisheye/panorama)
- type of the output (fisheye/panorama)

**Output:**
- a folder with the upscaled input images of the type provided by user

----

OS: any operational system that supports the dependencies below

Programming Language: Python 3.10

Development Environment: PyCharm Professional 2023.1.3

Libraries:
- TensorFlow 2.11.0
- NumPy 1.25.0
