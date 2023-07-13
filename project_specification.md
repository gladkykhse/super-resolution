## Image super-resolution (project specification)
**Idea:**

Explore the latest and the most powerful generative conditional Image Super-Resolution techniques based on the Demonising Diffusion Probabilistic Models (DDPM). Implement from scratch the Image Super-Resolution via Iterative Refinement. Research the best parameters and approaches to deliver the best possible result for the given data (low resolution sky images). Evaluate models using different metrics for the image quality after super-resolution such as: Peak Signal to Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM).

----

**Project structure:**
1. Data preprocessing and dataset creation module
	* TBD
2. Super-Resolution model
	* tf.keras.Model 
	* Convolution blocks (UNet architecture)
	* Attention mechanism
  * Timestamp embeddings
3. Evaluation and visualisation modules
	* Metrics
	* Upscaled result visualisation

----

OS: any operational system that supports the dependencies below

Programming Language: Python 3.10

Development Environment: PyCharm Professional 2023.1.3

Libraries:
- TensorFlow 2.11.0
- NumPy 1.25.0
