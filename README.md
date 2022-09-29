# LOPR: Latent Occupancy PRediction using Generative Models

This is the official implementation of LOPR used in "LOPR: Latent Occupancy PRediction using Generative Models" ([arXiv](google.com)).

## Abstract:
Environment prediction frameworks are essential for autonomous vehicles to facilitate safe maneuvers in a dynamic environment. Previous approaches have used occupancy grid maps as a bird’s eye-view representation of the scene and optimized the prediction architectures directly in pixel space. Although these methods have had some success in spatiotemporal prediction, they are, at times, hindered by unrealistic and incorrect predictions. We postulate that the quality and realism of the forecasted occupancy grids can be improved with the use of generative models. We propose a framework that decomposes occupancy grid prediction into task-independent low-dimensional representation learning and task-dependent prediction in the latent space. We demonstrate that our approach achieves state-of-the-art performance on the real-world autonomous driving dataset, NuScenes.

## Visualization

Below, we visualize two examples of predictions. For more results, check out our media folder and our preprint.

![](media/pred_1.gif)
![](media/pred_4.gif)

## Setup

- Python 3.7.10
- Libraries: PyTorch (1.7.1) + ...
- Tested on Ubuntu 20.04 + Nvidia RTX TITAN

## Training

1. Task-independent Representation Learning

```python
python src/representation_learning/main.py
```

2. Convert the OGM dataset to latent dataset.

```python
python src/dataset/process_dataset_to_latents.py
```

3. Task-dependent Supervised Learning Learning

```python
python src/prediction/train.py
```

## Visualize results

```python
python scripts/visualize.py
```
## References:

For the representation learning, we use the following implementations:
- part of the DriveGAN with encoder, generator, and discriminator implementation by Name et. al. available at: https://github.com/nv-tlabs/DriveGAN_code ([License](https://github.com/nv-tlabs/DriveGAN_code/blob/master/LICENSE))

DriveGAN implementation uses:

- VAE-GAN available at:  https://github.com/rosinality/stylegan2-pytorch ([License](https://github.com/nv-tlabs/DriveGAN_code/blob/master/LICENSE))

- Perceptual Similarity implementation available at: https://github.com/richzhang/PerceptualSimilarity ([License](https://github.com/nv-tlabs/DriveGAN_code/blob/master/LICENSE))

- StyleGAN custom ops available at:  https://github.com/NVlabs/stylegan2 ([License](https://github.com/nv-tlabs/DriveGAN_code/blob/master/LICENSE))

If you use this work, please cite:
```
ARXIV REFERENCE
```


