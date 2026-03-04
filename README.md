# VQ-VAE PyTorch

A PyTorch implementation of Vector Quantized Variational Autoencoder (VQ-VAE) trained on CIFAR-10.

## Architecture

- **Encoder**: Convolutional network that downsamples 32×32 images to 8×8 latent maps
- **Vector Quantizer**: Maps continuous latent vectors to the nearest entry in a learned codebook via straight-through estimator
- **Decoder**: Transposed convolutional network that reconstructs the original image from quantized latents

## Loss

```
loss = recon_loss + vq_loss + β * commitment_loss
```

- `recon_loss`: MSE between input and reconstruction
- `vq_loss`: MSE between codebook vectors and encoder output (trains codebook)
- `commitment_loss`: MSE between encoder output and codebook vectors (trains encoder)

## Usage

```bash
python main.py
```

## References

- [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) (van den Oord et al., 2017)
