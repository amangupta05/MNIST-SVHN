from .mmvae_mnist_svhn import MNIST_SVHN as VAE_mnist_svhn
from .vae_mnist import MNIST as VAE_mnist
from .vae_svhn import SVHN as VAE_svhn

__all__ = [VAE_mnist_svhn, VAE_mnist, VAE_svhn]