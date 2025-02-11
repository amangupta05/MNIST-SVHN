# MNIST-SVHN multi-modal model specification
import os

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from numpy import sqrt, prod
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid

from vis import plot_embeddings, plot_kls_df
from .mmvae import MMVAE
from .vae_mnist import MNIST
from .vae_svhn import SVHN


class ResampleDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        return self.dataset[index]


class MultiModalDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        return tuple(dataset[idx] for dataset in self.datasets)


class MNIST_SVHN(MMVAE):
    def __init__(self, params):
        super(MNIST_SVHN, self).__init__(dist.Laplace, params, MNIST, SVHN)
        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.vaes[0].llik_scaling = prod(self.vaes[1].data_size) / prod(self.vaes[0].data_size) \
            if params.llik_scaling == 0 else params.llik_scaling
        self.modelName = 'mnist-svhn'

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda'):
        if not (os.path.exists('../data/train-ms-mnist-idx.pt')
                and os.path.exists('../data/train-ms-svhn-idx.pt')
                and os.path.exists('../data/test-ms-mnist-idx.pt')
                and os.path.exists('../data/test-ms-svhn-idx.pt')):
            raise RuntimeError('Generate transformed indices with the script in bin')
        # get transformed indices
        t_mnist = torch.load('../data/train-ms-mnist-idx.pt')
        t_svhn = torch.load('../data/train-ms-svhn-idx.pt')
        s_mnist = torch.load('../data/test-ms-mnist-idx.pt')
        s_svhn = torch.load('../data/test-ms-svhn-idx.pt')

        # load base datasets
        t1, s1 = self.vaes[0].getDataLoaders(batch_size, shuffle, device)
        t2, s2 = self.vaes[1].getDataLoaders(batch_size, shuffle, device)

        train_mnist = ResampleDataset(t1.dataset, t_mnist)
        train_svhn = ResampleDataset(t2.dataset, t_svhn)
        test_mnist = ResampleDataset(s1.dataset, s_mnist)
        test_svhn = ResampleDataset(s2.dataset, s_svhn)

        train_dataset = MultiModalDataset([train_mnist, train_svhn])
        test_dataset = MultiModalDataset([test_mnist, test_svhn])

        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train_loader, test_loader

    def generate(self, runPath, epoch):
        N = 64
        samples_list = super(MNIST_SVHN, self).generate(N)
        for i, samples in enumerate(samples_list):
            samples = samples.data.cpu()
            # wrangle things so they come out tiled
            samples = samples.view(N, *samples.size()[1:])
            save_image(samples,
                       '{}/gen_samples_{}_{:03d}.png'.format(runPath, i, epoch),
                       nrow=int(sqrt(N)))

    def reconstruct(self, data, runPath, epoch):
        recons_mat = super(MNIST_SVHN, self).reconstruct([d[:8] for d in data])
        for r, recons_list in enumerate(recons_mat):
            for o, recon in enumerate(recons_list):
                _data = data[r][:8].cpu()
                recon = recon.squeeze(0).cpu()
                # resize mnist to 32 and colour. 0 => mnist, 1 => svhn
                _data = _data if r == 1 else resize_img(_data, self.vaes[1].data_size)
                recon = recon if o == 1 else resize_img(recon, self.vaes[1].data_size)
                comp = torch.cat([_data, recon])
                save_image(comp, '{}/recon_{}x{}_{:03d}.png'.format(runPath, r, o, epoch))

    def analyse(self, data, runPath, epoch):
        zemb, zsl, kls_df = super(MNIST_SVHN, self).analyse(data, K=10)
        labels = ['Prior', *[vae.modelName.lower() for vae in self.vaes]]
        plot_embeddings(zemb, zsl, labels, '{}/emb_umap_{:03d}.png'.format(runPath, epoch))
        plot_kls_df(kls_df, '{}/kl_distance_{:03d}.png'.format(runPath, epoch))


def resize_img(img, refsize):
    return F.pad(img, (2, 2, 2, 2)).expand(img.size(0), *refsize)
