# MNIST-SVHN Index Matching Script
import torch
from torchvision import datasets, transforms

def match_svhn_mnist_indices(mnist_labels, mnist_indices, svhn_labels, svhn_indices, max_d=10000, dm=30):
    """
    Matches indices of MNIST and SVHN labels.

    Args:
        mnist_labels: Sorted MNIST labels.
        mnist_indices: Indices of sorted MNIST labels in original list.
        svhn_labels: Sorted SVHN labels.
        svhn_indices: Indices of sorted SVHN labels in original list.
        max_d (int): Maximum number of datapoints per class. Default is 10000.
        dm (int): Data multiplier for random permutations to match. Default is 30.

    Returns:
        A tuple containing the matched indices for MNIST and SVHN.
    """
    matched_mnist_indices, matched_svhn_indices = [], []
    for label in mnist_labels.unique():  # assuming both have same indices
        mnist_label_indices = mnist_indices[mnist_labels == label]
        svhn_label_indices = svhn_indices[svhn_labels == label]
        num_samples = min(mnist_label_indices.size(0), svhn_label_indices.size(0), max_d)
        mnist_label_indices = mnist_label_indices[:num_samples]
        svhn_label_indices = svhn_label_indices[:num_samples]
        for _ in range(dm):
            matched_mnist_indices.append(mnist_label_indices[torch.randperm(num_samples)])
            matched_svhn_indices.append(svhn_label_indices[torch.randperm(num_samples)])
    return torch.cat(matched_mnist_indices), torch.cat(matched_svhn_indices)

if __name__ == '__main__':
    max_d = 10000  # maximum number of datapoints per class
    dm = 30        # data multiplier: random permutations to match

    # Define the transform
    transform = transforms.ToTensor()

    # Load the datasets
    train_mnist = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_mnist = datasets.MNIST('../data', train=False, download=True, transform=transform)
    train_svhn = datasets.SVHN('../data', split='train', download=True, transform=transform)
    test_svhn = datasets.SVHN('../data', split='test', download=True, transform=transform)

    # Process SVHN labels
    train_svhn.labels = torch.LongTensor(train_svhn.labels.squeeze().astype(int)) % 10
    test_svhn.labels = torch.LongTensor(test_svhn.labels.squeeze().astype(int)) % 10

    # Match and save training indices
    mnist_labels, mnist_indices = train_mnist.targets.sort()
    svhn_labels, svhn_indices = train_svhn.labels.sort()
    train_idx1, train_idx2 = match_svhn_mnist_indices(mnist_labels, mnist_indices, svhn_labels, svhn_indices, max_d=max_d, dm=dm)
    print(f'Number of training indices: MNIST={len(train_idx1)}, SVHN={len(train_idx2)}')
    torch.save(train_idx1, '../data/train-ms-mnist-idx.pt')
    torch.save(train_idx2, '../data/train-ms-svhn-idx.pt')

    # Match and save test indices
    mnist_labels, mnist_indices = test_mnist.targets.sort()
    svhn_labels, svhn_indices = test_svhn.labels.sort()
    test_idx1, test_idx2 = match_svhn_mnist_indices(mnist_labels, mnist_indices, svhn_labels, svhn_indices, max_d=max_d, dm=dm)
    print(f'Number of test indices: MNIST={len(test_idx1)}, SVHN={len(test_idx2)}')
    torch.save(test_idx1, '../data/test-ms-mnist-idx.pt')
    torch.save(test_idx2, '../data/test-ms-svhn-idx.pt')
