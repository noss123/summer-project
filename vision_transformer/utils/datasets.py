import os
import torch
from torchvision import datasets
# we're going to use v2 over the old transforms module
# it requires explicit conversion, to help avoid silent errors with toTensor()
# also natively supports batch-level augmentations and complex transformations
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def get_transforms(imagedim):
    train_transform = v2.Compose([
        v2.RandomResizedCrop(imagedim, antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandAugment(num_ops=2, magnitude=9),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        # normalise using mean and stddev of ImageNet
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = v2.Compose([
        v2.Resize(imagedim+32, antialias=True),
        v2.CenterCrop(imagedim),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        # normalise using mean and stddev of ImageNet
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, test_transform

def setup_dataloaders(directory, imagedim, batchsize, workers, seed, is_distributed=False):
    train_transform, test_transform = get_transforms(imagedim)

    train_dataset = datasets.Imagenette(root=directory, split='train', size='full', download=True, transform=train_transform)
    test_dataset = datasets.Imagenette(root=directory, split='val', size='full', download=True, transform=test_transform)

    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=seed)
        test_sampler = DistributedSampler(test_dataset, shuffle=False, seed=seed)
        shuffle_train = False
    else:
        train_sampler = None
        test_sampler = None
        shuffle_train = True

    # allows us to shuffle train dataset if not using the distributed sampler
    # cannot explicitly set shuffle to True if using a sampler
    trainloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=shuffle_train, 
                             num_workers=workers, pin_memory=True, sampler=train_sampler)
    testloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False,
                            num_workers=workers, pin_memory=True, sampler=test_sampler)

    # return train sampler so as to call set_epoch() later
    # this ensures each epoch uses a different random seed for shuffling train data
    # prevents model from developing some sequential bias
    return trainloader, testloader, train_sampler