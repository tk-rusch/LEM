import torch
import torchvision
import torchvision.transforms as transforms

def get_data(bs_train,bs_test):
    train_dataset = torchvision.datasets.CIFAR10(root='../../data/cifar10/',
                                                 train=True,
                                                 transform=transforms.ToTensor(),
                                                 download=True)

    test_dataset = torchvision.datasets.CIFAR10(root='../../data/cifar10/',
                                                train=False,
                                                transform=transforms.ToTensor())

    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [48000,2000])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=bs_train,
                                               shuffle=True)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                              batch_size=bs_test,
                                              shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=bs_test,
                                              shuffle=False)

    return train_loader, valid_loader, test_loader
