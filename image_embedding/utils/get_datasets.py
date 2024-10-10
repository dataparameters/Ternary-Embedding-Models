import torch
import torchvision
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import load_dataset


def imagenet_dataloader(data_dir, batch_size, num_workers, split='train'):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if split == 'train':
        train_dataset = torchvision.datasets.ImageNet(
            root=data_dir,
            split='train',
            # train=True,
            # download=False,
            transform=transform_train
        )
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        return train_data_loader

    if split == 'val':
        test_datatset = torchvision.datasets.ImageNet(
            root=data_dir,
            split='val',
            # train=False,
            # download=False,
            transform=transform_test
        )
        test_data_loader = torch.utils.data.DataLoader(
            test_datatset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        return test_data_loader


def cifar10_dataloader(batch_size, num_workers, split='train'):
    dataset = load_dataset("cifar10")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914,0.4822,0.4465], std=[0.2470,0.2435,0.2616])
        ])

    class CustomCIFAR10(torch.utils.data.Dataset):
        def __init__(self, dataset, transform=None):
            self.dataset = dataset
            self.transform = transform
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            image = self.dataset[idx]['img']
            label = self.dataset[idx]['label']
            if self.transform:
                image = self.transform(image)
            return image, label

    if split=='train':
        train_dataset = CustomCIFAR10(dataset['train'], transform=transform)
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    if split=='test':
        test_dataset = CustomCIFAR10(dataset['test'], transform=transform)
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)


def cifar100_dataloader(batch_size, num_workers, split='train'):
    dataset = load_dataset("cifar100")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071,0.4867,0.4408], std=[0.2675,0.2565,0.2761])
        ])

    class CustomCIFAR100(torch.utils.data.Dataset):
        def __init__(self, dataset, transform=None):
            self.dataset = dataset
            self.transform = transform
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            image = self.dataset[idx]['img']
            label = self.dataset[idx]['fine_label']
            if self.transform:
                image = self.transform(image)
            return image, label

    if split=='train':
        train_dataset = CustomCIFAR100(dataset['train'], transform=transform)
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    if split=='test':
        test_dataset = CustomCIFAR100(dataset['test'], transform=transform)
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)