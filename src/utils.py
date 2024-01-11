from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split


def get_transforms():
    return {"train": transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize((224, 224), antialias=True),
                                         transforms.Normalize((0.5,), (0.5,)),
                                         transforms.RandomPerspective(distortion_scale=0.5, p=0.1),
                                         transforms.RandomRotation(degrees=(0, 180)),
                                         transforms.RandomHorizontalFlip(p=0.5),
                                         transforms.RandomVerticalFlip(p=0.5)]),

            "test": transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((224, 224), antialias=True),
                                        transforms.Normalize((0.5,), (0.5,))])
            }


def load_data(path_data, transform, train_size, val_size):
    dataset = ImageFolder(path_data, transform)
    train_data, val_data = random_split(dataset, [train_size, val_size])
    return train_data, val_data
