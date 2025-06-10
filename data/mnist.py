
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

from data.base import BaseDataset


class MNISTDataset(BaseDataset):
    def __init__(self, data_dir: str, _type: str = "train", **kwargs):
        super().__init__(data_dir, _type)
        self.data = self.load_data(**kwargs)

    def __getitem__(self, index):
        image, label = self.data[index]
        image = image.view(-1)  # Flatten the image to a 1D tensor
        return image, label

    def __len__(self):
        return self.data.__len__()

    def load_data(self, **kwargs) -> datasets.MNIST:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        if self._type in ["train", "val"]:
            data = datasets.MNIST(root="./data/datasets", train=True, download=True, transform=transform)
            val_split = kwargs.get("val_split", 0.2)
            random_state = kwargs.get("random_state", 42)
            train, val = train_test_split(data, test_size=val_split, random_state=random_state)
            return train if self._type == "train" else val
        if self._type == "test":
            return datasets.MNIST(root="./data/datasets", train=False, download=True, transform=transform)
        raise ValueError(f"Expected dataset type: ['train', 'val', 'test'], got {self._type}!")

    def get_alias(self): return "mnist"