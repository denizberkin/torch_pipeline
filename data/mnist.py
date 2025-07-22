
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

from data.base import BaseDataset


class MNISTDataset(BaseDataset):
    """ Loads mnist dataset and defines [] and len built-ins. """
    def __init__(self, data_dir: str, _type: str = "train", **kwargs):
        super().__init__(data_dir, _type)
        self.data = self.load_data(**kwargs)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    def __getitem__(self, index):
        image, label = self.data[index]
        return image.view(-1), label

    def __len__(self): return self.data.__len__()

    def load_data(self, **kwargs) -> datasets.MNIST:
        if self._type in ["train", "val"]:
            data = datasets.MNIST(root="./data/datasets", train=True, download=True, transform=self.transform)
            val_split = kwargs.get("val_split", 0.2)
            random_state = kwargs.get("random_state", 1337)
            train, val = train_test_split(data, test_size=val_split, random_state=random_state)
            return train if self._type == "train" else val
        if self._type == "test":
            return datasets.MNIST(root="./data/datasets", train=False, download=True, transform=self.transform)
        raise ValueError(f"Expected dataset type: ['train', 'val', 'test'], got {self._type}!")

    def get_alias(self): return "mnist"