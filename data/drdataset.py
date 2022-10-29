from torch.utils.data import Dataset
from utils.augment import ToTensor, ToFloat, Compose
import json


class DrDataset(Dataset):
    def __init__(self, root, set):
        super()
        self.root = root

        with open(root + 'r') as file:
            data = json.load()

        self.imgs = data['filenames']
        self.labels = data['labels']

        transform = [ToFloat(), ToTensor()]

        self.transform = Compose(transform)

    def __getitem__(self, index):

        image, label = self.imgs[index], self.labels[index]
        image = self.transform(image)
        return image, label, self.imgs[index]

    def __len__(self):
        return self.imgs.__len__()
