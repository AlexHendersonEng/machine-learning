import numpy as np
from torch.utils.data import Dataset
import os
import requests
import gzip
import shutil


class CustomMNIST(Dataset):
    base_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    files = ['train-images-idx3-ubyte.gz',
             'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz']

    def __init__(self, root, transform=None, train=True, download=True):
        # Assign properties
        self.root = root
        self.transform = transform
        self.train = train
        self.download = download

        # Download data if requires
        if self.download:
            self._download_data()

        # Get training/testing data
        if self.train:
            self.data, self.targets = self._load_data(train=True)
        else:
            self.data, self.targets = self._load_data(train=False)

    def _download_data(self):
        # Make directory if required
        if not os.path.exists(self.root):
            os.mkdir(self.root)

        # Download files
        for file in self.files:
            gz_path = os.path.join(self.root, file)
            out_path = os.path.join(self.root, file.rstrip('.gz'))
            if not os.path.exists(out_path):
                print(f'Downloading {file}...')
                response = requests.get(self.base_url + file, stream=True)
                with open(gz_path, 'wb') as f:
                    f.write(response.content)
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(out_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(gz_path)

    def _load_data(self, train=True):
        # Get images and labels
        if train:
            images = os.path.join(self.root, self.files[0].rstrip('.gz'))
            labels = os.path.join(self.root, self.files[1].rstrip('.gz'))
        else:
            images = os.path.join(self.root, self.files[2].rstrip('.gz'))
            labels = os.path.join(self.root, self.files[3].rstrip('.gz'))

        # Get images rows and columns
        with open(images, 'rb') as image_data:
            n_rows, n_cols = np.frombuffer(image_data.read(), dtype='>i4', offset=8, count=2)
        n_rows, n_cols = int(n_rows), int(n_cols)

        # Read image data
        with open(images, 'rb') as image_data:
            images = np.frombuffer(image_data.read(), dtype=np.uint8, offset=16).reshape(-1, n_rows, n_cols)

        # Read label data
        with open(labels, 'rb') as label_data:
            targets = np.frombuffer(label_data.read(), dtype=np.uint8, offset=8)

        return images, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get image and label
        image, label = self.data[index], self.targets[index]

        # Apply transform to image
        if self.transform:
            image = self.transform(image)

        return image, label

