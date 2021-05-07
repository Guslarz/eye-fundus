import os
import numpy as np
from sys import stdout
from urllib.request import urlretrieve
from zipfile import ZipFile


class Dataset:
    @property
    def filenames(self):
        raise NotImplementedError()

    @staticmethod
    def download_progress(count, block_size, total_size):
        stdout.write("\rDownload: %.1f%%" % (float(count * block_size) / total_size * 100.0))
        stdout.flush()

    @staticmethod
    def list_directory(path):
        images_filenames = sorted(os.listdir(path))
        return [*map(lambda filename: os.path.join(path, filename), images_filenames)]

    @staticmethod
    def extract_name(path):
        return path.split('\\')[-1].split('.')[0]


class HRFDataset(Dataset):
    DATASET_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'hrfdataset')
    DATASET_URL = "https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip"

    def __init__(self, force_download=False):
        if not HRFDataset.__dir_exists() or force_download:
            HRFDataset.__download()

    @property
    def filenames(self):
        image_paths = Dataset.list_directory(os.path.join(HRFDataset.DATASET_DIR, 'images'))
        mask_paths = Dataset.list_directory(os.path.join(HRFDataset.DATASET_DIR, 'masks'))
        for image_path, mask_path in zip(image_paths, mask_paths):
            name = Dataset.extract_name(image_path)
            yield name, image_path, mask_path

    @staticmethod
    def __dir_exists():
        return os.path.exists(HRFDataset.DATASET_DIR)

    @staticmethod
    def __download():
        if not HRFDataset.__dir_exists():
            os.makedirs(HRFDataset.DATASET_DIR)
        path, _ = urlretrieve(HRFDataset.DATASET_URL, reporthook=Dataset.download_progress)
        HRFDataset.__unzip(path)
        os.remove(path)

    @staticmethod
    def __unzip(path):
        zipfile = ZipFile(path, 'r')
        zipfile.extractall(HRFDataset.DATASET_DIR)
        zipfile.close()


class DataLoader:
    def load(self, path):
        raise NotImplementedError()


class DataGenerator:
    def __init__(self, paths, image_loader, mask_loader):
        self.__paths = paths
        self.__image_loader = image_loader
        self.__mask_loader = mask_loader

    def __iter__(self):
        for name, image_path, mask_path in self.__paths:
            yield self.__image_loader.load(image_path), self.__mask_loader.load(mask_path)


class DatasetLoader:
    VALIDATION_SIZE = 5
    MAX_SIZE = None

    def __init__(self, dataset, seed=None):
        random_state = np.random.RandomState(seed=seed)
        paths = random_state.permutation(dataset.filenames)
        if DatasetLoader.MAX_SIZE is not None:
            paths = paths[:DatasetLoader.MAX_SIZE]
        self.__train_paths = paths[:DatasetLoader.VALIDATION_SIZE]
        self.__validation_paths = paths[DatasetLoader.VALIDATION_SIZE:]

    def load_training(self, image_loader, mask_loader):
        return DataGenerator(self.__train_paths, image_loader, mask_loader), \
               DataGenerator(self.__validation_paths, image_loader, mask_loader)
