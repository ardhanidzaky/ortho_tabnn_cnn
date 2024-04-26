import pandas as pd

from PIL import Image
from tqdm.contrib.concurrent import process_map
from torch.utils.data import Dataset

from .utils import crop_face

class FKGDataset(Dataset):
    def __init__(
        self
        , dataframe
        , target_width
        , target_height
        , transform
        , preload=True
    ):
        self.dataframe = dataframe
        self.transform = transform
        self.target_width = target_width
        self.target_height = target_height
        self.preload = preload

        self.image_list = list(self.dataframe.iloc[:, 0])
        self.label_list = list(self.dataframe.iloc[:, 1])

        if self.preload:
            # Move to RAM for faster training.
            print('Preloading image...')
            self.image_preload = []
            args = []

            for index in range(len(self.dataframe)):
                args.append({
                    'image_path': self.image_list[index]
                })
            self.image_preload = process_map(self._load_image, args, max_workers=30, chunksize=10)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if self.preload:
            image = self.image_preload[idx]

        if self.transform is not None:
            image = self.transform(image)

        label = int(self.label_list[idx])

        return image, label

    def _load_image(self, args):
        image_path = args['image_path']
        image, _ = crop_face(image_path, width=self.target_width, height=self.target_height)
        image = Image.fromarray(image)

        return image
        