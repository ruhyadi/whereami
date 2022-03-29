"""
Dataset module
"""

import os
from random import random
from typing import Optional, Tuple
import numpy as np
from glob import glob
import cv2
from PIL import Image
from tqdm import tqdm
from sklearn import preprocessing

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from pytorch_lightning import LightningDataModule

class LitWhereAmIDataset(LightningDataModule):
    def __init__(
        self,
        data_dir: str = 'data/',
        fpm: int = 20,
        train_val_test_split: Tuple[int, int, int] = (0.8, 0.1, 0.1),
        batch_size: int = 32,
        num_workers: int = 0,    
        ):
        super().__init__()
        # save hyperparamters
        self.save_hyperparameters(logger=False)

        # data transformation
        self.transforms = transforms.Compose([
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAutocontrast(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # data dir
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        # sets
        self.trainsize = train_val_test_split[0]
        self.valsize = train_val_test_split[1]
        self.testsize = train_val_test_split[2]

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = WhereIamDataset(self.hparams.data_dir, self.hparams.fpm, self.transforms)
            trainset = int(self.trainsize * len(dataset))
            valset = int(self.valsize * len(dataset))
            testsize = int(len(dataset) - trainset - valset)
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=[trainset, valset, testsize]
            )
    
    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False
        )

class WhereIamDataset(Dataset):
    def __init__(self, video_path, frame_per_minute, transforms):
        self.dataset_path = video_path
        self.frame_per_minute = frame_per_minute
        self.transforms = transforms
        self.img_list, self.label_list = self.video_to_frame(self.dataset_path, self.frame_per_minute)
        self.one_hot = self.get_labels()


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        # read image
        img = Image.open(self.img_list[index])
        label_idx = self.label_list[index]
        label = self.one_hot[label_idx]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label

    def get_labels(self):
        labels = self.class_list()
        le = preprocessing.LabelEncoder()
        targets = le.fit_transform(labels)
        targets = torch.as_tensor(targets)
        one_hot = torch.nn.functional.one_hot(targets, -1)

        return one_hot

    def class_list(self):
        class_list = [class_ for class_ in os.listdir(self.dataset_path) if not class_.endswith('.md')]

        return class_list

    def video_to_frame(self, video_path, frame_per_minute):
        """Convert video to frame"""
        class_dir = os.listdir(video_path)
        class_dict = {x: i for i, x in enumerate(self.class_list())}

        print('[INFO] Preprocessing Images...')
        for class_ in class_dir:
            files = glob(os.path.join(video_path, class_) + '/*.mp4')
            # list images and labels
            IMAGES = []
            LABELS = []

            for file in files:
                # read video file
                cap = cv2.VideoCapture(file)
                if cap.isOpened() == False:
                    print('[INFO] Cannot read video')

                # get video intrinsic
                cap_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                cap_fps = cap.get(cv2.CAP_PROP_FPS)
                cap_length = int(cap_count / cap_fps) # second

                # get frame per minute from video
                num_frame = int(cap_length/60 * frame_per_minute)
                frame_ids = [int(i) for i in np.linspace(0, cap_count, num_frame)]
                i = 0 # frame index
                id = 0 # id index

                with tqdm(desc=f'Preprocess {class_}', total=cap_count) as pbar:
                    while(cap.isOpened()):
                        ret, frame = cap.read()
                        if ret==True:
                            output_path = os.path.join(video_path, class_)
                            filename = f'{output_path}/{class_}_{id:03d}.png'
                            if i in frame_ids:
                                cv2.imwrite(filename, frame)
                                id = id + 1
                        else:
                            break
                        i = i + 1
                        pbar.update(1)
                    cap.release()

        print('[INFO] Preprocessing Images Finished...')
        # get #frame information
        for class_ in class_dir:
            files = sorted(glob(os.path.join(video_path, class_) + '/*.png'))
            print(f'[INFO] # {class_}: {len(files)} frames')

            img_list = [IMAGES.append(file) for file in files]
            label_list = [LABELS.append(class_dict[file.split('/')[-2]]) for file in files]

        return IMAGES, LABELS

if __name__ == "__main__":
    
    dataset = LitWhereAmIDataset(
        data_dir='/home/didi/Repository/whereami/data',
        fpm=30,
        train_val_test_split=[0.8, 0.1, 0.1]
    )

    print(dataset)

