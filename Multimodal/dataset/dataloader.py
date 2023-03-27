import os
import json
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from collections import OrderedDict


class HatefulMemesDataset(Dataset):
    def __init__(self, path, dataloader_type, batch_size, shuffle=True, cache_size=1000, data_filepath = None):
        self.path = path
        self.dataloader_type = dataloader_type
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cache_size = cache_size

        self.img_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=(224, 224)),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.data = self.load_data_from_file(filepath = data_filepath) if data_filepath else self.load_data()
        self.cache = {}


    def load_data(self):
        dataset_dict = load_dataset("neuralcatcher/hateful_memes")
        data = pd.DataFrame(dataset_dict[self.dataloader_type])
        if self.shuffle:
            data = data.sample(frac=1).reset_index(drop=True)
        return data

    def load_data_from_file(self, filepath):
        data = self.read_jsonl_file_to_dataframe(filepath)
        if self.shuffle:
            data = data.sample(frac=1).reset_index(drop=True)
        return data

    def read_jsonl_file_to_dataframe(self, filepath):
        # Read the JSON objects from the file into a list
        with open(filepath) as f:
            json_objs = [json.loads(line) for line in f]

        # Convert the list of JSON objects into a DataFrame
        df = pd.DataFrame(json_objs)
        return df


    def __len__(self):
        return len(self.data) // self.batch_size

    
    def __getitem__(self, index):
        if index in self.cache:
            batch_dict = self.cache[index]
        else:
            # Get a new batch of data
            data_batch = self.data.iloc[index * self.batch_size : (index + 1) * self.batch_size, :]

            img_batch = np.array(data_batch['img'])
            text_batch = list(data_batch['text'])
            output_batch = np.array(data_batch['label']).reshape((self.batch_size,-1))
            img_batch = []

            for i, img_path in enumerate(data_batch['img']):
                img = cv2.imread(self.path + img_path)
                img = self.img_transforms(img)
                img = np.array(img).transpose((2,0,1))
                img_batch.append(img)

            batch_dict = {}
            batch_dict["img"] = img_batch
            batch_dict["text"] = text_batch
            batch_dict["output"] = output_batch


            if len(self.cache) == self.cache_size:
                self.cache.pop(next(iter(self.cache)))
            self.cache[index] = batch_dict

        return batch_dict