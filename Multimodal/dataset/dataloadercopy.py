import os
import json
from PIL import Image
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from collections import OrderedDict


class HatefulMemesDataset(Dataset):
    def __init__(self, path, dataloader_type, shuffle=True, data_filepath = None):
        self.path = path
        self.dataloader_type = dataloader_type

        self.img_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=(224, 224)),
            transforms.ToTensor()
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.shuffle = shuffle
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
        return len(self.data)

    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.data.iloc[idx]
        image_path, text, label = row['img'], row['text'], row['label']
        img = Image.open(self.path + image_path)
        # Resize to 224 x 224
        img = img.resize((224, 224))
        img = np.array(img).transpose((2,0,1))
        # img = self.img_transforms(img)
        return {'img': img, 'text': text, 'output': np.array(label)}