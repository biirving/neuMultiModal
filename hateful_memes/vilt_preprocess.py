# dependencies
from datasets import load_dataset
from transformers import AutoModel
import torchvision.transforms as transforms
from torchvision.io import read_image
import numpy as np
from datasets import Features, ClassLabel, Array3D, Image
import torch
from torch import nn, tensor
import os
from torchmetrics import Accuracy, MatthewsCorrCoef
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForPreTraining
from transformers import ViltProcessor, ViltModel
from PIL import Image
import requests
from tqdm import tqdm
from transformers import Trainer
import math
import csv


os.environ["HF_ENDPOINT"] = "https://huggingface.co"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
@article{DBLP:journals/corr/abs-2005-04790,
  author    = {Douwe Kiela and
               Hamed Firooz and
               Aravind Mohan and
               Vedanuj Goswami and
               Amanpreet Singh and
               Pratik Ringshia and
               Davide Testuggine},
  title     = {The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes},
  journal   = {CoRR},
  volume    = {abs/2005.04790},
  year      = {2020},
  url       = {https://arxiv.org/abs/2005.04790},
  eprinttype = {arXiv},
  eprint    = {2005.04790},
  timestamp = {Thu, 14 May 2020 16:56:02 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2005-04790.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""
train_data = load_dataset("Multimodal-Fatima/Hatefulmemes_train")
test_data = load_dataset("Multimodal-Fatima/Hatefulmemes_test")
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

train_labels = None
train_data_list = []
for train_index in tqdm(range(train_data['train'].num_rows)):
    torch.cuda.empty_cache()

    image = train_data['train'][train_index]['image']
    text = train_data['train'][train_index]['text']
    try:
      # this should be done beforehand
      inputs = processor(image, text, padding = True, return_tensors="pt")
      train_data_list.append(inputs)
      truth = torch.nn.functional.one_hot(torch.tensor(train_data['train'][train_index]['label']), num_classes=2)
      if(train_labels is None):
        train_labels = truth.view(1, 2)
      else:
        train_labels = torch.cat((train_labels, truth.view(1, 2)), dim = 0)
    except ValueError:
      print('error')
      continue
    
torch.save(train_labels, 'train_labels.pt')


# Save the list of dictionaries as CSV
with open('train_data.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=train_data_list[0].keys())
    writer.writeheader()
    writer.writerows(train_data_list)


test_labels = None
test_data_list = []
for test_index in tqdm(range(test_data['test'].num_rows)):
    image = test_data['test'][test_index]['image']
    text = test_data['test'][test_index]['text']
    try:
      # this should be done beforehand
      inputs = processor(image, text, padding = True, return_tensors="pt")
      test_data_list.append(inputs)
      truth = torch.nn.functional.one_hot(torch.tensor(test_data['test'][test_index]['label']), num_classes=2)
      if(test_labels is None):
        test_labels = truth.view(1, 2)
      else:
        test_labels = torch.cat((test_labels, truth.view(1, 2)), dim = 0)
    except ValueError:
      print('error')
      continue
    
torch.save(test_labels, 'test_labels.pt')

# Save the list of dictionaries as CSV
with open('test_data.csv', 'w', newline='') as f:
    writer_test = csv.DictWriter(f, fieldnames=test_data_list[0].keys())
    writer_test.writeheader()
    writer_test.writerows(test_data_list)