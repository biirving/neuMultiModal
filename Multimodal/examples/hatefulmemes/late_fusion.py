# export PYTHONPATH=$PYTHONPATH:`pwd`
#------------------------- Import Libraries -----------------------------------#

import os
import torch
import pandas as pd
from torch import nn

from dataset.dataloader import HatefulMemesDataset
from preprocessing.embeddings import Embeddings
from models.basic_models import MLP
# from config.config import configuration
from supervised.train import *
from supervised.plots import *

#-------------------- Initialize Parameters ----------------------------#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
home_path = os.getcwd()
os.chdir("/work/socialmedia/multimodal_encoder/RAW_DATA/")

#------------------- Dataloader -------------------#

train_dataset = HatefulMemesDataset(path= configuration['Dataset']['path_to_data'], 
                                    dataloader_type= 'train', 
                                    batch_size= configuration['Dataset']['batch_size'], 
                                    shuffle=configuration['Dataset']['shuffle'],
                                    cache_size= configuration['Dataset']['cache_size'],
                                    data_filepath= None)


val_dataset = HatefulMemesDataset(path= configuration['Dataset']['path_to_data'], 
                                    dataloader_type= 'validation', 
                                    batch_size= configuration['Dataset']['batch_size'], 
                                    shuffle=configuration['Dataset']['shuffle'],
                                    cache_size= configuration['Dataset']['cache_size'],
                                    data_filepath="hatefulmemes/dev.jsonl")

test_dataset = HatefulMemesDataset(path= configuration['Dataset']['path_to_data'], 
                                    dataloader_type= 'test', 
                                    batch_size= configuration['Dataset']['batch_size'], 
                                    shuffle=configuration['Dataset']['shuffle'],
                                    cache_size= configuration['Dataset']['cache_size'],
                                    data_filepath="hatefulmemes/test.jsonl")

#---------------- Model Definition --------------------#

head = configuration['Models']['mlp'].to(device)
criterion = nn.BCELoss().to(device)
optimizer = torch.optim.Adam([p for p in head.parameters() if p.requires_grad])

get_embeddings = Embeddings()
model_head, train_metrics = train(train_dataset, val_dataset, configuration, device, get_embeddings, head, criterion, optimizer)
print('Finished Training')

# test_metrics = test(model_head, test_dataset, configuration, device)
# print('Finished Testing')

train_metrics = pd.DataFrame(train_metrics)
# test_metrics = pd.DataFrame(test_metrics)

print("Metrics: ")
print(train_metrics)
# print(test_metrics)

# Plots
os.chdir(home_path)
plots(train_metrics, f'hatefulmemes_{idx}')





























