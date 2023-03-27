# export PYTHONPATH=$PYTHONPATH:`pwd`
#------------------------- Import Libraries -----------------------------------#

import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from sklearn import metrics
from tqdm import tqdm
import random

from torch.utils.data import DataLoader

from dataset.dataloadercopy import HatefulMemesDataset
from preprocessing.embeddings import Embeddings
from models.basic_models import MLP
from config.config import configuration

#-------------------- Initialize Parameters ----------------------------#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.chdir("/work/socialmedia/multimodal_encoder/RAW_DATA/")



#---------------- Preprocessing Models -----------------------------#
# Image Models
img_processor = configuration['Models']['image_processor']
model_img = configuration['Models']['image_model'].to(device)

# Text Models
tokenizer_txt = configuration['Models']['text_tokenizer']
model_txt = configuration['Models']['text_model'].to(device)

#------------------- Dataloader -------------------#

train_dataset = HatefulMemesDataset(path= configuration['Dataset']['path_to_data'], 
                                    dataloader_type= 'train',
                                    shuffle= configuration['Dataset']['shuffle'],
                                    data_filepath= None)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# for batch in train_dataloader:
#     images, texts, labels = batch['image'], batch['text'], batch['output']
#     print(images.shape, len(texts), labels.shape)
#     exit()

# But dont know the true label of Test Data
# images_dir = os.path.join(path_source_files, 'img')
# test_dataset = pd.DataFrame()
# for each_image in os.listdir(images_dir):
#     if idx, each_image not in enumerate(list(data['img'])):
#         test_dataset['img'] = each_img
#         test_dataset['text'] = data['text'].iloc[idx]
#         test_dataset['label'] = data['label'].iloc[idx]


val_dataset = HatefulMemesDataset(path= configuration['Dataset']['path_to_data'], 
                                    dataloader_type= 'validation', 
                                    shuffle=configuration['Dataset']['shuffle'],
                                    data_filepath="hatefulmemes/dev.jsonl")

val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

#------------------- Metrics -----------------------#

def accuracy(truth, pred):
    return metrics.accuracy_score(truth, pred)

#---------------- Model Definition --------------------#

head = configuration['Models']['mlp'].to(device)
criterion = nn.BCELoss().to(device)
optimizer = torch.optim.Adam([p for p in head.parameters() if p.requires_grad])

get_embeddings = Embeddings()

for epoch in range(configuration['Hyperparameters']['epochs']):
    num_steps = 0
    val_num_steps = 0
    running_loss = 0.0
    val_running_loss = 0.0
    true = []
    pred = []
    val_true = []
    val_pred = []

    # for i in tqdm(range(len(train_dataset)), desc=f"[Epoch {epoch+1}]",ascii=' >='):
    for batch in train_dataloader:
        img_batch = batch['img']
        text_batch = batch['text']
        labels = batch['output'].to(torch.float32).to(device)

        last_hidden_states_img = get_embeddings.get_embeddings_img(img_batch, img_processor, model_img)
        last_hidden_states_txt = get_embeddings.get_embeddings_txt(text_batch, tokenizer_txt, model_txt)
        fused_embeddings = get_embeddings.extract_fused_embeddings(last_hidden_states_img, last_hidden_states_txt)

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = head(fused_embeddings.to(torch.float32))
        # loss = criterion(outputs, labels.to(torch.float32))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        # print(outputs.round().cpu().detach().numpy()[0][0])
        true.append(labels.cpu().detach().numpy()[0][0])
        if outputs.cpu().detach().numpy()[0][0] >= 0.5:
            pred.append(1)
        else:
            pred.append(0)
        
        num_steps +=1

    with torch.no_grad():
        # for j in tqdm(range(len(val_dataset)), desc=f"[Epoch {epoch+1}]",ascii=' >='):
        for batch in val_dataloader:
            val_img_batch = batch['img']
            val_text_batch = batch['text']
            val_labels = batch['output'].to(torch.float32).to(device)

            val_last_hidden_states_img = get_embeddings.get_embeddings_img(val_img_batch, img_processor, model_img)
            val_last_hidden_states_txt = get_embeddings.get_embeddings_txt(val_text_batch, tokenizer_txt, model_txt)
            val_fused_embeddings = get_embeddings.extract_fused_embeddings(val_last_hidden_states_img, val_last_hidden_states_txt)

            val_outputs = head(val_fused_embeddings.to(torch.float32))

            val_loss = criterion(val_outputs, val_labels)
            val_true.append(val_labels.cpu().detach().numpy()[0][0])
            val_running_loss += val_loss.item()

            if val_outputs.cpu().detach().numpy()[0][0] >= 0.5:
                val_pred.append(1)
            else:
                val_pred.append(0)
            
            val_num_steps +=1


    train_acc = accuracy(true, pred)
    val_acc = accuracy(val_true, val_pred)
    print(f'Num_steps : {num_steps}, train_loss : {running_loss/num_steps:.3f}, val_loss : {val_running_loss/val_num_steps:.3f},train_acc : {train_acc}, val_acc : {val_acc}')

print('Finished Training')


























