
#------------------------- Import Libraries -----------------------------------#

import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from sklearn import metrics
from tqdm import tqdm

from transformers import ViTModel, ViTImageProcessor
# https://huggingface.co/transformers/v4.5.1/model_doc/vit.html
# https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
from transformers import BartTokenizer, BartModel

# export PYTHONPATH=$PYTHONPATH:`pwd`
from dataset.dataloader import HatefulMemesDataset
from preprocessing.embeddings import Embeddings
from models.basic_models import MLP
from config.config import configuration


#-------------------- Initialize Parameters ----------------------------#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.chdir("/work/socialmedia/multimodal_encoder/RAW_DATA/")
path_source_files = "hatefulmemes/"

batch_size = 128

#---------------- Preprocessing Models -----------------------------#

image_processor_pretrained = 'google/vit-base-patch16-224-in21k'
image_model = 'google/vit-base-patch16-224'
# Image Models
img_processor = ViTImageProcessor.from_pretrained(image_processor_pretrained)
model_img = ViTModel.from_pretrained(image_model)
model_img = model_img.to(device)

text_processor_pretrained = 'facebook/bart-large'
text_model = 'facebook/bart-large'
# Text Models
tokenizer_txt = BartTokenizer.from_pretrained(text_processor_pretrained)
model_txt = BartModel.from_pretrained(text_model)
model_txt = model_txt.to(device)


#------------------- Dataloader -------------------#

train_dataset = HatefulMemesDataset(path_source_files, 'train', batch_size, shuffle=True)

# But dont know the true label of Test Data
# images_dir = os.path.join(path_source_files, 'img')
# test_dataset = pd.DataFrame()
# for each_image in os.listdir(images_dir):
#     if idx, each_image not in enumerate(list(data['img'])):
#         test_dataset['img'] = each_img
#         test_dataset['text'] = data['text'].iloc[idx]
#         test_dataset['label'] = data['label'].iloc[idx]


# test_dataset = HatefulMemesDataset(path_source_files, 'test', batch_size, shuffle=True)

#------------------- Metrics -----------------------#

def accuracy(truth, pred):
    return metrics.accuracy_score(truth, pred)

#---------------- Model Definition --------------------#

head = MLP(in_channels=1792,num_classes=1)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam([p for p in head.parameters() if p.requires_grad])

get_embeddings = Embeddings()

epochs = 20
for epoch in range(epochs):
    num_steps = 0
    running_loss = 0.0
    val_running_loss = 0.0
    val_acc = 0
    true = []
    pred = []
    val_true = []
    val_pred = []
    it = iter(train_dataset)

    for i in tqdm(range(len(train_dataset)), desc=f"[Epoch {epoch+1}]",ascii=' >='):
        data = next(it)
        
        img_batch = data['img']
        text_batch = data['text']
        labels = torch.from_numpy(data['output']).to(torch.float32)

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

    # valit = iter(val_dataset)

    # for j in tqdm(range(len(val_dataset))):
    #     val_data = next(valit)
    #     print(val_data)
        
        
    #     val_img_batch = val_data['img']
    #     print(val_img_batch.shape)
    #     val_labels = torch.from_numpy(val_data['output']).to(torch.float32)

    #     val_last_hidden_states_img = get_embeddings_img(val_img_batch, feature_extractor_img, model_img)
    #     val_outputs = head(val_last_hidden_states_img.to(torch.float32))

    #     val_loss = criterion(val_outputs, val_labels)
    #     val_true.append(val_labels.cpu().detach().numpy()[0][0])
    #     val_running_loss += loss.item()

    #     if val_outputs.cpu().detach().numpy()[0][0] >= 0.5:
    #         val_pred.append(1)
    #     else:
    #         val_pred.append(0)


    train_acc = accuracy(true, pred)
    # val_acc = accuracy(val_true, val_pred)
    print(f'Num_steps : {num_steps}, train_loss : {running_loss/num_steps:.3f}, val_loss : {val_running_loss/1:.3f},train_acc : {train_acc}, val_acc : {val_acc}')

print('Finished Training')


























