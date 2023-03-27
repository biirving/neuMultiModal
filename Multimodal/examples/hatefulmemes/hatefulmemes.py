from transformers import ViTFeatureExtractor, ViTModel
# https://huggingface.co/transformers/v4.5.1/model_doc/vit.html
# https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
from transformers import BartTokenizer, BartModel
from PIL import Image
import requests
import torch
from torch import nn

from MLP import MLP

from datasets import load_dataset
import pandas as pd
import cv2

from tqdm import tqdm

import os
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

cuda = torch.device('cuda')

dataset = load_dataset("neuralcatcher/hateful_memes")
data = pd.DataFrame(dataset['train'])
print("count", data['label'].value_counts())


# Image Models
feature_extractor_img = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model_img = ViTModel.from_pretrained('google/vit-base-patch16-224')

# Text Models
tokenizer_txt = BartTokenizer.from_pretrained('facebook/bart-large')
model_txt = BartModel.from_pretrained('facebook/bart-large')

def get_embeddings_txt(text, tokenizer, model):

    text_feats = tokenizer(text, return_tensors="pt")
    outputs_txt = model(**text_feats)
    last_hidden_states_txt = outputs_txt.last_hidden_state

    last_hidden_states_txt = torch.mean(last_hidden_states_txt,1)

    return last_hidden_states_txt

def get_embeddings_img(image, feature_extractor, model):

    inputs_img = feature_extractor(images=image, return_tensors="pt")
    
    outputs_img = model(**inputs_img)
    last_hidden_states_img = outputs_img.last_hidden_state

    last_hidden_states_img = torch.mean(last_hidden_states_img,1)

    return last_hidden_states_img

   
def extract_fused_embeddings(each, data):
    image = cv2.imread("data/"+data['img'][each])
    # print("*****",image)
    text = data['text'][each]

    last_hidden_states_img = get_embeddings_img(image, feature_extractor_img, model_img).cuda()
    last_hidden_states_txt = get_embeddings_txt(text, tokenizer_txt, model_txt).cuda()
    

    # if(each == 0):
    final_img_embeddings = last_hidden_states_img
    final_txt_embeddings = last_hidden_states_txt
    # else:
    #     final_img_embeddings = torch.cat((final_img_embeddings, last_hidden_states_img), 0)
    #     final_txt_embeddings = torch.cat((final_txt_embeddings, last_hidden_states_txt), 0) 


    fuse = torch.cat((final_img_embeddings,final_txt_embeddings),1).cuda()
    # fuse = fuse.view(-1,1)
    

    fuse = torch.unsqueeze(fuse, 0).cuda()

    return fuse

def accuracy(truth, pred):
    return metrics.accuracy_score(truth, pred)

head = MLP(in_channels=1792,num_classes=1).cuda()

criterion = nn.BCELoss()
optimizer = torch.optim.RMSprop([p for p in head.parameters() if p.requires_grad])
 
for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    true = []
    pred = []
    for i in tqdm(range(data.shape[0])):
        # get the inputs; data is a list of [inputs, labels]
        inputs = extract_fused_embeddings(i, data).cuda()
        labels = torch.tensor([float(data.loc[i,"label"])])
        labels = torch.unsqueeze(labels,0).cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = head(inputs).cuda()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        # print(outputs.round().cpu().detach().numpy()[0][0])

        # break

        true.append(labels.cpu().detach().numpy()[0][0])
        if outputs.cpu().detach().numpy()[0][0] >= 0.5:
            pred.append(1)
        else:
            pred.append(0)
        # pred.append()
        # print(true, pred)

        


        if i % 50 == 0:    # print every 2000 mini-batches
            # print(true, pred)
            acc = accuracy(true, pred)
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss/50:.3f} acc: {acc}')
            running_loss = 0.0

print('Finished Training')

pd.DataFrame(pred).to_csv("Pred_fuse.csv")
pd.DataFrame(acc).to_csv("Acc_fuse.csv")


# print(last_hidden_states.shape)

