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



"""
Here is the code for adding a classification layer to the VilBert model
"""
class classificationVILT(torch.nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.projection = nn.Linear(768, 2)
        self.classification = nn.Softmax(dim=1)
    
    def forward(self, input_ids, token_type_ids, attention_mask, pixel_values, pixel_mask):
        outputs = self.bert(input_ids, token_type_ids, attention_mask, pixel_values, pixel_mask)
        pooled_output = outputs[1]
        to_feed = self.projection(pooled_output)
        logits = self.classification(to_feed)
        return logits

class CustomTrainer(Trainer):
    def __init__(self, epochs, lr, train_data, test_data, model, processor):
        self.epochs = epochs
        self.lr = lr
        self.train_data = train_data
        self.test_data = test_data
        self.model = model
        self.processor = processor
        
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.BCELoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def plot(self, loss, epoch):
        timesteps = np.arange(1, loss.shape[0] + 1)
        # Plot the MSE vs timesteps
        plt.plot(timesteps, loss)
        # Add axis labels and a title
        plt.xlabel('Timestep')
        plt.ylabel('BCE Loss')
        plt.title('Loss')
        plt.savefig('loss_' + str(epoch) + '.png')
        plt.close()

    def train(self, batch_size):
        adam = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(adam, self.epochs)
        loss_fct = nn.BCELoss()
        accuracy = Accuracy(task='multiclass', num_classes=2).to(device)
        mcc = MatthewsCorrCoef(task='binary').to(device)
        counter = 0

        training_loss_over_epochs = []
        for epoch in range(self.epochs):
            training_loss = []
            total_acc = 0
            num_train_steps = train_data['train'].num_rows
            runtime = 0
            value = 0
            for train_index in tqdm(range(num_train_steps)):
                model.zero_grad()
                image = train_data['train'][train_index]['image']
                text = train_data['train'][train_index]['text']
                try:
                    inputs = processor(image, text, return_tensors="pt").to(device)
                except ValueError:
                    value += 1
                    continue
                try:
                    out = model(**inputs)
                except RuntimeError:
                    runtime += 1
                    continue
                if(train_data['train'][train_index]['label'] == 0):
                    truth = torch.tensor([0, train_data['train'][train_index]['label']], dtype=float)
                else: 
                    truth = torch.tensor([1, train_data['train'][train_index]['label']], dtype=float)
                loss = loss_fct(out.float().to(device), truth.view(1, 2).float().to(device))
                training_loss.append(loss.item())
                maximums = torch.argmax(out)
                truth_max = torch.argmax(truth)
                # here is the accuracy measurement
                #acc = accuracy(maximums, truth_max)
                total_acc += (maximums == truth_max)
                adam.zero_grad()
                loss.backward()
                adam.step()
            print('runtime errors: ', runtime)
            print('value errors: ', value)
            self.plot(np.array(training_loss), epoch)
            print('\n')
            print('epoch: ', counter)
            counter += 1
            print('training set accuracy: ', total_acc/train_index)
            print('loss total: ', sum(training_loss))
            print('\n')
            training_loss_over_epochs.append(training_loss)
            #exponential.step()
            cosine.step()

            model.eval()

        with torch.no_grad():
            num_correct = 0
            actuals = []
            outputs = []
            num_test_steps = test_data['test'].num_rows
            total_acc = 0
            runtime = 0
            value = 0
            for y in tqdm(range(num_test_steps)):
                image = test_data['test'][y]['image']
                text = test_data['test'][y]['text']
                try:
                    inputs = processor(image, text, return_tensors="pt").to(device)
                except ValueError:
                    value += 1
                    continue
                try:
                    out = model(**inputs)
                except RuntimeError:
                    runtime += 1
                    continue
                if(train_data['train'][train_index]['label'] == 0):
                    truth = torch.tensor([0, train_data['train'][train_index]['label']], dtype=float)
                else: 
                    truth = torch.tensor([1, train_data['train'][train_index]['label']], dtype=float)
                maximums = torch.argmax(out)
                truth_max = torch.argmax(truth)
                total_acc += (maximums == truth_max)
        accuracy = total_acc / y
        print('runtime errors: ', runtime)
        print('value errors: ', value)
        print("Basic accuracy on test set: ", accuracy)
        return training_loss_over_epochs, accuracy


processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
vilt = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
model =  classificationVILT(vilt).to(device)
train = CustomTrainer(1, 1e-3, train_data, test_data, model, processor)
train.train(1)