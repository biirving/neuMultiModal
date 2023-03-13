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


os.environ["HF_ENDPOINT"] = "https://huggingface.co"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Cite the dataset that you mean to use
"""
# Fill these in with the desired dataset
train_data = None
test_data = None



"""
Here is the code for adding a classification layer to the said model
"""
class classificationModel(torch.nn.Module):
    def __init__(self, base_model, output_size, num_classes):
        super().__init__()
        self.base_model = base_model
        self.projection = nn.Linear(output_size, num_classes)
        self.classification = nn.Softmax(dim=1)
    
    # change this function to meet the needs of your specific model
    def forward(self, input_ids, token_type_ids, attention_mask, pixel_values, pixel_mask):
        outputs = self.base_model(input_ids, token_type_ids, attention_mask, pixel_values, pixel_mask)
        pooled_output = outputs[1]
        to_feed = self.projection(pooled_output)
        logits = self.classification(to_feed)
        return logits

class CustomTrainer(Trainer):
    def __init__(self, epochs, lr, train_data, test_data, model, processor, num_classes, loss_func, optimizer, lr_scheduler):
        self.epochs = epochs
        self.lr = lr
        self.train_data = train_data
        self.test_data = test_data
        self.model = model
        self.processor = processor
        self.num_classes = num_classes
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        

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
        plt.savefig('./loss/loss_' + str(epoch) + '.png')
        plt.close()

    def train(self, batch_size):

        # subject to change
        # Use optimizer, and learning rate scheduler of your choice.
        optimizer = self.optimizer(self.model.parameters(), lr=self.lr)
        cosine = self.lr_scheduler(optimizer, self.epochs)
        loss_fct = self.loss_func
        accuracy = Accuracy(task='multiclass', num_classes=self.num_classes).to(device)
        counter = 0

        training_loss_over_epochs = []
        for epoch in range(self.epochs):
            accuracy = Accuracy("binary").to(device)
            training_loss = []
            total_acc = 0
            num_train_steps = math.floor(train_data['train'].num_rows/batch_size) * batch_size
            runtime = 0
            value = 0
            for train_index in tqdm(range(0, num_train_steps, batch_size)):
                model.zero_grad()
                image = train_data['train'][train_index:train_index+batch_size]['image']
                text = train_data['train'][train_index:train_index+batch_size]['text']
                try:
                    inputs = self.processor(image, text, padding = True, return_tensors="pt").to(device)
                except ValueError:
                    value += 1
                    continue
                try:
                    out = model(**inputs)
                except RuntimeError:
                    runtime += 1
                    continue
                truth = torch.nn.functional.one_hot(torch.tensor(train_data['train'][train_index:train_index+batch_size]['label']), num_classes=self.num_classes)
                loss = loss_fct(out.float().to(device), truth.view(batch_size, self.num_classes).float().to(device))
                training_loss.append(loss.item())
                maximums = torch.argmax(out, dim = 1)
                truth_max = torch.argmax(truth, dim = 1)
                accuracy.update(maximums.to(device), truth_max.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            acc = accuracy.compute()
            print("Accuracy:", acc)
            print('runtime errors: ', runtime)
            print('value errors: ', value)
            self.plot(np.array(training_loss), epoch)
            print('\n')
            print('epoch: ', counter)
            counter += 1
            acc = accuracy.compute()
            print("Accuracy:", acc)
            print('loss total: ', sum(training_loss))
            print('\n')
            training_loss_over_epochs.append(training_loss)
            #exponential.step()
            cosine.step()

            model.eval()

        with torch.no_grad():
            num_test_steps = math.floor(test_data['test'].num_rows/1) * 1
            total_acc = 0
            runtime = 0
            value = 0
            accuracy_two = Accuracy("binary").to(device)
            for y in tqdm(range(num_test_steps)):
                image = test_data['test'][y]['image']
                text = test_data['test'][y]['text']
                try:
                    inputs = self.processor(image, text, return_tensors="pt").to(device)
                except ValueError:
                    value += 1
                    continue
                try:
                    out = model(**inputs)
                except RuntimeError:
                    runtime += 1
                    continue
                truth = torch.nn.functional.one_hot(torch.tensor(test_data['test'][y]['label']), num_classes=self.num_classes)
                maximums = torch.tensor([torch.argmax(out).item()])
                truth_max = torch.tensor([torch.argmax(truth).item()])
                accuracy_two.update(maximums.to(device), truth_max.to(device))
                total_acc += (maximums == truth_max)
        accuracy = total_acc / y
        print('runtime errors: ', runtime)
        print('value errors: ', value)
        print("Basic accuracy on test set: ", accuracy_two.compute())
        return training_loss_over_epochs, accuracy



# Fill these in with the desired processor, base_model, and other arguments
processor = None
base_model = None
output_layer_size = None
model =  classificationModel(base_model, None, output_layer_size)

# For the custom trainer, fill these in
optimizer = None
lr_scheduler = None
loss_function = None
epochs = None
lr = None
num_classes = None
batch_size = None
train = CustomTrainer(epochs, lr, train_data, test_data, model, processor, num_classes, loss_function, optimizer, lr_scheduler)
train.train(batch_size)