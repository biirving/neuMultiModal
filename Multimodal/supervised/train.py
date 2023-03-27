from supervised.metrics import *
from tqdm import tqdm
import torch
import torch.nn
import torch.nn.functional as F
import numpy as np

def train(train_dataset, val_dataset, configuration, device, get_embeddings, head, criterion, optimizer):

    ### Preprocessing Models

    # Image Models
    img_processor = configuration['Models']['image_processor']
    model_img = configuration['Models']['image_model'].to(device)

    # Text Models
    tokenizer_txt = configuration['Models']['text_tokenizer']
    model_txt = configuration['Models']['text_model'].to(device)

    train_metrics = {"epoch":[],"num_steps":[],"train_loss":[],"val_loss":[],"train_acc":[],"val_acc":[]}
    for epoch in range(1, configuration['Hyperparameters']['epochs']+1):
        num_steps = 0
        val_num_steps = 0
        running_loss = 0.0
        val_running_loss = 0.0
        true = []
        pred = []
        val_true = []
        val_pred = []

        
        trainit = iter(train_dataset)
        for i in tqdm(range(len(train_dataset)), desc=f"[Epoch {epoch}]",ascii=' >='):
            data = next(trainit)
            # shuffle the dataset
            # items = list(data.items())
            # random.shuffle(items)
            # data = dict(items)
            
            img_batch = data['img']
            text_batch = data['text']
            labels = torch.from_numpy(data['output']).to(device)
            labels_one_hot = F.one_hot(labels.to(torch.int64).squeeze(), configuration['Models']['mlp_num_classes'])

            last_hidden_states_img = get_embeddings.get_embeddings_img(img_batch, img_processor, model_img)
            last_hidden_states_txt = get_embeddings.get_embeddings_txt(text_batch, tokenizer_txt, model_txt)
            fused_embeddings = get_embeddings.extract_fused_embeddings(last_hidden_states_img, last_hidden_states_txt)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = head(fused_embeddings.to(torch.float32))
            # add L2 regularization to the loss function
            regularization_loss = 0
            for param in head.parameters():
                regularization_loss += torch.sum(torch.square(param))
            loss = criterion(outputs, labels_one_hot.to(torch.float32)) + configuration['Loss']['reg_lambda'] * regularization_loss
            # loss = criterion(outputs, labels_one_hot)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            # print(outputs.round().cpu().detach().numpy()[0][0])
            # _, predicted = torch.max(outputs, 1)
            true.extend(labels_one_hot.cpu().detach().numpy())
            
            # print("out preds: ", predicted.shape)
            pred.extend(outputs.cpu().detach().numpy())
            # if outputs.cpu().detach().numpy()[0] >= 0.5:
            #     pred.append(1)
            # else:
            #     pred.append(0)
            
            num_steps +=1

        # Validation
        with torch.no_grad():
            valit = iter(val_dataset)
            for j in tqdm(range(len(val_dataset)), desc=f"[Epoch {epoch}]",ascii=' >='):
                val_data = next(valit)
                # shuffle the dataset
                # val_items = list(val_data.items())
                # random.shuffle(val_items)
                # val_data = dict(val_items)

                val_img_batch = val_data['img']
                val_text_batch = val_data['text']

                val_labels = torch.from_numpy(val_data['output']).to(device)
                val_labels_one_hot = F.one_hot(val_labels.to(torch.int64).squeeze(), configuration['Models']['mlp_num_classes'])


                val_last_hidden_states_img = get_embeddings.get_embeddings_img(val_img_batch, img_processor, model_img)
                val_last_hidden_states_txt = get_embeddings.get_embeddings_txt(val_text_batch, tokenizer_txt, model_txt)
                val_fused_embeddings = get_embeddings.extract_fused_embeddings(val_last_hidden_states_img, val_last_hidden_states_txt)

                val_outputs = head(val_fused_embeddings.to(torch.float32))

                val_loss = criterion(val_outputs, val_labels_one_hot.to(torch.float32))
                val_true.extend(val_labels_one_hot.cpu().detach().numpy())
                val_running_loss += val_loss.item()

                # _, val_predicted = torch.max(val_outputs, 1)    
                val_pred.extend(val_outputs.cpu().detach().numpy())
                # if val_outputs.cpu().detach().numpy()[0] >= 0.5:
                #     val_pred.append(1)
                # else:
                #     val_pred.append(0)
                
                val_num_steps +=1

        train_acc = accuracy(np.argmax(true, axis=1), np.argmax(pred, axis=1))
        val_acc = accuracy(np.argmax(val_true, axis=1), np.argmax(val_pred, axis=1))
        print(f'Num_steps : {num_steps}, train_loss : {running_loss/num_steps:.3f}, val_loss : {val_running_loss/val_num_steps:.3f}, train_acc : {train_acc}, val_acc : {val_acc}')

        train_metrics["epoch"].append(epoch)
        train_metrics["num_steps"].append(num_steps)
        train_metrics["train_loss"].append(running_loss/num_steps)
        train_metrics["val_loss"].append(val_running_loss/val_num_steps)
        train_metrics["train_acc"].append(train_acc)
        train_metrics["val_acc"].append(val_acc)

    return head, train_metrics