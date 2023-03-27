from supervised.metrics import *
from tqdm import tqdm
import torch

def test(model, test_dataset, configuration, device):

     # Image Models
    img_processor = configuration['Models']['image_processor']
    model_img = configuration['Models']['image_model'].to(device)

    # Text Models
    tokenizer_txt = configuration['Models']['text_tokenizer']
    model_txt = configuration['Models']['text_model'].to(device)

    test_metrics = {"epoch":[],"num_steps":[],"test_acc":[]}
    for epoch in range(1, configuration['Hyperparameters']['epochs']+1):
        num_steps = 0
        val_num_steps = 0
        running_loss = 0.0
        val_running_loss = 0.0
        true = []
        pred = []
        val_true = []
        val_pred = []
        with torch.no_grad():
            testit = iter(test_dataset)
            for j in tqdm(range(len(test_dataset)), desc=f"[Epoch {epoch+1}]",ascii=' >='):
                test_data = next(testit)

                test_img_batch = test_dataset['img']
                test_text_batch = test_dataset['text']

                test_labels = torch.from_numpy(test_data['output']).to(torch.float32).to(device)

                test_last_hidden_states_img = get_embeddings.get_embeddings_img(test_img_batch, img_processor, model_img)
                test_last_hidden_states_txt = get_embeddings.get_embeddings_txt(test_text_batch, tokenizer_txt, model_txt)
                test_fused_embeddings = get_embeddings.extract_fused_embeddings(test_last_hidden_states_img, test_last_hidden_states_txt)

                test_outputs = model(test_fused_embeddings.to(torch.float32))

                # test_loss = criterion(test_outputs, test_labels)
                test_true.append(test_labels.cpu().detach().numpy()[0][0])
                # test_running_loss += test_loss.item()

                if test_outputs.cpu().detach().numpy()[0][0] >= 0.5:
                    test_pred.append(1)
                else:
                    test_pred.append(0)
                
                test_num_steps +=1


        test_acc = accuracy(test_true, test_pred)
        print(f'Num_steps : {test_num_steps}, test_acc : {test_acc}')

        test_metrics["epoch"].append(epoch)
        test_metrics["num_steps"].append(test_num_steps)
        test_metrics["test_acc"].append(test_acc)

    return test_metrics
        
