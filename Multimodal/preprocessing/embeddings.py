import torch

class Embeddings():

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_embeddings_txt(self, text, tokenizer, model):

        text_feats = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs_txt = model(**text_feats)
        last_hidden_states_txt = outputs_txt.last_hidden_state

        last_hidden_states_txt = torch.mean(last_hidden_states_txt,1)

        return last_hidden_states_txt.detach().cpu()

    def get_embeddings_img(self, image, img_processor, model):
        img_processed = img_processor(image, return_tensors='pt').to(self.device)    
        outputs_img = model(**img_processed)

        last_hidden_states_img = outputs_img.last_hidden_state
        last_hidden_states_img = torch.mean(last_hidden_states_img,1)

        return last_hidden_states_img.detach().cpu()

    def extract_fused_embeddings(self, img_embeddings, txt_embeddings):
        fuse = torch.cat((img_embeddings,txt_embeddings),1)
        return fuse.to(self.device)

