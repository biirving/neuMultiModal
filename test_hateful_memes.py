# generalizeable testing script
# using the HuggingFace Library

from datasets import load_dataset
from transformers import AutoModel
import torchvision.transforms as transforms
from torchvision.io import read_image
from transformers import BeitFeatureExtractor, BeitForImageClassification


import os
os.environ["HF_ENDPOINT"] = "https://huggingface.co"


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


# transform the image for input
my_transforms = transforms.Compose([
transforms.ToPILImage(),                                                                                                                                                                                                   
transforms.Resize((224, 224)),                                                                                                  
transforms.ToTensor()                                                                                                           
])


from transformers import AutoTokenizer, AutoModelForPreTraining
model = AutoModelForPreTraining.from_pretrained("uclanlp/visualbert-vqa-coco-pre")



image_0 = read_image(test_data['test'][0]['image'])
print(my_transforms(image_0))




