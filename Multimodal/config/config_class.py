# Hyperparameters:
'''
Num epochs
Use GPU
'''
# Data
'''
Path to data
cache size
Batch Size
Shuffle
'''
# Models
'''
Image package - Hugging Face
Text package - Hugging Face
image_processor_pretrained
image_model
text_processor_pretrained
text_model
MLP - In channels
MLP - num classes
MLP - Hidden Sizes (list)
'''

# Optimizers
'''
Optimizers
Learning rate
Use LR scheduler (bool)
Learning rate schedulers
momentum
'''
# Loss
'''
loss function
'''
# Save Best
'''
Save Best Models (bool)
Save Best Weights (bool)
'''

import importlib
from models.basic_models import MLP


class Config():
    def __init__(self,
             # Hyperparameters
             epochs=100,
             use_gpu=False,
             # Dataset
             path_to_data='',
             cache_size=1000,
             batch_size=32,
             shuffle=True,
             # Models
             base_import_package='',
             image_processor_package = '',
             image_model_package='',
             text_tokenizer_package = '',
             text_model_package='',
             image_processor_pretrained='',
             image_model='',
             text_processor_pretrained='',
             text_model='',
             mlp_in_channels=1792,
             mlp_num_classes=1,
             mlp_hidden_sizes=[128, 64],
             mlp_dropout_prob = [0.5, 0.7],
             # Optimizers
             optimizer='',
             learning_rate=0.01,
             use_lr_scheduler=False,
             lr_scheduler='',
             momentum=0.9,
             # Loss
             loss_fn='',
             use_custom_loss=False,
             custom_loss_fn='',
             reg_lambda = 0.01,
             # Save Best
             save_best_models=False,
             save_best_weights=False):
        self.epochs = epochs
        self.use_gpu = use_gpu
        self.path_to_data = path_to_data
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.base_import_package = base_import_package
        self.image_processor_package = image_processor_package
        self.image_model_package = image_model_package
        self.text_tokenizer_package = text_tokenizer_package
        self.text_model_package = text_model_package
        self.image_processor_pretrained = image_processor_pretrained
        self.image_model = image_model
        self.text_processor_pretrained = text_processor_pretrained
        self.text_model = text_model
        self.mlp_in_channels = mlp_in_channels
        self.mlp_num_classes = mlp_num_classes
        self.mlp_hidden_sizes = mlp_hidden_sizes
        self.mlp_dropout_prob = mlp_dropout_prob
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.use_lr_scheduler = use_lr_scheduler
        self.lr_scheduler = lr_scheduler
        self.momentum = momentum
        self.loss_fn = loss_fn
        self.use_custom_loss = use_custom_loss
        self.custom_loss_fn = custom_loss_fn
        self.reg_lambda = reg_lambda
        self.save_best_models = save_best_models
        self.save_best_weights = save_best_weights
    
    def get_config(self):
        base_import_package = importlib.import_module(self.base_import_package)
        image_processor = getattr(base_import_package,self.image_processor_package)
        image_model = getattr(base_import_package,self.image_model_package)
        text_tokenizer = getattr(base_import_package,self.text_tokenizer_package)
        text_model = getattr(base_import_package,self.text_model_package)

        config = {
            'Hyperparameters': {
                'epochs': self.epochs,
                'use_gpu': self.use_gpu
            },
            'Dataset': {
                'path_to_data': self.path_to_data,
                'cache_size': self.cache_size,
                'batch_size': self.batch_size,
                'shuffle': self.shuffle
            },
            'Models': {
                'mlp_num_classes': self.mlp_num_classes,
                'image_processor': image_processor.from_pretrained(self.image_processor_pretrained),
                'image_model': image_model.from_pretrained(self.image_model),
                'text_tokenizer': text_tokenizer.from_pretrained(self.text_processor_pretrained),
                'text_model': text_model.from_pretrained(self.text_model),
                'mlp': MLP(in_channels=self.mlp_in_channels,num_classes=self.mlp_num_classes,hidden_sizes=self.mlp_hidden_sizes, dropout_probability= self.mlp_dropout_prob)
            },
            'Optimizers': {
                'optimizer': self.optimizer,
                'learning_rate': self.learning_rate,
                'use_lr_scheduler': self.use_lr_scheduler,
                'lr_scheduler': self.lr_scheduler,
                'momentum': self.momentum
            },
            'Loss': {
                'loss_fn': self.loss_fn,
                'use_custom_loss': self.use_custom_loss,
                'custom_loss_fn': self.custom_loss_fn,
                'reg_lambda' : self.reg_lambda
            },
            'Save Best': {
                'save_best_models': self.save_best_models,
                'save_best_weights': self.save_best_weights
            }
        }
        return config
