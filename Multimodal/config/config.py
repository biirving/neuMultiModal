from config_class import Config
import pandas as pd
import os

home_path = os.getcwd()
config_params_df = pd.read_csv(home_path + "/config/config_params.csv")

print(config_params_df)


for idx in range(config_params_df.shape[0]):
    print(f"*************CONFIGURATION: {idx+1}******************")

    configuration = Config(epochs=int(config_params_df.loc[idx,"epochs"]),
                use_gpu=False,
                path_to_data='hatefulmemes/',
                cache_size=1000,
                batch_size=int(config_params_df.loc[idx,"batch_size"]),
                shuffle=True,
                base_import_package=config_params_df.loc[idx,"base_import_package"],
                image_processor_package = config_params_df.loc[idx,"image_processor_package"],
                image_model_package=config_params_df.loc[idx,"image_model_package"],
                text_tokenizer_package = config_params_df.loc[idx,"text_tokenizer_package"],
                text_model_package=config_params_df.loc[idx,"text_model_package"],
                image_processor_pretrained=config_params_df.loc[idx,"image_processor_pretrained"],
                image_model=config_params_df.loc[idx,"image_model"],
                text_processor_pretrained=config_params_df.loc[idx,"text_processor_pretrained"],
                text_model=config_params_df.loc[idx,"text_model"],
                mlp_in_channels=int(config_params_df.loc[idx,"mlp_in_channels"]),
                mlp_num_classes=int(config_params_df.loc[idx,"mlp_num_classes"]),
                mlp_hidden_sizes=[eval(i) for i in config_params_df.loc[idx,"mlp_hidden_sizes"].split(", ")],
                mlp_dropout_prob=[eval(i) for i in config_params_df.loc[idx,"dropout_probability"].split(", ")],
                optimizer='',
                learning_rate=float(config_params_df.loc[idx,"learning_rate"]),
                use_lr_scheduler=False,
                lr_scheduler='',
                momentum=float(config_params_df.loc[idx,"momentum"]),
                loss_fn='',
                use_custom_loss=False,
                custom_loss_fn='',
                reg_lambda= float(config_params_df.loc[idx,"reg_lambda"]),
                save_best_models=False,
                save_best_weights=False).get_config()

    filename = home_path + "/examples/hatefulmemes/late_fusion.py"
    exec(compile(open(filename, "rb").read(), filename, 'exec'))

    
