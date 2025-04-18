import hydra
from omegaconf import DictConfig

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from models.model_generator import get_model
from src.dataset import DatasetTabular
from src.trainer import SaintSupLightningModule

import time





@hydra.main(config_path="configs", config_name="config")    
def main(args: DictConfig) -> None:
    """function to make automatic prediction from held-out dataset for example in kaggle test set"""
    
    # if args.print_config is True:
    print(args)
    
    if args.experiment.pretrained_checkpoint is None:
        raise ValueError('Pretrained checkpoint path missing in config')
    
    transformer, embedding = get_model(args.experiment.model, 
                                       **args.transformer, 
                                       **args.data.data_stats,)
    
    if args.data.data_paths.test_csv_path is None:
        raise ValueError('Test csv path is not provided in config')
    
    test_df = pd.read_csv(args.data.data_paths.test_csv_path)
    test_y = np.array([-1]*len(test_df))
    test_dataset = DatasetTabular(test_df.values, test_y)
    test_loader = DataLoader(test_dataset, batch_size=args.dataloader.test_bs, 
                            num_workers=args.dataloader.num_workers, 
                            pin_memory=args.dataloader.pin_memory, shuffle=False,)
    
    fc = nn.Linear(args.transformer.embed_dim, args.experiment.num_output)

    print("defined farzi linear layer")


    model_dict = dict(transformer=transformer, 
                      embedding=embedding, fc=fc)
    params = dict(**model_dict, **args.optimizer, **args.augmentation, 
                       **args.transformer, **args.data.data_stats, **args.experiment,)
    
    # model = SaintSupLightningModule(**params)
    # model.load_from_checkpoint(args.experiment.pretrained_checkpoint, **params)

    model = SaintSupLightningModule.load_from_checkpoint(args.experiment.pretrained_checkpoint, **params)
    print("loaded model from chekpoint without any hicckups")

    model.eval()
    model.freeze()

    cls_embeddings = []
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (tuple, list)):  
                batch = batch[0]  # Extract only the input tensor if batch contains (input, labels)
            
            print(type(batch), len(batch) if isinstance(batch, (tuple, list)) else batch.shape)
            batch = batch.to(model.device)  # Move to GPU if needed
            cls_tokens = model(batch)  # Directly get CLS embeddings
            cls_embeddings.append(cls_tokens.cpu().numpy())

    # Convert list of arrays to a single NumPy array
    cls_embeddings = np.concatenate(cls_embeddings, axis=0)
    print("hello from post model call")

    # Get the current UTC time as a struct_time object
    utc_time = time.gmtime()

    # Format the UTC time into a readable string (YYYY-MM-DD HH:MM:SS)
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", utc_time)

    print("Current UTC time:", formatted_time)

    np.save(f"cls_embeddings_{utc_time}.npy", cls_embeddings)
    print(f"CLS token embeddings saved as 'cls_embeddings_{utc_time}.npy' with shape {cls_embeddings.shape}")


    
    # if args.experiment.save_prediction:
    #     test_df = test_df.reset_index()
    #     test_df = test_df[[args.experiment.id_col]]
    #     test_df[args.experiment.target_col] = -1
        
    #     preds = []
        
    #     with torch.no_grad():
    #         for x, _ in test_loader:
    #             output = model(x)
    #             if args.experiment.num_output in [1, None]:
    #                 pred = torch.sigmoid(output)
    #                 pred = (pred > 0.5).long()
    #             else:
    #                 pred = nn.functional.softmax(output, dim=1)
    #                 pred = torch.argmax(pred, dim=1)
    #             preds.append(pred)
    #         preds = torch.cat(preds, dim=0).squeeze()
    #         preds = preds.numpy()
        
    #     assert len(preds) == len(test_df)
    #     test_df[args.experiment.target_col] = preds
        
    #     test_df.to_csv(args.experiment.pred_sav_path, index=False)
    
    # else:
    #     raise NotImplementedError('Could not make prediction. "save_prediction" set to false')
    
    # print(f'Prediction finished,  csv saved at {args.experiment.pred_sav_path}')
    

if __name__ == "__main__":    
    main()
