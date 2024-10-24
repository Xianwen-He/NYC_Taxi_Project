# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 15:24:45 2024

@author: XianwenHe
"""

import torch
from torch import nn
from scripts.model_utils import NYCTaxiExampleDataset, MLP
from scripts.data_utils import raw_taxi_df, clean_taxi_df, split_taxi_data
import random
import numpy as np
import argparse
import os


def main(max_epoch: int=5, lr: float=1e-4, train_size: int=500000, batch_size: int=10):
    """Simple training loop"""
    # Set fixed random number seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
  
    # load and process data
    raw_df = raw_taxi_df(filename="./data/yellow_tripdata_2024-01.parquet")
    clean_df = clean_taxi_df(raw_df=raw_df)
    location_ids = ['PULocationID', 'DOLocationID']
    X_train, X_test, y_train, y_test = split_taxi_data(clean_df=clean_df, 
                                                   x_columns=location_ids, 
                                                   y_column="fare_amount", 
                                                   train_size=train_size)

    # wrap the data with Pytorch
    dataset = NYCTaxiExampleDataset(X_train=X_train, y_train=y_train)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
  
    # Initialize the MLP
    mlp = MLP(encoded_shape=dataset.X_enc_shape)
  
    # Define the loss function and optimizer
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
  
    # Run the training loop
    for epoch in range(0, max_epoch): # specify the maximum number of epochs
        print('Starting epoch {}'.format(epoch+1))
        current_loss = 0.0
    
        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
            # Get and prepare inputs
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Perform forward pass
            outputs = mlp(inputs)
            
            # Compute loss
            loss = loss_function(outputs, targets)
            
            # Perform backward pass
            loss.backward()
            
            # Perform optimization
            optimizer.step()
            
            # Print the loss of the current iteration
            current_loss += loss.item()
            if i % 10 == 0:
                print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss))
            current_loss = 0.0
    # Process is complete.
    print('Training process has finished.')
    return X_train, X_test, y_train, y_test, mlp



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--save_path', type=str, default='./models/nyc_taxi_mlp.pt', help='path to save the model')
    parser.add_argument('--max_epoch', type=int, default=5, help='maximum number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--train_size', type=int, default=500000, help="size of the training set")
    parser.add_argument('--batch_size', type=int, default=10, help="size of each batch")
    
    args = parser.parse_args()
    
    # data processing and model training
    X_train, X_test, y_train, y_test, mlp = main(args.max_epoch, args.lr, args.train_size, args.batch_size)
    
    # save the model if applicable
    if args.save_path:
        torch.save(mlp.state_dict(), args.save_path)  # save the state dictionary of the model
    
    print("Training and saving finished.")
    
    
    