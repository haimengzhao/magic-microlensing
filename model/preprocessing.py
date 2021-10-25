import torch
from torch.utils.data import DataLoader
import h5py

import model.utils as utils

def parse_datasets(args, device):
    '''
    Prepare dataset.

    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/parse_datasets.py
    '''

    with h5py.File(args.dataset, mode='r') as dataset_file:
        X_random = dataset_file['X_random'][...]
        Y = dataset_file['Y'][...]
        X_even = dataset_file['X_even'][...]

    train_test_split = len(X_random) - 10
    input_dim = X_random.shape[-1] - 1
    output_dim = X_even.shape[-1] - 1
    n_labels = Y.shape[-1]

    train_random_dataloader = DataLoader(X_random[:train_test_split], batch_size=args.batch_size, shuffle=False)
    train_label_dataloader = DataLoader(Y[:train_test_split], batch_size=args.batch_size, shuffle=False)
    train_even_dataloader = DataLoader(X_even[:train_test_split], batch_size=args.batch_size, shuffle=False)
    test_random_dataloader = DataLoader(X_random[train_test_split:], batch_size=len(X_random[train_test_split:]), shuffle=False)
    test_label_dataloader = DataLoader(Y[train_test_split:], batch_size=len(X_random[train_test_split:]), shuffle=False)
    test_even_dataloader = DataLoader(X_even[train_test_split:], batch_size=len(X_random[train_test_split:]), shuffle=False)

    data_objects = {
                "train_random_dataloader": utils.inf_generator(train_random_dataloader), 
                "train_label_dataloader": utils.inf_generator(train_label_dataloader), 
                "train_even_dataloader": utils.inf_generator(train_even_dataloader), 
                "test_random_dataloader": utils.inf_generator(test_random_dataloader),
                "test_label_dataloader": utils.inf_generator(test_label_dataloader),
                "test_even_dataloader": utils.inf_generator(test_even_dataloader),
                "input_dim": input_dim,
                "output_dim": output_dim,
                "n_train_batches": len(train_random_dataloader),
                "n_test_batches": len(test_random_dataloader),
                "n_labels": n_labels}

    return data_objects


