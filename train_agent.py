from __future__ import print_function

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn.functional as F

from torchvision import transforms as T
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from model import Model
from utils import history_transform, rgb2gray, action_to_id, get_dl, accuracy


def read_data(datasets_dir="./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)
    f.close()

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=5):

    X_train = rgb2gray(X_train)
    X_valid = rgb2gray(X_valid)

    X_train, y_train = history_transform(X_train, y_train, history_length)
    X_valid, y_valid = history_transform(X_valid, y_valid, history_length)

    y_train = np.stack([action_to_id(o) for o in y_train])
    y_valid = np.stack([action_to_id(o) for o in y_valid])

    return X_train, y_train, X_valid, y_valid


def train_model(X_train, y_train, X_valid, y_valid, max_epoch, batch_size, lr, model_dir="./models"):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _m = torch.tensor([141.98462, 141.98868, 141.99257, 141.9944 , 141.9963 ])
    _s = torch.tensor([61.780296, 61.775593, 61.771965, 61.769146, 61.76684 ])

    x_tfm = T.Compose([
                    T.Lambda(lambda x: torch.from_numpy(x).to(device)),
                    T.Normalize(mean=_m, std=_s),
                    ])
    y_tfm = T.Compose([
                    T.Lambda(lambda x: torch.from_numpy(x).to(device)),
                    ])
    transform = {'x': x_tfm, 'y':y_tfm}

    train_ds = Dataset(X_train, y_train, transform)
    valid_ds = Dataset(X_valid, y_valid, transform)
    train_dl, valid_dl = get_dl(train_ds, valid_ds, batch_size)

    nh = 10
    model = Model([16, 32, 32, 16], nh).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    loss_func = DoubleCELoss()

    train_losses = []
    valid_losses = []
    train_metrics = []
    valid_metrics = []

    writer = SummaryWriter()
    xb, _ = next(iter(train_dl))
    writer.add_graph(model, xb)
    
    train_losses = []
    valid_losses = []
    train_metrics = []
    valid_metrics = []

    for pg in optimizer.param_groups: pg['lr'] = lr

    for t in range(max_epoch):
        #--------------TRAINING SET-------------#
        train_metric = 0.0
        train_loss = 0.0
        n_examples = 0
        train_loss_component = {"steer":0.0, "accelerate":0.0}

        for i, (xb, yb) in enumerate(train_dl):
            model.train()

            # Perform forward pass
            y_pred = model(xb)

            # Compute the loss and metrics
            loss = loss_func(y_pred, yb)
            metric = accuracy(y_pred, yb)

            # Perform backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss / metrics across batches
            bs = xb.size(0)
            n_examples += bs
            train_loss += loss.item() * bs
            train_metric += metric.item() * bs
            for key in train_loss_component: train_loss_component[key] += loss_func.loss_component[key].item() * bs

        train_metrics.append(train_metric / n_examples)
        train_losses.append(train_loss / n_examples)
        for key in train_loss_component: train_loss_component[key] /= n_examples

        #--------------VALIDATION SET-------------#
        valid_metric = 0.0
        valid_loss = 0.0
        n_examples = 0
        valid_loss_component = {"steer":0.0, "accelerate":0.0}

        with torch.no_grad():
            model.eval()
            for i, (xb, yb) in enumerate(valid_dl):
                # Perform forward pass
                y_pred = model(xb)
                loss = loss_func(y_pred, yb)
                metric = accuracy(y_pred, yb)

                # Accumulate loss / metrics across batches
                bs = xb.size(0)
                n_examples += bs
                valid_loss += loss.item() * bs
                valid_metric += metric.item() * bs
                for key in valid_loss_component: valid_loss_component[key] += loss_func.loss_component[key].item() * bs

            valid_metrics.append(valid_metric / n_examples)
            valid_losses.append(valid_loss / n_examples)
            for key in valid_loss_component: valid_loss_component[key] /= n_examples

        writer.add_scalar('Train Loss', train_losses[-1], t)
        writer.add_scalar('Train Accuracy', train_metrics[-1], t)
        writer.add_scalar('Valid Loss', valid_losses[-1], t)
        writer.add_scalar('Valid Accuracy', valid_metrics[-1], t)

        print(f"[EPOCH]: {t}, [TRAIN LOSS]: {train_losses[-1]:.6f}, [TRAIN STEER LOSS]: {train_loss_component['steer']:.6f}, [TRAIN ACCELERATE LOSS]: {train_loss_component['accelerate']:.6f}, [TRAIN ACCURACY]: {train_metrics[-1]:.3f}")
        print(f"[EPOCH]: {t}, [VAL LOSS]: {valid_losses[-1]:.6f}, [VAL STEER LOSS]: {valid_loss_component['steer']:.6f}, [VAL ACCELERATE LOSS]: {valid_loss_component['accelerate']:.6f}, [VAL ACCURACY]: {valid_metrics[-1]:.3f}\n")
        
    # TODO: save your agent
    st = model.state_dict()
    filename = "test"
    torch.save(st, model_dir + "/" + filename)
    print(f"Model saved in file: {model_dir}/{filename}")

    writer.close()


class Dataset():
    def __init__(self, x, y, transform=None): 
        self.x, self.y = x, y
        self.transform = transform

    def __getitem__(self, i): 
        x_i, y_i = self.x[i], self.y[i]
        if self.transform:
            if 'x' in self.transform: x_i = self.transform['x'](x_i)
            if 'y' in self.transform: y_i = self.transform['y'](y_i)
        return x_i, y_i
    
    def __len__(self): return len(self.x)
    def __repr__(self): return f'{self.__class__.__name__}\nx: {self.x}\ny: {self.y}\n'


class DoubleCELoss():
    def __init__(self, λ = 1.00):
        self.λ = λ
        self.steer_loss_func = nn.CrossEntropyLoss()
        self.accelerate_loss_func = nn.CrossEntropyLoss()
        self.loss_component = {}

    def __call__(self, input, target):
        self.loss_component["steer"] = self.steer_loss_func(input[0], target[:, 0])
        self.loss_component["accelerate"] = self.accelerate_loss_func(input[1], target[:, 1])
        return self.λ * self.loss_component["steer"] + self.loss_component["accelerate"]


if __name__ == "__main__":

    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid)

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, y_valid, max_epoch=10, batch_size=16, lr=1e-3)

