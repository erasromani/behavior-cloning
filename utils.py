from __future__ import print_function

import torch
import numpy as np
import pickle
import copy

from torch.utils.data import DataLoader

NOTHING = 0
LEFT = 1
RIGHT = 2
ACCELERATE = 1
BRAKE = 2


def get_dl(train_ds, valid_ds, batch_size, **kwargs):
    return (DataLoader(train_ds, batch_size, shuffle=True, **kwargs),
            DataLoader(valid_ds, 2 * batch_size, **kwargs))


def exponential_moving_average(x, beta=0.9):
        average = 0
        ema_x = x.copy()
        for i, o in enumerate(ema_x):
            average = average * beta + (1 - beta) * o
            ema_x[i] = average / (1 - beta**(i+1))
        return ema_x
        

def lr_find(dl, model, loss_func, optimizer, max_iter=100, min_lr=1e-6, max_lr=10):
    print("... finding learning rate")
    n_iter = 0
    lrs = []
    losses = []
    init_st = copy.deepcopy(model.state_dict())
    while n_iter < max_iter:
        for i, (xb, yb) in enumerate(dl):
            if n_iter >= max_iter: break
            model.train()
            lr = min_lr * (max_lr / min_lr) ** (n_iter / max_iter)
            lrs.append(lr)
            for pg in optimizer.param_groups: pg['lr'] = lr
            y_pred = model(xb)
            loss = loss_func(y_pred, yb)
            losses.append(loss.item())            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_iter += 1 
            if n_iter % 10 == 0: print(f"...    {n_iter*100/max_iter:.0f}% complete")
    model.load_state_dict(init_st)
    return lrs, losses


def accuracy(y_pred, y): 
    return torch.cat([torch.argmax(y_pred[0], dim=1)==y[:, 0], torch.argmax(y_pred[1], dim=1)==y[:, 1]]).float().mean()


def history_transform(X, y, history_length=5):
    X_tfm = []
    for i in range(history_length, len(X)):
        X_tfm.append(X[i-history_length:i])
    X_tfm = np.stack(X_tfm)
    return X_tfm, y[history_length:]


def rgb2gray(rgb):
    """ 
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    return gray.astype('float32') 


def action_to_id(a, tol=1e-3):
    if abs(a[0] - 0) < tol:     steer = NOTHING
    elif abs(a[0] - -1) < tol:  steer = LEFT
    else:                       steer = RIGHT

    if abs(a[1] - 1) < tol:     accelerate = ACCELERATE
    elif abs(a[2] - 0.2) < tol: accelerate = BRAKE 
    else:                       accelerate = NOTHING
    return np.array([steer, accelerate], dtype=np.int64)