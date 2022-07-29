import sys
sys.path.append('/media/data/hungnt/work/Datasets/BTMD/Code/src')

import os
import numpy as np
import torch
import config
import pandas as pd
import pickle
import utils

from Models.MLP.Model import DNAMLP
from torch.nn.functional import softmax
from tqdm import tqdm
from utils import make_ndarray_from_csv

def dump_probs ():
    '''
    For each outer fold, load the saved model states of every inner fold's model and predict on the respective inner fold's test set.
    Then, take the logits and put them through the softmax layer to get the classification probabilites output (probs) and the labels, with each inner fold corresponding to each outer fold, append them to the list of probs and list of labels of that outer fold, and then dump (save) them to the respective pickle files.
    '''
    cfg = config.config_dict
    PROBS_PICKLES_DIR = cfg['PROBS_PICKLES_DIR']
    for i in range(1, 6):
        all_probs = []
        all_labels = []
        probs_pickle_filename = f'{i}.0_probs.pickle'
        labels_pickle_filename = f'{i}.0_labels.pickle'
        probs_pickle_filepath = os.path.join(PROBS_PICKLES_DIR, probs_pickle_filename)
        labels_pickle_filepath = os.path.join(PROBS_PICKLES_DIR, labels_pickle_filename)
        for j in tqdm(range(1, 6)):
            # Take innerfold
            fold = f'{i}.{j}'
            # print(fold)
            
            test_features, test_labels = make_ndarray_from_csv(fold, mode = 'test')
            
            # [Not optimized] Getting probs and labels for each sample and append to respective global lists
            in_features = cfg['n_features']
            model = DNAMLP(in_features, cfg['n_classes'])
            BEST_STATE_PATH = os.path.join(cfg['MLP_BEST_STATES_DIR'], f'{fold}_best_state.pth')
            model.load_state_dict(torch.load(BEST_STATE_PATH))
            # print(len(test_labels))
            for feature, label in zip(test_features, test_labels):
                feature = torch.Tensor(feature).float()
                feature.to(cfg['device'])
                # the logits of the model must be passed through softmax to get the classification probs
                prob = softmax(model(feature), dim = 0)
                prob = prob.detach().cpu().numpy()
                all_probs.append(prob)
                all_labels.append(label)
        
        with open (probs_pickle_filepath, 'wb') as handle:
            pickle.dump(all_probs, handle, protocol = pickle.HIGHEST_PROTOCOL)
        with open (labels_pickle_filepath, 'wb') as handle:
            pickle.dump(all_labels, handle, protocol = pickle.HIGHEST_PROTOCOL)
    return

def load_probs (outer_fold):
    '''
    Load probs and labels from the pickle files corresponding to each outer fold
    '''
    print(f'Loading classification probabilities and labels from pickle file for fold {outer_fold}')
    cfg = config.config_dict
    PROBS_PICKLES_DIR = cfg['PROBS_PICKLES_DIR']
    probs_pickle_filename = f'{outer_fold}_probs.pickle'
    labels_pickle_filename = f'{outer_fold}_labels.pickle'
    probs_pickle_filepath = os.path.join(PROBS_PICKLES_DIR, probs_pickle_filename)
    labels_pickle_filepath = os.path.join(PROBS_PICKLES_DIR, labels_pickle_filename)
    
    with open(probs_pickle_filepath, 'rb') as handle:
        probs = pickle.load(handle)
    with open(labels_pickle_filepath, 'rb') as handle:
        labels = pickle.load(handle)
    return probs, labels



def cal_train_epoch(epoch, model, cal_train_loader, criterion, optimizer, device):
    correct = 0
    total = 0
    total_loss = 0
    
    for probs, labels in tqdm(cal_train_loader):
        # Move tensors to device
        probs, labels = probs.to(device), labels.to(device)
        
        # Zero out gradient
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(probs)
        # print(logits.detach().cpu().numpy().shape, labels.detach().cpu().numpy().shape)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate batch's loss
        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    # Epoch's average loss
    train_loss = total_loss / len(cal_train_loader)
    train_acc = (correct / total) * 100
    
    return train_loss, train_acc

def cal_val_epoch(epoch, model, cal_val_loader, criterion, optimizer, device):
    correct = 0
    total = 0
    total_loss = 0
    
    with torch.no_grad():
        for probs, labels in tqdm(cal_val_loader):
            # Move tensors to device
            probs, labels = probs.to(device), labels.to(device)
            
            # Forward pass
            logits = model(probs)
            loss = criterion(logits, labels)
            
            # Calculate batch's loss
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        val_loss = total_loss / len(cal_val_loader)
        val_acc = (correct / total) * 100
    return val_loss, val_acc

def cal_run (fold, cal_train_loader, cal_val_loader, model, criterion, optimizer, config):
    history = {'train_accs': [], 'train_losses': [], 'val_accs': [], 'val_losses': []}
    model.to(config['device'])
    n_epochs = config['cal_n_epochs']
    CAL_BEST_STATES_DIR = config['MLP_CAL_BEST_STATES_DIR']
    CAL_BEST_MODELS_DIR = config['MLP_CAL_BEST_MODELS_DIR']
    CAL_BEST_STATES_PATH = os.path.join(CAL_BEST_STATES_DIR, f'{fold}_best_cal_state.pth')
    CAL_BEST_MODELS_PATH = os.path.join(CAL_BEST_MODELS_DIR, f'{fold}_best_cal_model.pth')
    diff_threshold = config['cal_diff_threshold']
    max_patience = config['cal_max_patience']
    patience = 0
    
    for epoch in range(1, n_epochs + 1):
        print(f'Epoch {epoch}/{n_epochs} of the inner folds corresponding to  outer fold {fold}')
        train_loss, train_acc = cal_train_epoch(epoch, model, cal_train_loader, criterion, optimizer, config['device'])
        val_loss, val_acc = cal_val_epoch(epoch, model, cal_val_loader, criterion, optimizer, config['device'])
        
        history['train_accs'].append(train_acc)
        history['train_losses'].append(train_loss)
        history['val_accs'].append(val_acc)
        history['val_losses'].append(val_loss)

        print('train_loss: %.5f | train_acc: %.3f' %(train_loss, train_acc))
        print('val_loss: %.5f | val_acc: %.3f' %(val_loss, val_acc))
        
        if val_loss == min(history['val_losses']):
            print('Lowest validation loss => saving model weights...')
            torch.save(model.state_dict(), CAL_BEST_STATES_PATH)
        if len(history['val_losses']) > 1:
            if abs(history['val_losses'][-2] - val_loss) < diff_threshold or history['val_losses'][-2] < val_loss:
                patience = patience + 1
                print(f'Patience increased to {patience}')
                if patience == max_patience:
                    print('Early stopping.')
                    break
            else:
                patience = 0
        print('---------------------------------------------')
    return max(history['val_accs'])

def cal_run_no_save (fold, cal_train_loader, cal_val_loader, model, criterion, optimizer, config):
    history = {'train_accs': [], 'train_losses': [], 'val_accs': [], 'val_losses': []}
    model.to(config['device'])
    n_epochs = config['cal_n_epochs']
    CAL_BEST_STATES_DIR = config['MLP_CAL_BEST_STATES_DIR']
    CAL_BEST_MODELS_DIR = config['MLP_CAL_BEST_MODELS_DIR']
    CAL_BEST_STATES_PATH = os.path.join(CAL_BEST_STATES_DIR, f'{fold}_best_cal_state.pth')
    CAL_BEST_MODELS_PATH = os.path.join(CAL_BEST_MODELS_DIR, f'{fold}_best_cal_model.pth')
    diff_threshold = config['cal_diff_threshold']
    max_patience = config['cal_max_patience']
    patience = 0
    
    for epoch in range(1, n_epochs + 1):
        print(f'Epoch {epoch}/{n_epochs} of the inner folds corresponding to  outer fold {fold}')
        train_loss, train_acc = cal_train_epoch(epoch, model, cal_train_loader, criterion, optimizer, config['device'])
        val_loss, val_acc = cal_val_epoch(epoch, model, cal_val_loader, criterion, optimizer, config['device'])
        
        history['train_accs'].append(train_acc)
        history['train_losses'].append(train_loss)
        history['val_accs'].append(val_acc)
        history['val_losses'].append(val_loss)

        print('train_loss: %.5f | train_acc: %.3f' %(train_loss, train_acc))
        print('val_loss: %.5f | val_acc: %.3f' %(val_loss, val_acc))

        if len(history['val_losses']) > 1:
            if abs(history['val_losses'][-2] - val_loss) < diff_threshold or history['val_losses'][-2] < val_loss:
                patience = patience + 1
                print(f'Patience increased to {patience}')
                if patience == max_patience:
                    print('Early stopping.')
                    break
            else:
                patience = 0
        print('---------------------------------------------')
    return max(history['val_accs'])

if __name__ == "__main__":
    # dump_probs()
    probs, labels = load_probs('1.0')
    print(probs)
    print(len(labels))
