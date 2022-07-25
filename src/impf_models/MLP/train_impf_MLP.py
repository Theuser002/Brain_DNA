import sys
sys.path.append('../../../src/')

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import config
import itertools

from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from utils import brier_score_tensor
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score 
from torch.nn.functional import softmax, one_hot
from sklearn.metrics import confusion_matrix
    
def impf_train_epoch(epoch, model, train_loader, criterion, optimizer, device):
    correct = 0
    total = 0
    total_loss = 0
    total_bs = 0
    model.to(device)
    model.train()
    
    # For loop through all batches
    all_labels = []
    all_logits = []
    for features, labels in train_loader:
        # Move tensors to device
        features = features.to(device)
        labels = labels.to(device)
        
        # Zero out gradient
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(features)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Evaluation
        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        # batch BS
        batch_bs = brier_score_tensor(logits, labels)
        total_bs += batch_bs
        
        # save logits and labels to calculate AUC
        for logit, label in zip(logits, labels):
            all_labels.append(label.item())
            all_logits.append(np.array(logit.detach().cpu().numpy()))
        
    # epoch's avrage LL
    train_loss = total_loss / len(train_loader)
    # epoch's average acc & ME
    train_acc = (correct / total) * 100.
    train_me = 100 - train_acc
    # epoch's average BS
    train_bs = total_bs/len(train_loader)
    # epoch's average AUC
    # all_labels_one_hot = one_hot(torch.Tensor(np.array(all_labels)).long())
    all_probs = softmax(torch.Tensor(np.array(all_logits)), dim = 1)
    # train_auc = roc_auc_score(all_labels_one_hot, all_probs)
    
    all_preds = [np.argmax(prob) for prob in all_probs]
    train_auc = roc_auc_score(all_labels, all_preds)
    train_f1 = f1_score(all_labels, all_preds)
    train_precision = precision_score(all_labels, all_preds)
    train_recall = recall_score(all_labels, all_preds)
    train_cfs = confusion_matrix(all_labels, all_preds)
    
    return train_loss, train_acc, train_me, train_bs, train_auc, train_f1, train_precision, train_recall, train_cfs

def impf_val_epoch(epoch, model, val_loader, criterion, device):
    correct = 0
    total = 0
    total_loss = 0
    total_bs = 0
    model.to(device)
    # For loop through all batches
    with torch.no_grad():
        # For loop through all batches
        all_labels = []
        all_logits = []
        for features, labels in val_loader:
            # Move tensors to device
            features, labels = features.to(device), labels.to(device)
            # Forward pass
            logits = model(features)
            
            # Evaluation and batch loss
            loss = criterion(logits, labels)
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total  += labels.size(0)
            
            # batch BS
            batch_bs = brier_score_tensor(logits, labels)
            total_bs += batch_bs
            
            # save logits and labels to calculate AUC
            for logit, label in zip(logits,labels):
                all_labels.append(label.item())
                all_logits.append(np.array(logit.detach().cpu().numpy()))
        
        # epoch's average LL
        val_loss = total_loss / len(val_loader)
        # epoch's average acc & ME
        val_acc = (correct / total) * 100
        val_me = (100 - val_acc)
        # epoch's average BS
        val_bs = total_bs/len(val_loader)
        # epoch's AUC
        # all_labels_one_hot = one_hot(torch.Tensor(np.array(all_labels)).long())
        all_probs = softmax(torch.Tensor(np.array(all_logits)), dim = 1)
        # val_auc = roc_auc_score(all_labels_one_hot, all_probs)
        
        all_preds = [np.argmax(prob) for prob in all_probs]
        val_auc = roc_auc_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds)
        val_precision = precision_score(all_labels, all_preds)
        val_recall = recall_score(all_labels, all_preds)
        val_cfs = confusion_matrix(all_labels, all_preds)
         
    return val_loss, val_acc, val_me, val_bs, val_auc, val_f1, val_precision, val_recall, val_cfs

def impf_run(class_name, alg, fold, train_loader, val_loader, test_loader, model, criterion, optimizer, config, save):
    history = {'train_accs': [], 'train_losses': [], 'train_precisions': [], 'train_recalls': [], 'train_aucs': [], 'train_f1s':[], 'val_accs': [], 'val_losses': [], 'val_precisions': [], 'val_recalls': [], 'val_aucs': [], 'val_f1s':[], 'test_accs': [], 'test_losses': [], 'test_precisions': [], 'test_recalls': [], 'test_aucs': [], 'test_f1s': []}
    model.to(config['device'])
    n_epochs = config['mlp_n_epochs']
    BEST_STATES_DIR= config['MLP_BEST_STATES_DIR']
    BEST_STATE_PATH = os.path.join(BEST_STATES_DIR, alg, class_name, f'{fold}.pth')
    diff_threshold = config['mlp_diff_threshold']
    max_patience = config['mlp_max_patience']
    patience = 0
    selected_metrics = ['train_acc', 'train_precision', 'train_recall', 'train_auc', 'train_f1', 'train_cfs', 'val_acc', 'val_precision', 'val_recall', 'val_auc', 'val_f1', 'val_cfs', 'test_acc', 'test_precision', 'test_recall', 'test_auc', 'test_f1', 'test_cfs']
    
    for epoch in tqdm(range(1, n_epochs + 1), desc='Epochs: ', position=2):
        # tqdm.write(f'Epoch {epoch}/{n_epochs} of fold {fold} of group {class_name}')
        
        # GET EPOCH'S RESULTS FROM TRAINING AND VALIDATING AND TESTING MODEL
        train_loss, train_acc, train_me, train_bs, train_auc, train_f1, train_precision, train_recall, train_cfs = impf_train_epoch(epoch, model, train_loader, criterion, optimizer, config['device'])
        val_loss, val_acc, val_me, val_bs, val_auc, val_f1, val_precision, val_recall, val_cfs = impf_val_epoch(epoch, model, val_loader, criterion, config['device'])
        test_loss, test_acc, test_me, test_bs, test_auc, test_f1, test_precision, test_recall, test_cfs = impf_val_epoch(epoch, model, test_loader, criterion, config['device'])
        
        # Record history for evaluation
        # history['train_accs'].append(train_acc)
        # history['train_losses'].append(train_loss)
        # history['train_precisions'].append(train_precision)
        # history['train_recalls'].append(train_precision) 
        # history['train_aucs'].append(train_auc)
        # history['train_f1s'].append(train_f1)
        # history['val_accs'].append(val_acc)
        history['val_losses'].append(val_loss)
        # history['val_precisions'].append(val_precision)
        # history['val_recalls'].append(val_precision)
        # history['val_aucs'].append(val_auc)
        # history['val_f1s'].append(val_f1)
        # history['test_accs'].append(test_acc)
        # history['test_losses'].append(test_loss)
        # history['test_precisions'].append(test_precision)
        # history['test_recalls'].append(test_precision)
        # history['test_aucs'].append(test_auc)
        # history['test_f1s'].append(test_f1)
        
        # PRINT EPOCH'S RESULTS OUT TO CONSOLE
        tqdm.write(f'[{class_name.upper()}] - {fold} - {epoch}/{n_epochs}')
        tqdm.write('train_loss: %.5f | train_acc: %.3f | train_precision: %.3f | train_recall: %.3f | train_f1: %.3f | train_auc: %.3f' % (train_loss, train_acc, train_precision, train_recall, train_f1,  train_auc))
        tqdm.write('val_loss: %.5f | val_acc: %.3f | val_precision: %.3f | val_recall: %.3f | val_f1: %.3f | val_auc: %.3f' % (val_loss, val_acc, val_precision, val_recall, val_f1, val_auc))
        tqdm.write('test_loss: %.5f | test_acc: %.3f | test_precision: %.3f | test_recall: %.3f | test_f1: %3f | test_auc: %.3f' % (test_loss, test_acc, test_precision, test_recall, test_f1, test_auc))
            
        if val_loss == min(history['val_losses']):
            # GET BEST EPOCH'S RESULTS AND WRITE IT TO AN EVALUATION FILE
            best_epoch_results = {'train_loss': train_loss, 'train_acc': train_acc, 'train_me': train_me, 'train_bs': train_bs, 'train_precision': train_precision, 'train_recall': train_recall, 'train_auc': train_auc, 'train_f1': train_f1, 'train_cfs': train_cfs, 'val_loss': val_loss, 'val_acc': val_acc, 'val_me': val_me, 'val_bs': val_bs, 'val_precision': val_precision,  'val_recall': val_recall, 'val_auc': val_auc, 'val_f1': val_f1, 'val_cfs': val_cfs, 'test_loss': test_loss, 'test_acc': test_acc, 'test_me': test_me, 'test_bs': test_bs, 'test_precision': test_precision, 'test_recall': test_recall, 'test_auc': test_auc, 'test_f1': test_f1, 'test_cfs': test_cfs}
            if save.lower() == 'save':
                tqdm.write('Lowest validation loss => saving model weights...')
                torch.save(model.state_dict(), BEST_STATE_PATH)
        if len(history['val_losses']) > 1:
            if abs(history['val_losses'][-2] - val_loss) < diff_threshold or history['val_losses'][-2] < val_loss:
                patience = patience + 1
                tqdm.write(f'Patience increased to {patience}')
                if patience == max_patience:
                    tqdm.write('Early stopping.')
                    break
            else:
                patience = 0
        tqdm.write('---------------------------------------------')

    eval_file = open(config[f'MLP_{alg}_EVALUATION_RESULTS'], 'a+')
    eval_file.write(f'\n-------------------------------\n[{class_name.upper()} - {alg} - {fold}]:\n')
    for key, value in best_epoch_results.items():
        if key in selected_metrics:
            if 'cfs' in key:
                eval_file.write(f'{key}:\n{value}\n')
            else:
                eval_file.write(f'{key}: {value} | ')
    eval_file.close()
    return best_epoch_results

if __name__ == "__main__":
    print('Train MLP.py running...')
