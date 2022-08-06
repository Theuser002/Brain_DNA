import os
import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import config_91
import itertools
import pickle

from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from utils_91 import brier_score_tensor
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score 
from torch.nn.functional import softmax, one_hot
from sklearn.metrics import confusion_matrix


def train_epoch(epoch, model, train_loader, criterion, optimizer, device):
    correct = 0
    total = 0
    total_loss = 0
    total_bs = 0
    model.to(device)
    model.train()
    
    # For loop through all batches
    all_labels = []
    all_logits = []
    for features, labels in tqdm(train_loader, desc= 'Training process', position=3):
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

    all_probs = np.array(all_probs.detach().cpu().numpy())
    all_labels = np.array(all_labels)
    all_preds = [np.argmax(prob) for prob in all_probs]
    all_logits = np.array(all_logits)

    train_auc = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovr')
    train_f1 = f1_score(all_labels, all_preds, average='macro')
    train_precision = precision_score(all_labels, all_preds, average='macro')
    train_recall = recall_score(all_labels, all_preds, average='macro')
    train_cfs = confusion_matrix(all_labels, all_preds)
    
    return train_loss, train_acc, train_me, train_bs, train_auc, train_f1, train_precision, train_recall, train_cfs, all_probs, all_labels, all_logits


def val_epoch(epoch, model, val_loader, criterion, device):
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
        for features, labels in tqdm(val_loader, desc='Validating process: ', position=3):
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
        
        all_probs = np.array(all_probs.detach().cpu().numpy())
        all_labels = np.array(all_labels)
        
        all_preds = [np.argmax(prob) for prob in all_probs]
        all_logits = np.array(all_logits)
        
        val_auc = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovr')
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_precision = precision_score(all_labels, all_preds, average='macro')
        val_recall = recall_score(all_labels, all_preds, average='macro')
        val_cfs = confusion_matrix(all_labels, all_preds)
        
    return val_loss, val_acc, val_me, val_bs, val_auc, val_f1, val_precision, val_recall, val_cfs, all_probs, all_labels, all_logits
            
    
def run(fold, train_loader, val_loader, model, criterion, optimizer, config, save):
    history = {'train_accs': [], 'train_losses': [], 'train_precisions': [], 'train_recalls': [], 'train_aucs': [], 'train_f1s':[], 'val_accs': [], 'val_losses': [], 'val_precisions': [], 'val_recalls': [], 'val_aucs': [], 'val_f1s':[]}
    device = config['device']
    model.to(device)
    n_epochs = config['nn91_n_epochs']
    BEST_STATE_PATH = os.path.join(config['NN91_BEST_STATES_DIR'], f'{fold}.pth')
    diff_threshold = config['nn91_diff_threshold']
    max_patience = config['nn91_max_patience']
    patience = 0
    selected_metrics = ['train_acc', 'train_bs' 'train_precision', 'train_recall', 'train_auc', 'train_f1','train_cfs', 'val_acc', 'val_bs', 'val_precision', 'val_recall', 'val_auc', 'val_f1', 'val_cfs']
    
    for epoch in tqdm(range(1, n_epochs + 1), desc='Epoch: ', position=1):
        train_loss, train_acc, train_me, train_bs, train_auc, train_f1, train_precision, train_recall, train_cfs, train_probs, train_labels, train_logits = train_epoch(epoch, model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_me, val_bs, val_auc, val_f1, val_precision, val_recall, val_cfs, val_probs, val_labels, val_logits = val_epoch(epoch, model, val_loader, criterion, config['device'])

        history['train_accs'].append(train_acc)
        history['train_losses'].append(train_loss)
        # history['train_precisions'].append(train_precision)
        # history['train_recalls'].append(train_precision)
        # history['train_aucs'].append(train_auc)
        # history['train_f1s'].append(train_f1)
        history['val_accs'].append(val_acc)
        history['val_losses'].append(val_loss)
        # history['val_precisions'].append(val_precision)
        # history['val_recalls'].append(val_precision)
        # history['val_aucs'].append(val_auc)
        # history['val_f1s'].append(val_f1)
        
        tqdm.write(f'{fold} - {epoch}/{n_epochs}')
        tqdm.write('train_loss: %.5f | train_acc: %.3f | train_precision: %.3f | train_recall: %.3f | train_auc: %.3f | train_f1: %.3f' % (train_loss, train_acc, train_precision, train_recall, train_auc, train_f1))
        tqdm.write('val_loss: %.5f | val_acc: %.3f | val_precision: %.3f | val_recall: %.3f | val_auc: %.3f | val_f1: %.3f' % (val_loss, val_acc, val_precision, val_recall, val_auc, val_f1))
        
        if val_loss == min(history['val_losses']):
            best_epoch_results = {'train_loss': train_loss, 'train_acc': train_acc, 'train_me': train_me, 'train_bs': train_bs, 'train_precision': train_precision, 'train_recall': train_recall, 'train_auc': train_auc, 'train_f1': train_f1, 'train_cfs': train_cfs, 'val_loss': val_loss, 'val_acc': val_acc, 'val_me': val_me, 'val_bs': val_bs, 'val_precision': val_precision,  'val_recall': val_recall, 'val_auc': val_auc, 'val_f1': val_f1, 'val_cfs': val_cfs}
            if save.lower() == 'save':
                tqdm.write('Lowest validation loss => saving model weights...')
                torch.save(model.state_dict(), BEST_STATE_PATH)
                with open(os.path.join(config['TRAIN_PROBS_DIR'], f'{fold}.pkl'), 'wb') as handle:
                    pickle.dump(train_probs, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.close()
                with open(os.path.join(config['TRAIN_LABELS_DIR'], f'{fold}.pkl'), 'wb') as handle:
                    pickle.dump(train_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.close()
                with open(os.path.join(config['TRAIN_LOGITS_DIR'], f'{fold}.pkl'), 'wb') as handle:
                    pickle.dump(train_logits, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.close()
                with open(os.path.join(config['VAL_PROBS_DIR'], f'{fold}.pkl'), 'wb') as handle:
                    pickle.dump(val_probs, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.close()
                with open(os.path.join(config['VAL_LABELS_DIR'], f'{fold}.pkl'), 'wb') as handle:
                    pickle.dump(val_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.close()
                with open(os.path.join(config['VAL_LOGITS_DIR'], f'{fold}.pkl'), 'wb') as handle:
                    pickle.dump(val_logits, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.close()
                
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
        
    eval_file = open(config[f'NN91_EVALUATION_RESULTS'], 'a+')
    eval_file.write(f'\n-------------------------------\n[{fold}]:\n')
    for key, value in best_epoch_results.items():
        if key in selected_metrics:
            if 'cfs' in key:
                eval_file.write(f'{key}:\n{value}\n')
            else:
                eval_file.write(f'{key}: {value} | ')
    eval_file.close()
    return best_epoch_results, history
                
if __name__ == "__main__":
    print('train_NN91.py is running...')