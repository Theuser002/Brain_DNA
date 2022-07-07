import numpy as np
import torch
import torch.nn as nn
import  config

from torch.nn import CrossEntropyLoss
from sklearn.metrics import roc_auc_score
from torch.nn.functional import softmax

from torch.nn.functional import one_hot
import itertools
if __name__ == "__main__":
    print('TEST')
    # arr = np.random.randint(1, 3, size=(1, 3))
    # print(arr)
# t = torch.tensor([[[1, 2],[3, 4]],[[5, 6],[7, 8]]])
# rand = np.random.rand(10, 12)
# rand = torch.tensor(rand)
# print(rand.size())
# rand  = rand.detach().cpu().numpy()
# print(rand, rand.shape)

# loss = nn.CrossEntropyLoss()
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.randn(3, 5).softmax(dim=1)
# print(input, input.shape, type(input))
# print(target, target.shape, type(target))
# output = loss(input, target)
# print(output)
# output.backward()

# labels = np.array([9, 5])
# labels = torch.Tensor(labels).long()
# one_hot_label = one_hot(labels, num_classes = 12)
# print(one_hot_label)

# m = nn.Softmax(dim=1)
# input = torch.randn(2, 3)
# output = m(input)
# print(input, output)

# labels = np.array([9, 5])
# print(labels)
# labels = torch.Tensor(labels).long()
# one_hot_label = one_hot(labels, num_classes = 12)
# logits = np.random.rand(2, 12)
# logits = torch.tensor(logits)
# out_probs = softmax(logits, dim = 1)
# print(out_probs.shape, one_hot_label.shape)

# y_true = np.random.randint(0, 5, (1000))
# y_true = torch.Tensor(y_true).long()
# print(y_true.shape)
# one_hot_true = one_hot(y_true, num_classes = 5)
# y_score = np.random.rand(1000, 5)
# y_score = torch.tensor(y_score)
# y_score = softmax(y_score)
# print(one_hot_true.shape, y_score.shape)
# # print(one_hot_true, y_score)
# print(roc_auc_score(one_hot_true, y_score))

# arr = [[1, 2, 3], [4, [5, 6]], [7, 8, 9]]
# chain = itertools.chain(*arr)
# print(list(chain))

# cfg = config.config_dict
# print(config.root_dir)
