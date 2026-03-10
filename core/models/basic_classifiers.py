import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class FC_Classifier(nn.Module):
    """ plain FC classifier """

    def __init__(self, num_classes=10, in_dim=640):
        super(FC_Classifier, self).__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x, **kwargs):
        x = self.fc(x)
        return x


class Cos_Classifier(nn.Module):
    """ plain cosine classifier """

    def __init__(self,  num_classes=10, in_dim=640, scale=16, bias=False):
        super(Cos_Classifier, self).__init__()
        self.scale = scale
        self.weight = Parameter(torch.Tensor(num_classes, in_dim).cuda())
        self.bias = Parameter(torch.Tensor(num_classes).cuda(), requires_grad=bias)
        self.init_weights()

    def init_weights(self):
        self.bias.data.fill_(0.)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, **kwargs):
        ex = x / torch.norm(x.clone(), 2, 1, keepdim=True)
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        out = torch.mm(ex, self.scale * ew.t()) + self.bias
        return out


class NCE_Classifier(nn.Module):
    """ Norm classifier """

    def __init__(self,  num_classes=10, in_dim=640):
        super(NCE_Classifier, self).__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x, **kwargs):
        for _, module in self.fc.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.data = F.normalize(module.weight, p=2, dim=1)
        x = self.fc(x)
        return x


class CosPlus_Classifier(nn.Module):
    """ class of basic cosine classifier with more features for flexible adjustments """

    def __init__(self, num_classes=10, in_dim=640,
                 scale=16, bias=False, gamma=0.03125, eta=1,
                 moving_avg=True, mu=0.9, **kwargs):
        super(CosPlus_Classifier, self).__init__()
        self.num_classes = num_classes
        self.moving_avg = moving_avg
        self.in_dim = in_dim
        self.scale = scale
        self.gamma = gamma
        self.eta = eta
        self.mu = mu

        self.weight = Parameter(torch.Tensor(num_classes, in_dim).cuda())
        self.bias = Parameter(torch.Tensor(num_classes).cuda(), requires_grad=bias)
        if self.moving_avg:
            self.moving_ed = Parameter(torch.Tensor(1, in_dim).cuda(), requires_grad=False)

        self.init_weights()
        print(">> CosPlus Classifier built!")

    def init_weights(self):
        self.bias.data.fill_(0.)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, **kwargs):

        if self.eta != 1:
            ex = x / torch.norm(x.clone(), 2, 1, keepdim=True) ** self.eta
            ew = self.weight / (torch.norm(self.weight, 2, 1, keepdim=True)** self.eta + self.gamma)
        else:
            ex = x / torch.norm(x.clone(), 2, 1, keepdim=True)
            ew = self.weight / (torch.norm(self.weight, 2, 1, keepdim=True) + self.gamma)
        x = self.scale * (torch.mm(ex, ew.t())) + self.bias

        if self.training and self.moving_avg:
            # record moving average
            self.moving_ed.data = self.moving_ed.data * self.mu + torch.mean(ex, dim=0)

        return x


class Cos_Center_Classifier(nn.Module):
    """ cosine classifier based on the feature center of the training set """

    def __init__(self,  num_classes=10, in_dim=640):
        super(Cos_Center_Classifier, self).__init__()

        self.num_classes = num_classes
        self.in_dim = in_dim
        self.process_train = True

    def process_train_data(self, train_data):
        print(">> estimating center from saved training data features")
        labels = np.asarray(train_data['labels'])
        features = train_data['features']

        self.feature_means = torch.zeros(self.num_classes, self.in_dim).cuda()
        for i in range(self.num_classes):
            feature_cla = features[labels == i]
            self.feature_means[i] = torch.tensor(np.mean(feature_cla[0], axis=0)).cuda()


    def forward(self, x, **kwargs):
        ex = x / torch.norm(x.clone(), 2, 1, keepdim=True)
        ew = self.feature_means / torch.norm(self.feature_means, 2, 1, keepdim=True)
        out = torch.mm(ex, ew.t())
        return out
