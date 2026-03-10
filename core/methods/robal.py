import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import AdversarialTraining


class MultiMarginLoss(nn.Module):
    """ multiple margin terms, usually applied along with cosine classifier """

    def __init__(self, samples_per_class, m=0, s=1, tau_b=0, tau_m=0):
        super().__init__()
        self.s = s
        self.use_margin = m > 0 or tau_m > 0
        m_list = torch.tensor(samples_per_class / samples_per_class.min()).cuda()
        m_list = tau_m * torch.log(m_list).float()
        self.m_list = torch.cuda.FloatTensor(m_list) / self.s + m
        # print(">> Margins: \n", self.m_list)

        if tau_b > 0:
            prior = torch.tensor(samples_per_class / samples_per_class.sum()).cuda()
            self.prior_bias = tau_b * torch.log(prior)
            # print(">> Prior bias: ", self.prior_bias)
        else:
            self.prior_bias = 0
        print(">> multiple margin terms with s={}, m={}, tau_b={}, tau_m={}".format(
            s, m, tau_b, tau_m ))

    def forward(self, input, target, reduction='mean', criterion=F.cross_entropy):
        if self.use_margin:
            index = torch.zeros_like(input, dtype=torch.uint8)
            index.scatter_(1, target.data.view(-1, 1), 1)
            index_float = index.type(torch.cuda.FloatTensor)
            batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
            batch_m = batch_m.view((-1, 1))
            x_m_s = input - batch_m
            input = torch.where(index, x_m_s, input)
        input *= self.s
        input += self.prior_bias
        loss = criterion(input, target, reduction=reduction)
        return loss


class RoBal(AdversarialTraining):

    def __init__(self, beta=1.0, beta_adv=1.0, m=0, s=1, tau_b=0, tau_m=0, **kwargs):
        super().__init__(**kwargs)
        self.beta_adv = beta_adv
        self.beta = beta
        self.multimarginloss = MultiMarginLoss(self.samples_per_class, m, s, tau_b, tau_m)
        self.criterion_kl = nn.KLDivLoss(reduction='sum')

    def loss(self, model, wa_model, input, target, input_adv, target_adv, reduction='mean'):
        logits_adv, logits = model(input_adv), model(input)
        loss_margin = self.multimarginloss(logits_adv, target_adv, reduction=reduction, criterion=self.criterion)
        nat_probs, adv_probs = F.softmax(logits, dim=1), F.softmax(logits_adv, dim=1)
        loss_kl = self.criterion_kl(torch.log(adv_probs + 1e-12), nat_probs) / adv_probs.size(0)
        loss = self.beta_adv * loss_margin + self.beta * loss_kl
        return loss
