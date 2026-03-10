import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .utils import AdversarialTraining


def RBL(labels, logits, sample_pre_class, at_pre_class, reduction='mean'):
	beta = np.zeros(len(sample_pre_class)).astype(np.float32)
	E = np.zeros(len(sample_pre_class)).astype(np.float32)
	for i in range(len(sample_pre_class)):
		beta[i] = (sample_pre_class[i] - 1.) / sample_pre_class[i]
		E[i] = (1. - beta[i]**at_pre_class[i]) / (1. - beta[i])
	weights = 1. / (E + 1e-5)
	weights = weights / np.sum(weights) * len(sample_pre_class)
	loss = F.cross_entropy(logits, labels, weight=torch.from_numpy(weights.astype(np.float32)).cuda(), reduction=reduction)
	return loss

def BSL(logits, targets, sample_per_class, tau_b=1., reduction='mean'):
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log() * tau_b
    loss = F.cross_entropy(logits, targets, reduction=reduction)
    return loss


class REAT(AdversarialTraining):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.criterion_kl = nn.KLDivLoss(size_average='none')
		self.pre_at_sample = copy.deepcopy(self.samples_per_class)
		self.next_at_sample = [0 for i in range(len(self.samples_per_class))]
		self.save.extend(['pre_at_sample', 'next_at_sample'])

	def update(self, loss, model, wa_model, optimizer, input, target, input_adv, target_adv):
		loss = loss.mean()
		batch_metrics = {'loss': loss.item()}
		optimizer.zero_grad()
		loss.backward()
		if self.grad_clip is not None:
			torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
		optimizer.step()
		with torch.no_grad():
			logits_adv, logits = model(input_adv), model(input)
			self.logger.update(logits, target, logits_adv)
			if self.current_step % self.num_batches == 0:
				result, result_adv = self.logger.result()
				cw_result, cw_result_adv = result.mean(), result_adv.mean()
				batch_metrics[f'cw_nat_acc'] = cw_result.item()
				batch_metrics[f'cw_adv_acc'] = cw_result_adv.item()
				overall_result, overall_result_adv = self.logger.result_overall()
				batch_metrics[f'nat_acc'] = overall_result.item()
				batch_metrics[f'adv_acc'] = overall_result_adv.item()

			_, predicted = torch.max(logits.detach(), 1)
			_, predictedadv = torch.max(logits_adv.detach(), 1)
			for j in range(predictedadv.size(0)):
				self.next_at_sample[predictedadv[j].item()] += 1
			if self.current_step % self.num_batches == 0:
				self.pre_at_sample = copy.deepcopy(self.next_at_sample)
				self.next_at_sample = [0 for _ in range(len(self.samples_per_class))]
		return batch_metrics

	def loss(self, model, wa_model, input, target, input_adv, target_adv, reduction='mean'):
		fea_adv, logits_adv = model(input_adv, intermediate=True)
		spc = self.samples_per_class.type_as(input)
		weights = torch.sqrt(1. / (spc / spc.sum()))
		tail_class = [i for i in range(self.num_classes // 3 * 2 + 1, self.num_classes)]
		TAIL = None
		counter = 0.0
		for bi in range(target.size(0)):
			if target[bi].item() in tail_class:
				idt = torch.tensor(
					[-1. if target[bi].item() == target[bj].item() else 1. for bj in range(target.size(0))]).cuda()
				W = torch.tensor(
					[weights[target[bi].item()] + weights[target[bj].item()] for bj in range(target.size(0))]).cuda()
				l = self.criterion_kl(F.log_softmax(fea_adv, 1),
									  F.softmax(fea_adv[bi, :].clone().detach().view(1, -1).tile(target.size(0), ).view(
										  target.size(0), -1),1)) * idt * W
				TAIL = l if TAIL is None else TAIL + l
				counter += 1
		TAIL = TAIL.mean() / counter if counter > 0. else torch.tensor(0).to(target.device)
		bsl_loss = BSL(logits=logits_adv, targets=target_adv, sample_per_class=self.samples_per_class, reduction=reduction)
		loss = bsl_loss + TAIL
		return loss

	def adversarial_loss(self, model, wa_model, input, target, input_adv, target_adv, reduction='mean'):
		f_adv, logits_adv = model(input_adv, True)
		return RBL(target, logits_adv, self.samples_per_class, self.pre_at_sample, reduction=reduction)

