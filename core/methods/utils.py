import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class NormalTraining(nn.Module):
    """
    template for normal training algorithm
    """
    def __init__(self, samples_per_class, num_batches, num_epochs, label_smoothing=0., **kwargs):
        super(NormalTraining, self).__init__()
        self.criterion = smooth_cross_entropy(smoothing=label_smoothing)
        self.samples_per_class = samples_per_class
        self.num_classes = self.samples_per_class.size(0)

        self.logger = classwise_accuracy_logger(class_num=self.num_classes)
        self.num_batches, self.num_epochs = num_batches, num_epochs
        self.current_step, self.current_epoch = 0, 0
        self.save, self.save_model = ['current_step'], []

    def forward(self, model, wa_model, optimizer, input, target):
        if self.current_step % self.num_batches == 1:
            self.logger.reset()
        loss = self.loss(model=model, wa_model=wa_model, input=input, target=target)
        batch_metrics = self.update(loss=loss, model=model, wa_model=wa_model, optimizer=optimizer, input=input, target=target)
        return batch_metrics

    def update(self, loss, model, wa_model, optimizer, input, target):
        loss = loss.mean()
        batch_metrics = {'loss': loss.item()}
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            logits_adv, logits = None, model(input)
            self.logger.update(logits, target, logits_adv)
            if self.current_step % self.num_batches == 0:
                result, _ = self.logger.result()
                cw_result = result.mean()
                batch_metrics[f'cw_nat_acc'] = cw_result.item()
                overall_result, _ = self.logger.result_overall()
                batch_metrics[f'nat_acc'] = overall_result.item()
        return batch_metrics

    def loss(self, model, wa_model, input, target, reduction='mean'):
        logits = model(input)
        return self.criterion(logits, target, reduction=reduction)

class AdversarialTraining(nn.Module):
    """
    template for adversarial training algorithm
    """
    def __init__(self, samples_per_class, num_batches, num_epochs, grad_clip=None,
                 attack='linf-pgd', epsilon=0.031, step_size=0.003, perturb_steps=10, label_smoothing=0., **kwargs):
        super(AdversarialTraining, self).__init__()
        # adversarial parameters
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.attack = attack
        self.criterion = smooth_cross_entropy(smoothing=label_smoothing)
        self.grad_clip = grad_clip

        # dataset prior
        self.samples_per_class = samples_per_class
        self.num_classes = self.samples_per_class.size(0)

        # training log
        self.logger = classwise_accuracy_logger(class_num=self.num_classes)
        self.num_batches, self.num_epochs = num_batches, num_epochs
        self.current_step, self.current_epoch = 0, 0
        self.save, self.save_model = [], []

    # the sign of beginning of an epoch is 'current_step % self.num_batches == 1'
    # the sign of end of an epoch is 'current_step % self.num_batches == 0'
    # NOTE: if something is updated at the end 4of an epoch, add it to save list
    def forward(self, model, wa_model, optimizer, input, target):
        if self.current_step % self.num_batches == 1:
            self.logger.reset()
        args = dict(zip(["model", "wa_model", "input", "target"], [model, wa_model, input, target]))
        adversarial_exmples = dict(zip(["input_adv", "target_adv"], (self.get_adversarial_examples(**args))))
        loss = self.loss(**adversarial_exmples, **args)
        batch_metrics = self.update(optimizer=optimizer, loss=loss, **adversarial_exmples, **args)
        return batch_metrics

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
        return batch_metrics

    def get_adversarial_examples(self, model, wa_model, input, target, epsilon=None, step_size=None, perturb_steps=None):
        epsilon = self.epsilon if epsilon is None else epsilon
        step_size = self.step_size if step_size is None else step_size
        perturb_steps = self.perturb_steps if perturb_steps is None else perturb_steps

        model.train()
        # set BN-mode to eval
        track_bn_stats(model, False)
        x_adv = input.detach() + torch.FloatTensor(input.shape).uniform_(-self.epsilon, self.epsilon).to(
            target.device).detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        if self.attack == 'linf-pgd':
            for _ in range(perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss = self.adversarial_loss(
                        model=model, wa_model=wa_model, input=input, target=target, input_adv=x_adv, target_adv=target)
                grad = torch.autograd.grad(loss, [x_adv])[0]
                x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, input - epsilon), input + epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            raise ValueError(f'Attack={self.attack} not supported!')

        model.train()
        track_bn_stats(model, True)
        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

        return x_adv, target

    def loss(self, model, wa_model, input, target, input_adv, target_adv, reduction='mean'):
        return self.criterion(model(input_adv), target_adv, reduction=reduction)

    def adversarial_loss(self, model, wa_model, input, target, input_adv, target_adv, reduction='mean'):
        return F.cross_entropy(model(input_adv), target, reduction=reduction)


class smooth_cross_entropy(torch.nn.Module):

    def __init__(self, smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, targets, reduction='mean'):
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=targets.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        return loss


class classwise_accuracy_logger():
    def __init__(self, class_num=10) -> None:
        self.class_num = class_num
        self.reset()

    def update(self, output, y, output_adv):
        idx = torch.tensor(np.array(range(len(output)))).to(y.device)
        idx_l, idx_u = idx[y != -1], idx[y == -1]
        y = y[idx_l]
        if output is None and output_adv is None:
            return
        if output is not None:
            output = output[idx_l]
            pred = output.max(1)[1]
            correct = pred == y
        if output_adv is not None:
            output_adv = output_adv[idx_l]
            pred_adv = output_adv.max(1)[1]
            correct_adv = pred_adv == y
        for index, label in enumerate(y):
            self.cw_n[label] += 1
            if output is not None and correct[index]:
                self.cw_natural[label] += 1
            if output_adv is not None and correct_adv[index]:
                self.cw_robust[label] += 1

    def result(self):
        cw_natural, cw_robust = torch.zeros_like(self.cw_natural), torch.zeros_like(self.cw_robust)
        for i in range(self.class_num):
            if self.cw_n[i] > 0:
                cw_natural[i] = self.cw_natural[i] / self.cw_n[i]
                cw_robust[i] = self.cw_robust[i] / self.cw_n[i]
        return cw_natural, cw_robust

    def result_overall(self):
        return self.cw_natural.sum().div(self.cw_n.sum()), self.cw_robust.sum().div(self.cw_n.sum())

    def reset(self):
        self.cw_n = torch.zeros(self.class_num)
        self.cw_robust = torch.zeros(self.class_num)
        self.cw_natural = torch.zeros(self.class_num)


def track_bn_stats(model, track_stats=True):
    """
    If track_stats=False, do not update BN running mean and variance and vice versa.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = track_stats


def set_bn_momentum(model, momentum=1):
    """
    Set the value of momentum for all BN layers.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = momentum


def ema_update(wa_model, model, global_step, decay_rate=0.995, warmup_steps=0, dynamic_decay=True):
    """
    Exponential model weight averaging update.
    """
    factor = int(global_step >= warmup_steps)
    if dynamic_decay:
        delta = global_step - warmup_steps
        decay = min(decay_rate, (1. + delta) / (10. + delta)) if 10. + delta != 0 else decay_rate
    else:
        decay = decay_rate
    decay *= factor

    for p_swa, p_model in zip(wa_model.parameters(), model.parameters()):
        p_swa.data *= decay
        p_swa.data += p_model.data * (1 - decay)


@torch.no_grad()
def update_bn(avg_model, model):
    """
    Update batch normalization layers.
    """
    avg_model.eval()
    model.eval()
    for module1, module2 in zip(avg_model.modules(), model.modules()):
        if isinstance(module1, torch.nn.modules.batchnorm._BatchNorm):
            module1.running_mean = module2.running_mean
            module1.running_var = module2.running_var
            module1.num_batches_tracked = module2.num_batches_tracked


def sigmoid_rampup(global_step, start_iter, end_iter):
    if global_step < start_iter:
        return 0.
    elif start_iter >= end_iter:
        return 1.
    else:
        rampup_length = end_iter - start_iter
        cur_ramp = global_step - start_iter
        cur_ramp = np.clip(cur_ramp, 0, rampup_length)
        phase = 1.0 - cur_ramp / rampup_length
        return np.exp(-5.0 * phase * phase)


@torch.no_grad()
def update_bn(avg_model, model):
    """
    Update batch normalization layers.
    """
    avg_model.eval()
    model.eval()
    for module1, module2 in zip(avg_model.modules(), model.modules()):
        if isinstance(module1, torch.nn.modules.batchnorm._BatchNorm):
            module1.running_mean = module2.running_mean
            module1.running_var = module2.running_var
            module1.num_batches_tracked = module2.num_batches_tracked





