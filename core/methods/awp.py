import copy
from collections import OrderedDict
import torch
from .utils import AdversarialTraining

EPS = 1E-20
def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


class AdvWeightPerturb(AdversarialTraining):
    def __init__(self, awp_gamma, awp_warmup, **kwargs):
        super().__init__(**kwargs)
        self.awp_gamma = awp_gamma
        self.awp_warmup = awp_warmup

    def update(self, loss, model, wa_model, optimizer, input, target, input_adv, target_adv):
        if self.current_epoch > self.awp_warmup:
            origin_model = copy.deepcopy(model)
            origin_opt = copy.deepcopy(optimizer)
            loss = - loss.mean()
            optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            optimizer.step()
            diff = diff_in_weights(origin_model, model)
            model.load_state_dict(origin_model.state_dict())
            optimizer.load_state_dict(origin_opt.state_dict())
            add_into_weights(model, diff, coeff=1.0 * self.awp_gamma)
        loss = self.loss(model=model, wa_model=wa_model, input=input, target=target, input_adv=input_adv, target_adv=target_adv)
        optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
        optimizer.step()
        if self.current_epoch > self.awp_warmup:
            add_into_weights(model, diff, coeff=-1.0 * self.awp_gamma)
        batch_metrics = {'loss': loss.item()}
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




