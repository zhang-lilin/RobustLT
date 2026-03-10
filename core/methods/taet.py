import torch
from .utils import AdversarialTraining


class TAET(AdversarialTraining):

    def __init__(self, ce_epochs=40, alpha=0.1, beta=0.1, gamma=0.1, **kwargs):
        super().__init__(**kwargs)
        self.taet_alpha = alpha
        self.taet_beta = beta
        self.taet_gamma = gamma
        self.ce_epochs = ce_epochs

    def get_adversarial_examples(self, model, wa_model, input, target, epsilon=None, step_size=None, perturb_steps=None):
        if self.current_epoch < self.ce_epochs:
            return input, target
        else:
            return super().get_adversarial_examples(model, wa_model, input, target, epsilon, step_size, perturb_steps)

    def update(self, loss, model, wa_model, optimizer, input, target, input_adv, target_adv):
        loss = loss.mean()
        batch_metrics = {'loss': loss.item()}
        optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
        optimizer.step()
        with torch.no_grad():
            logits = model(input)
            if input_adv is not None:
                logits_adv = model(input_adv)
            else:
                logits_adv = model(input)
            self.logger.update(logits, target, logits_adv)
            if self.current_step % self.num_batches == 0:
                if self.current_epoch < self.ce_epochs:
                    result, _ = self.logger.result()
                    cw_result = result.mean()
                    batch_metrics[f'cw_nat_acc'] = cw_result.item()
                    overall_result, _ = self.logger.result_overall()
                    batch_metrics[f'nat_acc'] = overall_result.item()
                else:
                    result, result_adv = self.logger.result()
                    cw_result, cw_result_adv = result.mean(), result_adv.mean()
                    batch_metrics[f'cw_nat_acc'] = cw_result.item()
                    batch_metrics[f'cw_adv_acc'] = cw_result_adv.item()
                    overall_result, overall_result_adv = self.logger.result_overall()
                    batch_metrics[f'nat_acc'] = overall_result.item()
                    batch_metrics[f'adv_acc'] = overall_result_adv.item()
        return batch_metrics

    def loss(self, model, wa_model, input, target, input_adv, target_adv, reduction='mean'):
        if self.current_epoch < self.ce_epochs:
            loss = self.criterion(model(input), target, reduction=reduction)
        else:
            logits_adv, logits = model(input_adv), model(input)

            # Compute cross-entropy loss
            pixel_loss = self.criterion(logits_adv, target_adv, reduction='none')

            # Compute class-wise losses
            class_losses = torch.zeros(self.num_classes).to(target.device)
            for cls in range(self.num_classes):
                mask = (target_adv == cls).float()
                class_loss = (pixel_loss * mask).sum() / (mask.sum() + 1e-10)
                class_losses[cls] = class_loss

            # Compute average and normalized class losses
            avg_class_loss = class_losses.mean()
            normalized_class_losses = class_losses / (class_losses.sum() + 1e-10)

            # Loss components
            balanced_loss = self.taet_alpha * avg_class_loss
            hierarchical_loss = self.taet_beta * ((class_losses - avg_class_loss) ** 2).mean()
            rare_class_loss = self.taet_gamma * (normalized_class_losses ** 2).sum()

            # Total loss
            loss = balanced_loss + hierarchical_loss + rare_class_loss
        return loss
