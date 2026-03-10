import types

from .base_method import *


def robustlt(base_method, robustlt_alpha=0., robustlt_beta=0., **kwargs):
    # initial the base methods to be enhanced
    BASE = BASE_METHOD_DICT[base_method](**kwargs)

    # hyper-parameters for robustlt
    BASE.robustlt_alpha = robustlt_alpha
    BASE.robustlt_beta = robustlt_beta

    # initial the maximum value of classwise puturbation budget
    n_max, N = BASE.samples_per_class.max(), BASE.samples_per_class.sum()
    tau = robustlt_alpha / ((BASE.samples_per_class / N) * (n_max / BASE.samples_per_class).log().sqrt()).sum()
    BASE.classwise_epsilon_max = (1 - robustlt_alpha) * BASE.epsilon + tau * (n_max / BASE.samples_per_class).log().sqrt() * BASE.epsilon

    # Rewrite the adversarial sample generation function
    get_adversarial_examples = BASE.get_adversarial_examples
    def new_get_adversarial_examples(self, target, epsilon=None, step_size=None, **kwargs):
        assert epsilon is None and step_size is None
        intensity = min((self.current_epoch - 1) / (self.robustlt_beta * self.num_epochs), 1) if self.robustlt_beta > 0 else 1.
        self.classwise_epsilon = self.classwise_epsilon_max * intensity
        epsilon = self.classwise_epsilon.to(target.device)[target].view(-1, 1, 1, 1)
        step_size = epsilon / self.epsilon * self.step_size
        x_adv, target = get_adversarial_examples(target=target, epsilon=epsilon, step_size=step_size, **kwargs)
        return x_adv, target
    BASE.get_adversarial_examples = types.MethodType(new_get_adversarial_examples, BASE)

    # record the current perturbation budget (optional)
    update = BASE.update
    def new_update(self, **kwargs):
        batch_metrics = update(**kwargs)
        for cls in range(self.num_classes):
            batch_metrics[f'eps_{cls}'] = self.classwise_epsilon[cls].item()
        return batch_metrics
    BASE.update = types.MethodType(new_update, BASE)

    return BASE
