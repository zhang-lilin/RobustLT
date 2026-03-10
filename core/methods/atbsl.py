from .utils import AdversarialTraining, balanced_softmax_loss
from torch.nn import functional as F

def BSL(logits, targets, sample_per_class, tau_b=1., reduction='mean'):
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log() * tau_b
    loss = F.cross_entropy(logits, targets, reduction=reduction)
    return loss


class ATBSL(AdversarialTraining):

    def __init__(self, tau_b=1., **kwargs):
        super().__init__(**kwargs)
        self.tau_b = tau_b

    def loss(self, model, wa_model, input, target, input_adv, target_adv, reduction='mean'):
        logits_adv, logits = model(input_adv), model(input)
        loss = BSL(logits=logits_adv, targets=target_adv, sample_per_class=self.samples_per_class,
                                     tau_b=self.tau_b, reduction=reduction)
        return loss


