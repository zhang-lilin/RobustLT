import core.methods.utils
from .logger import Logger
from .parser import *
from .utils import *
import copy
import pandas as pd
from tqdm import tqdm as tqdm
from core.methods import get_method
from core.models import Networks
import core.metrics

class Trainer(object):
    """
    Helper class for training a deep neural network.
    Arguments:
        info (dict): dataset information.
        args (dict): input arguments.
        logger: (Logger): logger object.
    """
    def __init__(self, args, data_info, dataloader, logger, verbose=True):
        super(Trainer, self).__init__()
        device = self.device = args.device
        self.logger, self.info, self.params = logger, data_info, args
        self.dataloader = dataloader

        seed(args.seed)
        self.model = Networks(args=self.params, info=data_info, device=device, logger=logger if verbose else None)
        self.init_optimizer(verbose)
        self.init_scheduler(verbose)
        self.init_training_method(verbose)
        self.init_attack(self.get_model(), 'linf-pgd')

    def init_attack(self, model, attack_type='linf-pgd'):
        """
        Initialize adversary (for evaluation).
        """
        if attack_type == 'linf-fgsm':
            self.eval_attack = core.metrics.FGSM(model, eps=8 / 255)
        elif attack_type == 'linf-pgd':
            self.eval_attack = core.metrics.PGD(model, eps=8 / 255, alpha=1 / 255, steps=20)
        elif attack_type == 'linf-cw':
            self.eval_attack = core.metrics.CW_PGD(model, eps=8 / 255, alpha=1 / 255, steps=20)
        elif attack_type == 'linf-aa':
            self.eval_attack = core.metrics.AutoAttack(model, eps=8 / 255, version='standard', n_classes=self.info['num_classes'])
        else:
            raise NotImplementedError

    def init_optimizer(self, verbose=True):
        """
        Initialize optimizer.
        """
        verbose and self.logger and self.logger.log('Optimizer')

        if self.params.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr,
                                             weight_decay=self.params.weight_decay,
                                             momentum=0.9, nesterov=self.params.nesterov)
            info = f"--- sgd: lr-{self.params.lr} momentum-0.9 nesterove-{self.params.nesterov} weight_decay-{self.params.weight_decay}"

        elif self.params.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.lr)
            info = f"--- adam: lr-{self.params.lr}"

        else:
            raise NotImplementedError

        verbose and self.logger and self.logger.log(info)

    def init_scheduler(self, verbose=True):
        """
        Initialize scheduler.
        """

        lr, num_epochs = self.params.lr, int(self.params.num_epochs)

        if self.params.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, **self.params.scheduler_opt)
            info = f'--- lr scheduler: step {self.params.scheduler_opt}'

        elif self.params.scheduler == 'cosine':
            self.scheduler = CosineLR(self.optimizer, max_lr=lr, epochs=num_epochs)
            info = f'--- lr scheduler: cosine max_lr-{lr} epochs-{num_epochs}'

        elif self.params.scheduler == 'cosinew':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=lr, pct_start=0.05,
                                                                 total_steps=num_epochs)
            info = f'--- lr scheduler: cosinew max_lr-{lr} pct_start-{0.05} epochs-{num_epochs}'

        elif self.params.scheduler == 'none':
            self.scheduler = None
            info = f'--- no lr scheduler'

        else:
            raise NotImplementedError(self.params.scheduler)

        verbose and self.logger and self.logger.log(info)

    def init_training_method(self, verbose=True):
        """
        Initialize training method.
        """
        verbose and self.logger and self.logger.log('Algorithm')
        args = self.params

        data_param = dict(
            samples_per_class=torch.tensor(self.dataloader.dataset.samples_per_class).to(self.device),
            num_batches=len(self.dataloader), num_epochs=args.num_epochs)

        adv_train_params = dict(
            step_size=args.attack_step,
            epsilon=args.attack_eps,
            perturb_steps=args.attack_iter,
            grad_clip=1.0 if 'deit' in args.model else None,
        )

        method, method_opt = args.method, getattr(args, 'method_opt', dict())

        if method == 'std':
            from core.methods.standard import STD
            self.training_method = STD(**data_param, **method_opt)

        else:
            self.training_method = get_method(method, self.logger, **data_param, **method_opt, **adv_train_params)

        self.training_method.to(self.device)
        self.wa_model = None

        # Weight Averaging
        if 'ema_tau' in method_opt and method_opt['ema_tau'] > 0:
            tau = method_opt['ema_tau']
            verbose and self.logger and self.logger.log(f'--- using WA: tau-{tau}.')
            self.wa_model = copy.deepcopy(self.model)


    def train(self, epoch, verbose=True):
        """
        Run one epoch of training.
        """
        self.model.train()
        self.training_method.current_epoch = epoch
        device = self.device
        metrics_list = []

        for update_iter, (inputs, targets) in enumerate(
                tqdm(self.dataloader, desc=f"Epoch {epoch}: ", disable=not verbose), start=1):
            self.training_method.current_step = update_iter
            targets = targets.to(device)
            try:
                inputs = inputs.to(device)
            except AttributeError:
                inputs = [x.to(device) for x in inputs]

            args = dict(model=self.model, wa_model=self.wa_model, input=inputs, target=targets, optimizer=self.optimizer,)
            batch_metrics = self.training_method(**args)
            metrics_list.append(batch_metrics)

        metrics = pd.DataFrame(metrics_list).mean().to_dict()

        if verbose:
            for key, val in metrics.items():
                if "loss" in key:
                    self.logger.add("training", key, val, epoch)
                elif "eps" in key:
                    self.logger.add("budget", key, val, epoch)

            self.logger.log_stats(epoch, ["training", "budget"])

        if self.scheduler:
            self.scheduler.step()

        return pd.DataFrame(metrics_list).mean().to_dict()

    def model_parameters(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                yield p

    def class_wise_eval(self, dataloader, model=None, adversarial=False, verbose=True):
        model = model or self.get_model()
        model.eval()
        num_classes = self.info['num_classes']
        acc = torch.zeros(num_classes, device=self.device)
        total = torch.zeros(num_classes, device=self.device)

        for x, y in tqdm(dataloader, desc='Eval : ', disable=not verbose):
            x, y = x.to(self.device), y.to(self.device)

            if adversarial:
                with ctx_noparamgrad_and_eval(model):
                    x = self.eval_attack(x, y)
            with torch.no_grad():
                out = model(x)

            preds = out.argmax(dim=1)
            for label in range(num_classes):
                mask = y == label
                total[label] += mask.sum()
                acc[label] += (preds[mask] == label).sum()

        model.train()
        per_class_acc = acc / total.clamp(min=1)  # advoid 0
        return per_class_acc.mean().item()

    def get_model(self):
        if self.wa_model is None:
            return self.model
        else:
            return self.wa_model

    def save_model(self, path, epoch):
        if self.scheduler is not None:
            scheduler_state_dict = self.scheduler.state_dict()
        else:
            scheduler_state_dict = None

        if self.wa_model is None:
            save_dict = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': scheduler_state_dict,
                'epoch': epoch,
            }
        else:
            save_dict = {
                'model_state_dict': self.wa_model.state_dict(),
                'unaveraged_model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': scheduler_state_dict,
                'epoch': epoch,
            }
        if hasattr(self.training_method, 'save'):
            for key in self.training_method.save:
                save_dict[f'method_opt_{key}'] = getattr(self.training_method, key)
        if hasattr(self.training_method, 'save_model'):
            for key in self.training_method.save_model:
                save_dict[f'method_opt_model_{key}'] = getattr(self.training_method, key).state_dict()

        torch.save(save_dict, path)

    def load_model(self, path, weights_only=False):
        """
        load model weights and optimizer.
        """
        checkpoint = torch.load(path, weights_only=False)
        if 'model_state_dict' not in checkpoint:
            raise RuntimeError('Model weights not found at {}.'.format(path))
        else:
            if self.wa_model is not None:
                self.wa_model.load_state_dict(checkpoint['model_state_dict'])
                self.model.load_state_dict(checkpoint['unaveraged_model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])

            if not weights_only:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.scheduler is not None:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

                if hasattr(self.training_method, 'save'):
                    for key in self.training_method.save:
                        setattr(self.training_method, key, checkpoint[f'method_opt_{key}'])
                if hasattr(self.training_method, 'save_model'):
                    for key in self.training_method.save_model:
                        m = getattr(self.training_method, key)
                        m.load_state_dict(checkpoint[f'method_opt_model_{key}'])
                        setattr(self.training_method, key, m)

        return checkpoint['epoch']