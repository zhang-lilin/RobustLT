from .basic_classifiers import *


CLASSIFIERS = ['FC', 'Cos', 'CosPlus']


class Networks(nn.Module):

    def __init__(self, args, info, device, logger=None):
        super(Networks, self).__init__()
        self.info, self.logger, self.device = info, logger, device
        self.logger and self.logger.log('Model')
        self.name = args.model

        self.backbone = self.build_backbone(name=self.name, normalize=args.normalize)
        self.classifier = self.build_classifier(name=args.classifier,
                                                classifier_opt=getattr(args, 'classifier_opt', dict()))

    def build_backbone(self, name, normalize=True):

        self.logger and self.logger.log('--- backbone {}'.format(name))

        if 'tiny' in self.info['data'] or 'imagenet' in self.info['data']:
            from .tiny_wideresnet import wideresnet
            from .tiny_deit import deit

            if 'wrn' in name:
                backbone = wideresnet(name, self.logger, num_classes=self.info['num_classes'], device=self.device,
                                      normalize=normalize, mean=self.info['mean'], std=self.info['std'])
            elif 'deit' in name:
                backbone = deit(name, self.logger, num_classes=self.info['num_classes'], device=self.device,
                                normalize=normalize, mean=self.info['mean'], std=self.info['std'])

            else:
                raise ValueError('Invalid model name {}!'.format(name))
        else:
            from .resnet import resnet
            from .wideresnet import wideresnet
            from .deit import deit

            if 'resnet' in name and 'preact' not in name:
                backbone = resnet(name, self.logger, num_classes=self.info['num_classes'], device=self.device,
                                  normalize=normalize, mean=self.info['mean'], std=self.info['std'])
            elif 'wrn' in name:
                backbone = wideresnet(name, self.logger, num_classes=self.info['num_classes'], device=self.device,
                                      normalize=normalize, mean=self.info['mean'], std=self.info['std'])
            elif 'deit' in name:
                backbone = deit(name, self.logger, num_classes=self.info['num_classes'], device=self.device,
                                      normalize=normalize, mean=self.info['mean'], std=self.info['std'])
            else:
                raise ValueError('Invalid model name {}!'.format(name))

        return backbone.to(self.device)

    def build_classifier(self, name, classifier_opt):

        if self.logger:
            self.logger.log('--- classifier {}'.format(name))
            if classifier_opt:
                for k, v in classifier_opt.items():
                    self.logger.log('{} : {}'.format(k, v))

        if 'FC' == name:
            if 'deit' in self.name:
                classifier = self.backbone.head
            else:
                classifier = FC_Classifier(num_classes=self.info['num_classes'], in_dim=self.backbone.feat_dim)
        elif 'Cos' == name:
            classifier = Cos_Classifier(num_classes=self.info['num_classes'], in_dim=self.backbone.feat_dim,
                                        **classifier_opt)
        elif 'CosPlus' in name:
            classifier = CosPlus_Classifier(self.info['num_classes'], in_dim=self.backbone.feat_dim, **classifier_opt)
        else:
            raise NameError

        return classifier.to(self.device)

    def forward(self, x, intermediate=False):
        h = self.backbone(x)
        out = self.classifier(h)
        if intermediate:
            return h, out
        return out