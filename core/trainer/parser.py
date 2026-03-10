from core.data import DATASETS
from core.models import CLASSIFIERS
from core.trainer.utils import str2bool, str2float


def set_parser_train(parser):
    parser.add_argument('--desc', type=str, default='METHODINFO_MODELINFO_SEEDINFO',
                        help='Description of experiment. It will be used to name directories.')
    parser.add_argument('--log-dir', type=str, default='trained_models/DATAINFO')
    parser.add_argument('-seed', '--seed', type=int, default=102, help='Random seed.')

    # Dataloader
    parser.add_argument('--data-dir', type=str, default='../dataset_data')
    parser.add_argument('--data', type=str, default='cifar10lf', choices=DATASETS, help='Data to use.')
    parser.add_argument('--imbalance_type', type=str, default='exp', help='Imbalance type for manipulating dataset.')
    parser.add_argument('--imbalance_rate', type=int, default=50, help='Imbalance rate for manipulating dataset.')
    parser.add_argument('--augment', type=str, default='base', help='Augmentation for training set.')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--num-batches', default=None, help='The number of batches of an epoch for training.')
    parser.add_argument('--balance-train', action='store_true', default=False, help='Use class-balanced sampler for training.')

    # Model
    parser.add_argument('--model', default='wrn-28-10', help='Model architecture to be used.')
    parser.add_argument('--classifier', choices=CLASSIFIERS, default='FC', help='Classifier to be used.')
    parser.add_argument('--classifier_opt', type=dict, default={}, help='Classifier settings.')
    parser.add_argument('--normalize', type=str2bool, default=True, help='Normalize input.')
    parser.add_argument('--pre_resume_path', default='', type=str, help='A pre-trained model path for initialization.')

    # Train
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--optimizer', default='sgd', help='Type of optimizer.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for optimizer.')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Optimizer (SGD) weight decay.')
    parser.add_argument('--scheduler', default='none', help='Type of scheduler.')
    parser.add_argument('--nesterov', type=str2bool, default=True, help='Use Nesterov momentum.')
    parser.add_argument('--attack', type=str, default='linf-pgd', help='Type of attack.')
    parser.add_argument('--attack-eps', type=str2float, default=8 / 255, help='Epsilon for the attack.')
    parser.add_argument('--attack-step', type=str2float, default=2 / 255, help='Step size for PGD attack.')
    parser.add_argument('--attack-iter', type=int, default=10, help='Max. number of iterations (if any) for the attack.')
    parser.add_argument('--tau', type=float, default=0., help='Weight averaging decay.')

    # Algorithm
    parser.add_argument('--method', type=str, default='std', help='Training method.')
    parser.add_argument('--method_opt', type=dict, default={}, help='Parameters of training method.')

    # Validation
    parser.add_argument('--batch-size-validation', type=int, default=128, help='Batch size for testing.')
    parser.add_argument('--validation', action='store_true', default=False, help='split validation set for early stopping.')
    parser.add_argument('--adv-eval-freq', type=int, default=100, help='Adversarial evaluation frequency (in epochs).')

    return parser


def set_parser_eval(parser):
    parser.add_argument('--data-dir', type=str, default='../dataset_data')
    parser.add_argument('--desc', type=str, help='Description of model to be evaluated.')
    parser.add_argument('--log-dir', type=str, default='test_info')
    parser.add_argument('--batch-size-validation', type=int, default=128, help='Batch size for testing.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--attack_opt', type=dict, default={}, help='Parameters of evaluation attacks.')
    return parser