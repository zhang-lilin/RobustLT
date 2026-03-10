import argparse
import json
import os
import pickle
import shutil
import socket
import numpy as np
from tqdm import tqdm
import torch
from core.data import get_data_info
from core.data import load_test_data
from core.models import Networks
from core.trainer import Logger, seed
from core.trainer.config import args, load_config
from core.metrics import _METRICS

# Setup
load_config(train=False)
if not os.path.exists(args.desc):
    print('File not found.')
    exit()
LOG_DIR = os.path.join(args.desc, args.log_dir)
if not os.path.isdir(LOG_DIR):
    os.mkdir(os.path.join(LOG_DIR))
attack_name = args.attack_name
test_log_path = os.path.join(LOG_DIR, 'log-test-{}.log'.format(attack_name))
test_save_path = os.path.join(LOG_DIR, f'{attack_name}.pt')
stats_path = os.path.join(LOG_DIR, 'eval_stats.pkl')
if os.path.exists(stats_path):
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    if attack_name in stats:
        if 'all' in stats[attack_name]:
            if args.save:
                if os.path.exists(test_save_path):
                    print('Already tested.')
                    exit()
                else:
                    pass
            else:
                print('Already tested.')
                exit()
if os.path.exists(test_log_path):
    os.remove(test_log_path)
if os.path.exists(test_save_path):
    os.remove(test_save_path)
logger = Logger(test_log_path)

host_info = "# " + ("%30s" % "Host Name") + ":\t" + socket.gethostname()
logger.log("#" * 120)
logger.log("----------Configurable Parameters In this Model----------")
logger.log(host_info)
for k in args.get_dict():
    logger.log("# " + ("%30s" % k) + ":\t" + str(args.__getattr__(k)))
logger.log("#" * 120)

ARG_PATH = os.path.join(args.desc, 'args.txt')
with open(ARG_PATH, 'r') as f:
    cfg_special = json.load(f)
    model_train_seed = cfg_special['seed']
    logger.log(f'Model training seed {model_train_seed}.')
    all_keys = args.get_dict()
    for k in cfg_special:
        if k in all_keys:
            pass
        else:
            v = cfg_special[k]
            if type(v) == bool:
                args.DEFINE_boolean("-" + k, "--" + k, default=argparse.SUPPRESS)
            else:
                args.DEFINE_argument(
                    "-" + k, "--" + k, default=argparse.SUPPRESS, type=type(v)
                )
            args.__setattr__(k, cfg_special[k])
            print("OLD ARG: {} with value {}".format(k, args.__getattr__(k)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.log('Using device: {}'.format(device))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
torch.backends.cudnn.benchmark = True

DATA_DIR = os.path.join(args.data_dir, args.data)
info = get_data_info(DATA_DIR)
test_dataset, test_dataloader = load_test_data(DATA_DIR, logger, batch_size_test=args.batch_size_validation, num_workers=0)
model = Networks(args=args, info=info, device=device)
WEIGHTS = os.path.join(args.desc, 'state-last.pt')
checkpoint = torch.load(WEIGHTS, weights_only=False)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint['epoch']
else:
    raise FileExistsError

logger.log(f'Resuming target model at {WEIGHTS} (epoch-{epoch}).')
if epoch < args.num_epochs:
    shutil.rmtree(LOG_DIR)
    print('Training unfinished.')
    exit()

num_classes = info['num_classes']
logger.log(f'Begin evaluation: {attack_name}.')
attack_opt = args.attack_opt
if attack_name == 'aa':
    attack_opt['n_classes'] = num_classes
adversary = _METRICS[attack_name](model, **attack_opt)

seed(args.seed)
model.eval()
all_true, all_pred = [], []
total, acc = [0 for _ in range(num_classes)], [0 for _ in range(num_classes)]
for x, y in tqdm(test_dataloader, desc="Evaluation : ", disable=False):
    x, y = x.to(device), y.to(device)
    x_adv = adversary(x, y)
    with torch.no_grad():
        y_pred = model(x_adv).argmax(1)
    correct_adv = y_pred == y
    for index, label in enumerate(y):
        total[label] += 1
        if correct_adv[index]:
            acc[label] += 1
    all_true.extend(y.cpu().numpy())
    all_pred.extend(y_pred.cpu().numpy())

all_acc = np.sum(acc) / np.sum(total) * 100
for i in range(num_classes):
    acc[i] = acc[i] / total[i] * 100

if os.path.exists(stats_path):
    with open(stats_path, "rb") as f:
        logger.stats = pickle.load(f)
for i in range(num_classes):
    logger.add(category=attack_name, k=i, v=acc[i], global_it=model_train_seed, unique=True)
    logger.log('class {}:  acc-{:.2f}%'.format(i, acc[i]))
logger.add(category=attack_name, k='all', v=all_acc, global_it=model_train_seed, unique=True)
logger.save_stats('eval_stats.pkl')

logger.log('Overall:  acc-{:.2f}%'.format(all_acc))
logger.log('\nTesting completed.')
