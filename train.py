import json
import os
import shutil
import time
import torch
import pandas as pd
from core.data import get_data_info, load_imb_data, load_test_data
from core.trainer import seed, format_time, Logger, Trainer
from core.trainer.config import args, load_config

# Setup
load_config(train=True)
DATA_DIR = os.path.join(args.data_dir, args.data)
LOG_DIR = os.path.join(args.log_dir, f"{args.imbalance_type}{args.imbalance_rate}", args.desc)
WEIGHTS = os.path.join(LOG_DIR, 'weights-best.pt')
resume_path = None
if os.path.exists(LOG_DIR):
    print("File exists already.")
    resume_path = os.path.join(LOG_DIR, 'state-last.pt')
    if os.path.exists(resume_path):
        print('Try loading from the last checkpoint in the exist file. ')
        logger = Logger(os.path.join(LOG_DIR, 'log-train.log'))
        logger.transcribe = False
    else:
        print("No checkpoint saved.")
        shutil.rmtree(LOG_DIR)
        resume_path = None
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    logger = Logger(os.path.join(LOG_DIR, 'log-train.log'))
    logger.transcribe = True

# device = torch.device("cpu")
# args.device = "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.log(f'Using device: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu"}')
args.device = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
CUDA_LAUNCH_BLOCKING = 1
torch.backends.cudnn.benchmark = True
seed(args.seed)

ARGS_FILE = os.path.join(LOG_DIR, 'args.txt')
if os.path.exists(ARGS_FILE) and resume_path is not None:
    with open(ARGS_FILE, 'r') as f:
        cfg_special = json.load(f)
        all_keys = args.get_dict()
        for k in cfg_special:
            v = cfg_special[k]
            if k in all_keys and v != all_keys[k]:
                logger.log(f"ARG CONFLICT for {k}: old-{v} new-{all_keys[k]}")
else:
    with open(ARGS_FILE, 'w') as f:
        json.dump(args.get_dict(), f, indent=4)

args.device = device
info = get_data_info(DATA_DIR)

# Load data
logger.log("\n" + "*"*10 + ("%10s" % "DATASET") + "*"*10)
kwargs = {'num_workers': 1, 'pin_memory': torch.cuda.is_available()}
train_dataset, eval_dataset, train_dataloader, eval_dataloader = load_imb_data(
    data_dir=DATA_DIR, logger=logger, validation=args.validation,
    shuffle_train=True, balance_train=args.balance_train,
    batch_size=args.batch_size, batch_size_test=args.batch_size_validation,
    seed=args.seed, imb_rate=args.imbalance_rate, imb_type=args.imbalance_type,
    augmentation=args.augment, num_batches=args.num_batches, **kwargs,
)
test_dataset, test_dataloader = load_test_data(DATA_DIR, logger, batch_size_test=args.batch_size_validation, **kwargs)
del train_dataset, test_dataset, eval_dataset

logger.log("\n" + "*"*10 + ("%10s" % "TRAINER") + "*"*10)
trainer = Trainer(args, logger=logger, data_info=info, dataloader=train_dataloader, verbose=True)
NUM_EPOCHS = args.num_epochs

# Adversarial Training
if NUM_EPOCHS > 0:
    logger.log('\n\nAdversarial training for {} epochs'.format(NUM_EPOCHS))

start_epoch = 1
best_epoch, best_eval_score = -1, {'nat_acc': 0.0, 'adv_acc': 0.0}
if resume_path is not None:
    start_epoch = trainer.load_model(resume_path, weights_only=False) + 1
    logger.log(f'Resuming at epoch {start_epoch - 1}')
elif args.pre_resume_path:
    pre_epoch = trainer.load_model(args.pre_resume_path, weights_only=True) + 1
    logger.log(f'Resuming pre-trained model at {args.pre_resume_path} (pre-trained for {pre_epoch} epoch)')

def validation_based_earlystop(epoch, eval_acc, eval_adv_acc):
    global best_epoch, best_eval_score
    if eval_adv_acc >= best_eval_score['adv_acc']:
        best_eval_score['nat_acc'], best_eval_score['adv_acc'] = eval_acc, eval_adv_acc
        best_epoch = epoch
        trainer.save_model(WEIGHTS, epoch)

if NUM_EPOCHS >= start_epoch:
    logger.transcribe = True
    metrics = pd.DataFrame()
    test_acc = trainer.class_wise_eval(test_dataloader, adversarial=False)
    logger.add('test', 'nat_acc', test_acc * 100, start_epoch - 1)
    if eval_dataloader and os.path.exists(WEIGHTS):
        if os.path.exists(WEIGHTS):
            best_epoch = trainer.load_model(WEIGHTS, weights_only=False)
            eval_acc = trainer.class_wise_eval(eval_dataloader, adversarial=False)
            eval_adv_acc = trainer.class_wise_eval(eval_dataloader, adversarial=True)
            best_eval_score['nat_acc'], best_eval_score['adv_acc'] = eval_acc, eval_adv_acc
            start_epoch = trainer.load_model(resume_path, weights_only=False) + 1
            logger.log(f'Best checkpoint resuming at epoch {best_epoch}. ')
        else:
            eval_acc = trainer.class_wise_eval(eval_dataloader, adversarial=False)
            logger.add('eval', 'cw_nat_acc', eval_acc * 100, start_epoch - 1)
            eval_adv_acc = trainer.class_wise_eval(eval_dataloader, adversarial=True)
            logger.add('eval', 'cw_adv_acc', eval_adv_acc * 100, start_epoch - 1)
            validation_based_earlystop(start_epoch - 1, eval_acc, eval_adv_acc)
    logger.log_stats(start_epoch - 1, ['test', 'eval'])

for epoch in range(start_epoch, NUM_EPOCHS+1):
    logger.log('======= Epoch {} ======='.format(epoch))
    if trainer.scheduler is not None:
        last_lr = trainer.scheduler.get_last_lr()[0]
        logger.add('scheduler', 'lr', last_lr, epoch)

    start = time.time()
    res = trainer.train(epoch=epoch, verbose=True)
    for k in res:
        if 'acc' in k:
            logger.add('train', k, res[k] * 100, epoch)
    end = time.time()
    logger.add('time', 'train', format_time(end - start), epoch)

    start = time.time()
    test_acc = trainer.class_wise_eval(test_dataloader, adversarial=False, verbose=True)
    logger.add('test', 'nat_acc', test_acc * 100, epoch)
    if epoch % args.adv_eval_freq == 0:
        test_adv_acc = trainer.class_wise_eval(test_dataloader, adversarial=True, verbose=True)
        logger.add("test", "adv_acc", test_adv_acc * 100, epoch)

    if eval_dataloader:
        eval_acc = trainer.class_wise_eval(eval_dataloader, adversarial=False)
        logger.add('eval', 'cw_nat_acc', eval_acc * 100, epoch)
        eval_adv_acc = trainer.class_wise_eval(eval_dataloader, adversarial=True)
        logger.add('eval', 'cw_adv_acc', eval_adv_acc * 100, epoch)
        validation_based_earlystop(epoch, eval_acc, eval_adv_acc)

    end = time.time()
    logger.add('time', 'eval', format_time(end - start), epoch)
    trainer.save_model(os.path.join(LOG_DIR, 'state-last.pt'), epoch)

    logger.log_stats(epoch, ['train', 'test', 'eval', 'scheduler', 'time'])
    logger.plot_learning_curve()
    logger.save_stats('stats.pkl')

    if epoch == NUM_EPOCHS:
        if eval_dataloader:
            trainer.load_model(WEIGHTS)
            logger.log(
                'Best checkpoint:  epoch-{}  eval-adv-{:.2f}%  eval-adv-{:.2f}%.'.format(
                    best_epoch, best_eval_score['nat_acc'] * 100, best_eval_score['adv_acc'] * 100))

        os.system(f"python eval.py ./configs/eval/nat.yaml --desc {LOG_DIR}")
        os.system(f"python eval.py ./configs/eval/pgd.yaml --desc {LOG_DIR}")

logger.log('\nTraining completed.')