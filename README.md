## CVPR 2026 - Taming the Long Tail: Rebalancing Adversarial Training via Adaptive Perturbation

Official implementation based on PyTorch.

### Requirements

#### Dependencies 
```markdown
python==3.10
coloredlogs==15.0.1
matplotlib==3.10.6
numpy==2.3.2
pandas==2.3.2
Pillow==11.3.0
PyYAML==6.0.2
torch==2.6.0+cu126
torchattacks==3.5.1
torchvision==0.21.0+cu126
tqdm==4.67.1
```


### Example usage

#### 1. For training

Train wrn-28-10 by AT-BSL with and without RobustLT on CIFAR10-LT (imbalance ratio=50). 

```sh
python train.py ./configs/cifar10/robustlt_atbsl_cifar10.yaml --data cifar10 --imbalance_rate 50 --seed 102 --model wrn-28-10
python train.py ./configs/cifar10/atbsl_cifar10.yaml --data cifar10 --imbalance_rate 50 --seed 102 --model wrn-28-10
```

`train_multi.py` can be used to run `train.py` with multiprocessing.

#### 2. For evaluation

Test the natual and robust accuracies.

```sh
python eval.py ./configs/eval/nat.yaml --desc ./trained_models/cifar10lt/exp50/robustlt_atbsl_wrn-28-10_seed102
python eval.py ./configs/eval/pgd.yaml --desc ./trained_models/cifar10lt/exp50/robustlt_atbsl_wrn-28-10_seed102
python eval.py ./configs/eval/cw.yaml --desc ./trained_models/cifar10lt/exp50/robustlt_atbsl_wrn-28-10_seed102
python eval.py ./configs/eval/aa.yaml --desc ./trained_models/cifar10lt/exp50/robustlt_atbsl_wrn-28-10_seed102
```

### Citation
If you find this code useful for your research, please consider citing the following paper:

```markdown
@InProceedings{zhang2026taming,
    author    = {Zhang, Lilin and Guo, Yimo and Li, Yue and Shi, Jiancheng and Liu, Xianggen},
    title     = {Taming the Long Tail: Rebalancing Adversarial Training via Adaptive Perturbation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
}
```

