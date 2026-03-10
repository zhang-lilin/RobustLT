import itertools
import os
import multiprocessing



gpu_list = [0]

dataset = 'cifar10'
bases = [
    'pgdat',
    'awp',
    'robal',
    'reat',
    'atbsl',
    'taet',
]
enhances = ['robustlt']

methods = []
for base in bases:
    methods.append(base)
    for enh in enhances:
        methods.append(f'{base}_{enh}')

ARGS_FOR_TUNE = dict(
    data = [dataset],
    seed = [102,103,104,],
    model = ['wrn-28-10',],
    imbalance_rate = [50],
)

if __name__ == '__main__':

    commands = []
    tunner_groups = []
    for method in methods:
        tunner_groups.append(f"python train.py ./configs/{dataset}/{method}_{dataset}.yaml")

    command_template = " {}"
    for k in ARGS_FOR_TUNE:
        command_template += " --" + k + " {}"
    possible_value = []
    possible_value.append(tunner_groups)
    for k in ARGS_FOR_TUNE:
        possible_value.append(ARGS_FOR_TUNE[k])
    for args in itertools.product(*possible_value):
        commands.append(command_template.format(*args))
    print(commands)
    print("# experiments = {}".format(len(commands)))

    def exp_runner(para):
        cmd, gpus, proc_to_gpu_map = para[:1][0], para[1:2][0], para[2:3][0]
        process_id = multiprocessing.current_process().name
        if process_id not in proc_to_gpu_map:
            proc_to_gpu_map[process_id] = gpus.pop()
            print("assign gpu {} to {}".format(proc_to_gpu_map[process_id], process_id))
        gpuid = proc_to_gpu_map[process_id]
        return os.system("CUDA_VISIBLE_DEVICES={} ".format(gpuid) + cmd)

    gpus = multiprocessing.Manager().list(gpu_list)
    proc_to_gpu_map = multiprocessing.Manager().dict()

    p = multiprocessing.Pool(processes=len(gpu_list))
    para = [[com, gpus, proc_to_gpu_map] for com in commands]
    rets = p.map(exp_runner, para)
    p.close()
    p.join()
    print(rets)