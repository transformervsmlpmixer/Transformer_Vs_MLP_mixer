import os
import dill
import yaml
import numpy as np
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from my_utils import RatiosList
from pathlib import Path
from itertools import chain, combinations
from my_utils import get_run_id

import pandas as pd

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)
pd.set_option("display.max_rows", 2 ** 7, "display.max_columns", 10)


def get_criterion(config):
    criterion = nn.CrossEntropyLoss()
    return criterion


def one_d_loss(inp, target):
    loss = nn.MSELoss()(inp[:, 0].squeeze(), target)
    return loss


def get_model(config):
    if config.model_name == 'Mlp-Mixer':
        from mlp_mixer_pytorch import MLPMixer
        net = MLPMixer(
            image_size=config.img_size,
            patch_size=config.patch_size,
            embed_dim=config.width // config.heads,
            num_heads=config.heads,
            depth=config.depth,
            in_channels=3,
            num_classes=config.num_classes,
            dropout=0.5,
        )
    else:
        raise NotImplementedError(f"{config.model_name} is not implemented yet...")

    return net


def get_transform(config):
    train_transform = []
    test_transform = []
    train_transform += [
        transforms.RandomCrop(size=config.size, padding=config.padding)
    ]
    if config.dataset != 'svhn':
        train_transform += [transforms.RandomHorizontalFlip()]

    print(f"No AutoAugment for {config.dataset}")

    train_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std)
    ]

    test_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std)
    ]

    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    return train_transform, test_transform


def get_dataset(config):
    root = "data"
    if config.dataset == "c10":
        config.in_c = 3
        config.num_classes = 10
        config.size = 32
        config.padding = 4
        config.mean, config.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        train_transform, test_transform = get_transform(config)
        train_ds = torchvision.datasets.CIFAR10(root, train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR10(root, train=False, transform=test_transform, download=True)

    elif config.dataset == "c100":
        config.in_c = 3
        config.num_classes = 100
        config.size = 32
        config.padding = 4
        config.mean, config.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        train_transform, test_transform = get_transform(config)
        train_ds = torchvision.datasets.CIFAR100(root, train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR100(root, train=False, transform=test_transform, download=True)

    elif config.dataset == "svhn":
        config.in_c = 3
        config.num_classes = 10
        config.size = 32
        config.padding = 4
        config.mean, config.std = [0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]
        train_transform, test_transform = get_transform(config)
        train_ds = torchvision.datasets.SVHN(root, split="train", transform=train_transform, download=True)
        test_ds = torchvision.datasets.SVHN(root, split="test", transform=test_transform, download=True)

    else:
        raise NotImplementedError(f"{config.dataset} is not implemented yet.")

    return train_ds, test_ds


def get_experiment_name(config):
    experiment_name = f"{config.model_name}_{config.dataset}"
    print(f"Experiment:{experiment_name}")
    return experiment_name


################## MY CODE ########################


def update_config_using_config(config):
    with open("a_config.yaml", 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    config.max_epochs = int(data_loaded['max-epochs'])


def results_dir_path(run_id=-1):
    if run_id == -1:
        run_id = get_run_id()

    path = Path(f'./results/process_{run_id}_files/')
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_path(filename, run_id=-1):
    if run_id == -1:
        run_id = get_run_id()
    results_dir_path_ = results_dir_path(run_id)
    accuracies_file_name = f'{filename}.pickle'
    accuracies_path = results_dir_path_ / accuracies_file_name
    return accuracies_path


def get_train_loss_file_path(run_id=-1):
    path = get_file_path(filename='train_loss_dct', run_id=run_id)
    return path


def get_train_acc_file_path(run_id=-1):
    path = get_file_path(filename='train_acc_dct', run_id=run_id)
    return path


def get_val_loss_file_path(run_id=-1):
    path = get_file_path(filename='val_loss_dct', run_id=run_id)
    return path


def get_val_acc_file_path(run_id=-1):
    path = get_file_path(filename='val_acc_dct', run_id=run_id)
    return path


def results_saver(avg_loss_dct_, avg_acc_dct_, avg_val_loss_dct_, avg_val_acc_dct_):
    train_loss_filepath, train_acc_filepath = get_train_loss_file_path(), get_train_acc_file_path()
    val_loss_filepath, val_acc_filepath = get_val_loss_file_path(), get_val_acc_file_path()

    pickle_saver(path=train_loss_filepath, data_dct=avg_loss_dct_)
    pickle_saver(path=train_acc_filepath, data_dct=avg_acc_dct_)
    pickle_saver(path=val_loss_filepath, data_dct=avg_val_loss_dct_)
    pickle_saver(path=val_acc_filepath, data_dct=avg_val_acc_dct_)


def pickle_saver(path, data_dct):
    with open(path, 'wb') as pickle_file:
        dill.dump(data_dct, file=pickle_file)


def results_cleaner():
    import shutil
    results_path = results_dir_path().parent
    shutil.rmtree(results_path)


def get_minimax_epoch(data_dct_1, data_dct_2, data_dct_3, data_dct_4):
    num_epochs_1 = max(list(data_dct_1.keys()))
    num_epochs_2 = max(list(data_dct_2.keys()))
    num_epochs_3 = max(list(data_dct_3.keys()))
    num_epochs_4 = max(list(data_dct_4.keys()))
    minimax_epoch = min(num_epochs_1, num_epochs_2, num_epochs_3, num_epochs_4)
    return minimax_epoch


def data_processor(data_dct, mini_max_epoch):
    data_dct_1 = {key: value for key, value in data_dct.items() if key <= mini_max_epoch}
    return data_dct_1


def results_loader(ids_set=None):
    if ids_set is None:
        ids_set = {get_run_id()}

    runs_dct = {}

    max_epochs = [max(get_run_results(run_id)[0].keys()) for run_id in ids_set]
    minimax_epoch = sorted(max_epochs)[5]

    for run_id in ids_set:
        train_loss_dct, train_acc_dct, val_loss_dct, val_acc_dct = get_run_results(run_id)

        train_loss_dct = data_processor(train_loss_dct, minimax_epoch)
        train_acc_dct = data_processor(train_acc_dct, minimax_epoch)
        val_loss_dct = data_processor(val_loss_dct, minimax_epoch)
        val_acc_dct = data_processor(val_acc_dct, minimax_epoch)

        best_train_loss = min(train_loss_dct.values())
        best_val_loss = min(val_loss_dct.values())

        best_train_acc = max(train_acc_dct.values())
        best_val_acc = max(val_acc_dct.values())

        num_epochs = max(train_loss_dct.keys())
        ratios_dct = RatiosList().get_run_ratios(run_id=run_id)
        runs_dct[run_id] = {
            'depth': ratios_dct['depth'],
            'width': ratios_dct['width'],
            'num_params': ratios_dct['params'],
            'depth/(log2 width)': ratios_dct['depth'] / np.log2(ratios_dct['width']),
            'heads': ratios_dct['heads'],
            'best_train_loss': best_train_loss, 'best_train_acc': best_train_acc,
            'best_val_loss': best_val_loss, 'best_val_acc': best_val_acc,
            'num_epochs': num_epochs, }

    sorted_combs_by_acc = get_train_acc_combs_order(runs_dct)

    return runs_dct, sorted_combs_by_acc


def get_train_acc_combs_order(runs_dct):
    combs_lst = [(item['num_params'], item['depth/(log2 width)'], item['depth'],
                  item['width'], item['best_train_acc'], item['best_val_acc'])
                 for item in runs_dct.values()]
    sorted_combs_lst = sorted(combs_lst, key=lambda x: x[1])
    return sorted_combs_lst


def get_ids_set():
    ids_set = set()
    max_run_id = RatiosList().ratios_amount()
    for run_id in range(max_run_id):
        try:
            get_run_results(run_id)
            ids_set.add(run_id)
        except:
            pass
    print(f'After filtering exceptions there left {len(ids_set)} runs\n')
    return ids_set


from experiment_graph_plotter import graph_plotter


def results_table_printer():
    ids_set = get_ids_set()
    runs_dct, sorted_combs = results_loader(ids_set)

    graph_plotter(runs_dct)
    display_runs_dct(runs_dct, ids_set)
    display_spaces()
    display_sorted_combs(sorted_combs, ids_set)


def display_spaces():
    print('\n' * 2)
    print('=' * 25)
    print('\n' * 2)


def display_sorted_combs(sorted_combs, ids_set):
    columns = ['num_params', 'depth/(log2 width)', 'depth', 'width', 'best_train_acc', 'best_val_acc']
    data_dct = {i: list(item) for i, item in enumerate(sorted_combs)}
    df = pd.DataFrame.from_dict(data_dct, orient='index', columns=columns)
    msg = str(df)
    print(msg)
    return msg


def display_runs_dct(runs_dct, ids_set):
    columns = ['num_params', 'depth/(log2 width)', 'depth', 'width',
               'best_train_acc', 'best_val_loss', 'best_val_acc', 'num_epochs']
    runs_dct_1 = {run_id: [runs_dct[run_id][col_name] for col_name in columns]
                  for run_id in ids_set}

    df = pd.DataFrame.from_dict(runs_dct_1, orient='index', columns=columns)
    msg = str(df)
    print(msg)
    return msg


def get_run_results(run_id):
    train_loss_file_path, train_acc_file_path = get_train_loss_file_path(run_id), get_train_acc_file_path(run_id)
    val_loss_file_path, val_acc_file_path = get_val_loss_file_path(run_id), get_val_acc_file_path(run_id)

    train_loss_dct, train_acc_dct = pickle_loader(train_loss_file_path), pickle_loader(train_acc_file_path)

    val_loss_dct, val_acc_dct = pickle_loader(val_loss_file_path), pickle_loader(val_acc_file_path)

    return train_loss_dct, train_acc_dct, val_loss_dct, val_acc_dct


def pickle_loader(pickle_path):
    with open(pickle_path, 'rb') as pickle_file:
        data_dct = dill.load(pickle_file)
    return data_dct


##########################################################

MSG_PRINTED_BEFORE = False


def get_num_of_params(model):
    network = model.model
    with_embed = get_model_params(network)
    without_embed = get_model_params(network.mixers)
    return with_embed, without_embed


def get_model_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    number_of_parameters = sum([np.prod(p.size()) for p in model_parameters])
    return number_of_parameters


def config_updater(config):
    run_id = get_run_id()
    ratios_dct = RatiosList().get_run_ratios()
    config.depth = ratios_dct['depth']
    config.width = ratios_dct['width']
    config.heads = ratios_dct['heads']


if __name__ == '__main__':
    num_ratios = RatiosList().ratios_amount()
    print(f'Number of different run ratios {num_ratios}')
