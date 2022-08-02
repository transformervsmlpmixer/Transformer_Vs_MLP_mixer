import matplotlib.pyplot as plt
import numpy as np

from my_utils import RatiosList


# 'depth': 1,
# 'width': 89,
# 'num_params': 16000,
# 'depth/(log2 width)': 0.15442266280111014,
# 'heads': 1,
# 'best_train_loss': 2.8450566106440163,
# 'best_train_acc': 0.2866528132916107,
# 'best_val_loss': 2.734748094896727,
# 'best_val_acc': 0.3133900316455696,
# 'num_epochs': 41}

# All on one canvas

# Budget_graphs_list = {B:{p_divided_by_log_2_d: acc_after_90_epochs} for B in possible_budgets}
def graph_plotter(results_dct, epochs):
    ratios_obj = RatiosList()
    ratios_list = ratios_obj.ratios_list
    budget_graphs_list = {f'{ratio_dct["params"]}_{ratio_dct["seed"]}': {}
                          for ratio_dct in ratios_list
                          # if ratio_dct["params"] // 10 ** 3 > 17
                          }

    # move_const = .1
    for cnt, results_dct in enumerate(results_dct.values()):
        seed = ratios_list[cnt]['seed']
        budget, ratio = results_dct['num_params'], results_dct['depth/(log2 width)']
        if ratio not in budget_graphs_list[f'{budget}_{seed}']:
            budget_graphs_list[f'{budget}_{seed}'][ratio] = {}
        budget_graphs_list[f'{budget}_{seed}'][ratio] = results_dct['best_val_acc']

    budget_graphs_lists_2 = {
        f'{ratio_dct["params"]}_{ratio_dct["seed"]}': ([], [])
        for ratio_dct in ratios_list
        # if ratio_dct["params"] // 10 ** 3 > 17
    }

    for budget_seed, dct in budget_graphs_list.items():
        for ratio, acc in dct.items():
            budget_graphs_lists_2[budget_seed][0].append(ratio)
            budget_graphs_lists_2[budget_seed][1].append(acc)

    # plt.rcParams['text.usetex'] = True

    log_best_ratios(budget_graphs_lists_2)

    color_cnt = 0
    colors = ['blue', 'red', 'green', 'black', 'purple', 'brown'] * ratios_obj.seeds.__len__()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    labels_lst = []

    for budget_ in ratios_obj.budgets:
        all_accs = []
        ratios_ = None

        # Still shows but - overfit issues - find harder DS
        if budget_ >= 16000:
            continue

        for budget_seed, (ratios, accs) in budget_graphs_lists_2.items():
            budget, seed = int(budget_seed.split('_')[0]), budget_seed.split('_')[1]
            if budget != budget_:
                continue
            if len(all_accs) > 0:
                all_accs = [acc_1 + [acc_2, ] for acc_1, acc_2 in zip(all_accs, accs)]
            else:
                all_accs = [[acc] for acc in accs]
                ratios_ = ratios
        avg_accs = [np.mean(acc) for acc in all_accs]
        max_deviation = [np.max(np.array(acc) - np.mean(acc)) for acc in all_accs]
        # ax.plot(ratios[1:], accs[1:], color=colors[color_cnt % len(colors)], label=f"{int(budget // 10 ** 3)}K")
        plt.errorbar(ratios_[1:], avg_accs[1:], yerr=max_deviation[1:], uplims=[False, ] * len(ratios_[1:]),
                     lolims=[False, ] * len(ratios_[1:]), color=colors[color_cnt % len(colors)],
                     label=f"{int(budget_ // 10 ** 3)}K", capsize=6)

        # plt.plot(ratios_[1:], avg_accs[1:], color=colors[color_cnt % len(colors)],
        #          label=f"{int(budget_ // 10 ** 3)}K")

        ax.set_xscale('log')
        ax.set_xticks([1 / k for k in range(1, 6)] + [k for k in range(1, 10)])
        ax.set_xticklabels([f'1/{k}' for k in range(1, 6)] + [f'{k}' for k in range(1, 10)])

        print('=' * 15)
        print(budget_)
        print(avg_accs)
        print(max_deviation)
        print('=' * 15)

        color_cnt += 1

        # ax.legend(loc='upper right', title=f'{int(budget_ // 10 ** 3)}K')
        labels_lst.append(f"{int(budget_ // 10 ** 3)}K")
        plt.xlabel(r'p / log_2 d')
        plt.ylabel('accuracy')

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, _ = [sum(var, []) for var in zip(*lines_labels)]
    plt.legend(lines, labels_lst, loc='upper right', title='Budget')

    plt.title(f'SVHN ({epochs} epochs)')
    plt.show()
    plt.cla()


def log_best_ratios(budget_graphs_lists_2):
    budget_best_ratio = {}
    for budget_seed, (ratios, accs) in budget_graphs_lists_2.items():
        budget, seed = int(budget_seed.split('_')[0]), budget_seed.split('_')[1]
        if budget not in budget_best_ratio:
            budget_best_ratio[budget] = {}
        best_ratio = ratios[np.argmax(accs)]
        if best_ratio not in budget_best_ratio[budget]:
            budget_best_ratio[budget][best_ratio] = 0
        budget_best_ratio[budget][best_ratio] += 1
    budget_best_ratio_2 = {budget: list(best_ratios_counter.keys())[np.argmax(best_ratios_counter.values())]
                           for budget, best_ratios_counter in budget_best_ratio.items()}
    print(budget_best_ratio_2)
