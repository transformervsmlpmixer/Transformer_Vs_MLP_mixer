import os
import numpy as np


class RatiosList:
    def __init__(self):
        super(RatiosList, self).__init__()
        self.ratios = [2 ** i for i in range(-3, 4)]  # 7
        self.budgets = [1_000 * (2 ** i) for i in range(5, 9)]  # 4
        self.seeds = [20, 30, 40, 50, 60, 70]  # 6
        heads = 1
        self.ratios_list = []

        for budget in self.budgets:
            for ratio in self.ratios:
                for seed in self.seeds:
                    depth, width = self.get_width_and_depth(budget, ratio)
                    self.ratios_list.append({'seed': seed,
                                             'depth': depth, 'width': width, 'params': budget, 'heads': heads,
                                             'ratio': depth / np.log2(width)
                                             })

    def get_width_and_depth(self, budget, ratio):
        for depth in range(1, 130):
            width = self.get_width(budget, depth)
            current_ratio = depth / np.log2(width)
            if current_ratio < ratio:
                continue
            return depth, width

    def ratios_amount(self):
        return self.ratios_list.__len__()

    def get_run_ratios(self, run_id=None):
        if run_id is None:
            run_id = get_run_id()
        ratios_dct = self.ratios_list[run_id]
        return ratios_dct

    def get_width(self, param, depth):
        layer_params = param // depth
        width = int(np.sqrt(layer_params // 2))
        return width


def get_run_id():
    if 'SLURM_ARRAY_TASK_ID' not in os.environ:
        run_id = 68
    else:
        run_id = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
    return run_id
