
# -*- coding:utf-8  -*-
# Time  : 2021/02/26 16:25
# Author: Yutong Wu

from pathlib import Path  # 路径获取的库，.cwd（当前工作目录）和.home（用户的主目录）
import os
import yaml

def make_logpath(game_name, algo):
    base_dir = Path('../../model_saved')
    model_dir = base_dir / game_name / Path(algo)
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]  # 获取 model_dir 文件夹下所有run文件，统计编号
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir
    return run_dir, log_dir

def save_config(args, save_path):
    file = open(os.path.join(str(save_path), 'config.yaml'), mode='w', encoding='utf-8')
    yaml.dump(vars(args), file)
    file.close()

