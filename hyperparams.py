import torch
import os
DATA_DIR = 'dataset'

# 获取所有 iteration 文件夹的编号
ITER_NUM = 0
NUM_GAMES = 1000

if os.path.isdir(DATA_DIR):
    iteration_numbers = []
    for iteration_dir in os.listdir(DATA_DIR):
        if iteration_dir.startswith('ITER_NUM_'):
            iter_num = int(iteration_dir.split('_')[-1])
            iteration_numbers.append(iter_num)
    # 找到最新的 iteration，并计算 start_iter
    if iteration_numbers:
        if len(os.listdir(os.path.join(DATA_DIR, f'ITER_NUM_{max(iteration_numbers)}'))) < NUM_GAMES:
            ITER_NUM = max(iteration_numbers)
        else:
            ITER_NUM = max(iteration_numbers) + 1

EXPLORE_COUNT = 10

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HEIGHT = 9
WIDTH = 9
HISTORY_REC = 3
CPUCT = 4.0
VIRTUAL_LOSS = 0.3
MCTS_BATCH_SIZE = 24
TRAIN_MINIBATCH_SIZE = 512
TRAIN_ITERS = 250
NUM_FILTERS = 64
NUM_BLOCKS = 5
MODEL_DIR = "models"
MIN_VISITS = WIDTH * HEIGHT
MAX_VISITS = 800