import os
import pickle
import random
from pprint import pprint
import numpy as np
from utils import *
from hyperparams import *

def random_transform(state, policy):
    """
    对 state 和 policy 进行随机旋转和翻转。
    state 形状为 (7, 9, 9)，policy 形状为 (9, 9)。
    返回变换后的 state 和 policy。
    """
    # 随机选择旋转次数（0次, 1次, 2次, 3次，每次90度）
    num_rotations = np.random.choice([0, 1, 2, 3])
    state = np.rot90(state, num_rotations, axes=(1, 2))  # 旋转9x9的部分
    policy = policy.reshape(state.shape[1], state.shape[2])
    policy = np.rot90(policy, num_rotations)  # 对应旋转 policy
    
    # 随机选择是否进行水平翻转
    if np.random.rand() > 0.5:
        state = np.flip(state, axis=1)  # 水平翻转
        policy = np.flip(policy, axis=0)  # 对应翻转 policy

    # 随机选择是否进行垂直翻转
    if np.random.rand() > 0.5:
        state = np.flip(state, axis=2)  # 垂直翻转
        policy = np.flip(policy, axis=1)  # 对应翻转 policy
    policy = policy.flatten()
    return state, policy

def collect_data_info(base_dir='dataset', recent_iters=10):
    data_info = []
    total_states = 0
    
    # 获取所有 iteration 文件夹的编号
    iteration_numbers = []
    for iteration_dir in os.listdir(base_dir):
        if iteration_dir.startswith('ITER_NUM_'):
            iter_num = int(iteration_dir.split('_')[-1])
            iteration_numbers.append(iter_num)
    
    # 找到最新的 iteration，并计算 start_iter
    if iteration_numbers:
        latest_iter = max(iteration_numbers)
        start_iter = max(latest_iter - recent_iters, 0)
    else:
        start_iter = 0
    
    # 遍历符合条件的 iteration 文件夹
    for iteration_dir in os.listdir(base_dir):
        iter_num = int(iteration_dir.split('_')[-1])
        if iter_num >= start_iter:
            iteration_path = os.path.join(base_dir, iteration_dir)
            if os.path.isdir(iteration_path):
                for game_file in os.listdir(iteration_path):
                    if game_file.endswith('.pkl'):
                        game_path = os.path.join(iteration_path, game_file)
                        with open(game_path, 'rb') as f:
                            game_data = pickle.load(f)
                            num_states = len(game_data['states'])
                            total_states += num_states
                            data_info.append((game_path, num_states))
    
    return data_info, total_states

def select_random_states(data_info, total_states, num_samples=256):
    selected_points = []
    
    selected_indices = sorted(random.sample(range(total_states), num_samples))

    current_index = 0
    for file_path, num_states in data_info:
        for i in range(num_states):
            if selected_indices and selected_indices[0] == current_index:
                selected_points.append((file_path, i))
                selected_indices.pop(0)
            current_index += 1
            if not selected_indices:
                break
    
    return selected_points

def load_selected_data(selected_points):
    states = []
    policies = []
    values = []
    for file_path, index in selected_points:
        with open(file_path, 'rb') as f:
            game_data = pickle.load(f)
            state = game_data['states'][index]
            policy = game_data['policies'][index]
            value = game_data['values'][index]
            state, policy = random_transform(state, policy)
            states.append(state)
            policies.append(policy)
            values.append(value)
    return np.array(states), np.array(policies), np.array(values)

def load_minibatch(num_samples = 256):
    base_dir = 'dataset'
    data_info, total_states = collect_data_info(base_dir)
    selected_points = select_random_states(data_info, total_states, num_samples=num_samples)

    # 加载 minibatch
    minibatch = load_selected_data(selected_points)
    return minibatch

import torch
import torch.nn as nn
import torch.optim as optim
from cnn import *

if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)

cnn = Network(2*HISTORY_REC+1, NUM_FILTERS, NUM_BLOCKS, WIDTH, HEIGHT)
cnn.load_state_dict(torch.load(get_latest_model(MODEL_DIR)))
for e in range(TRAIN_ITERS):
    minibatch = load_minibatch(TRAIN_MINIBATCH_SIZE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn.to(device)

    # 定义损失函数
    kl_loss_fn = nn.KLDivLoss(reduction='batchmean')  # KL 散度损失
    mse_loss_fn = nn.MSELoss()  # 均方误差损失

    # 定义优化器，包含 L2 正则化 (weight_decay)
    optimizer = optim.SGD(cnn.parameters(), lr=2e-5, momentum = 0.9, weight_decay=1e-4)

    # 转换 minibatch 数据到 tensor
    states_tensor = torch.tensor(minibatch[0], dtype=torch.float32).to(device)
    policies_tensor = torch.tensor(minibatch[1], dtype=torch.float32).to(device)
    values_tensor = torch.tensor(minibatch[2], dtype=torch.float32).to(device)

    # 训练步骤
    cnn.train()
    optimizer.zero_grad()

    # Forward pass
    logits, pred_values = cnn(states_tensor)
    pred_policies = torch.softmax(logits, dim=1)

    # 计算 policy 的 KL 散度损失
    log_pred_policies = torch.log_softmax(pred_policies, dim=1)  # log softmax 后用于 KLDivLoss
    kl_loss = kl_loss_fn(log_pred_policies, policies_tensor)

    # 计算 value 的 MSE 损失
    mse_loss = mse_loss_fn(pred_values.squeeze(), values_tensor)

    # 总损失
    loss = kl_loss + mse_loss

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    print(f'KL_Loss: {kl_loss.item()}, mse_Loss: {mse_loss.item()}, iter: {e}')

save_model(cnn, MODEL_DIR)
