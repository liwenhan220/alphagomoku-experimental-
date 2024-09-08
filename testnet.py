import numpy as np
import torch
import torch.nn as nn

# 定义棋盘大小和分数
BOARD_SIZE = 9
SCORES = {
    'open_four': 1000,  # 两头空的四子
    'double_three': 800,  # 双三
    'open_three': 300,  # 两头空的三子
    'closed_four': 500,  # 单头堵四子
    'open_two': 100,  # 两头空的二子
    'closed_three': 200,  # 单头堵三子
    'closed_two': 50,  # 单头堵二子
    'other': 0  # 其他
}

# 五子棋评估函数
def evaluate_board(board):
    """
    评估当前棋盘状态。包括检测多种棋型并为其赋分。
    - board: 形状为 [7, board_size, board_size] 的输入
    第0通道: 黑棋位置
    第1通道: 白棋位置
    最后一个通道: 当前轮到的玩家（0为黑棋，1为白棋）

    返回一个表示局面对当前玩家的价值评分，范围为 [-1, 1]。
    """
    player = 1 if board[-1, 0, 0] == 0 else -1  # 判断是轮到黑棋(1)还是白棋(-1)
    
    # 获取黑棋和白棋的位置
    black_pieces = board[0]
    white_pieces = board[1]
    
    # 当前玩家和对手的棋盘
    current_pieces = black_pieces if player == 1 else white_pieces
    opponent_pieces = white_pieces if player == 1 else black_pieces
    
    score = 0
    
    # 检查横、纵、对角线方向上的棋型
    for direction in ['horizontal', 'vertical', 'diagonal1', 'diagonal2']:
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                current_pattern = extract_pattern(current_pieces, opponent_pieces, i, j, direction)
                opponent_pattern = extract_pattern(opponent_pieces, current_pieces, i, j, direction)
                
                score += evaluate_pattern(current_pattern, True)
                score -= evaluate_pattern(opponent_pattern, False)

    # 归一化分数，将其压缩到 [-1, 1] 之间
    score = np.tanh(score / 2000.0)  # 分数除以较大值来保证压缩效果

    return score

# 提取棋型
def extract_pattern(player_pieces, opponent_pieces, i, j, direction):
    """
    提取以 (i, j) 为起点，给定方向上的5子模式（如水平、竖直、对角线）。
    """
    pattern = []
    for step in range(5):
        if direction == 'horizontal':
            if j + step < BOARD_SIZE:
                pattern.append(player_pieces[i][j + step] - opponent_pieces[i][j + step])
        elif direction == 'vertical':
            if i + step < BOARD_SIZE:
                pattern.append(player_pieces[i + step][j] - opponent_pieces[i + step][j])
        elif direction == 'diagonal1':
            if i + step < BOARD_SIZE and j + step < BOARD_SIZE:
                pattern.append(player_pieces[i + step][j + step] - opponent_pieces[i + step][j + step])
        elif direction == 'diagonal2':
            if i + step < BOARD_SIZE and j - step >= 0:
                pattern.append(player_pieces[i + step][j - step] - opponent_pieces[i + step][j - step])
    return pattern

# 评估特定的5子棋型
def evaluate_pattern(pattern, is_player):
    """
    根据5子模式评估分数，找两头空的四、双三等棋型。
    """
    if len(pattern) != 5:
        return 0

    player_pieces = sum([1 for p in pattern if p == 1])
    opponent_pieces = sum([1 for p in pattern if p == -1])

    # 判断两头空的四子
    if player_pieces == 4 and opponent_pieces == 0:
        return SCORES['open_four']

    # 判断双三
    if player_pieces == 3 and opponent_pieces == 0:
        return SCORES['double_three']

    # 判断活三
    if player_pieces == 3 and opponent_pieces == 1:
        return SCORES['open_three']

    # 判断单头堵四子
    if player_pieces == 4 and opponent_pieces == 1:
        return SCORES['closed_four']

    # 判断两头空的二子
    if player_pieces == 2 and opponent_pieces == 0:
        return SCORES['open_two']

    # 判断单头堵三子
    if player_pieces == 3 and opponent_pieces == 2:
        return SCORES['closed_three']

    # 判断单头堵二子
    if player_pieces == 2 and opponent_pieces == 1:
        return SCORES['closed_two']

    return SCORES['other']

class GaussianPolicyNetwork(nn.Module):
    def __init__(self, board_size=9, scale=0.2):
        super(GaussianPolicyNetwork, self).__init__()
        self.board_size = board_size
        self.center_x = board_size // 2
        self.center_y = board_size // 2
        self.scale = scale  # 控制中心区域的权重

    def forward(self, state):
        """
        高斯分布策略网络：输出未归一化的 logits，优先选择靠近中心的位置。
        - state: 当前输入状态 [batch_size, 7, board_size, board_size]
        返回未归一化的 logits。
        """
        batch_size = state.shape[0]

        # 创建高斯分布的 logits 矩阵
        x, y = torch.meshgrid(torch.arange(self.board_size), torch.arange(self.board_size))
        x = x.float()
        y = y.float()

        # 计算到中心点的距离，并使用高斯分布公式生成 logits
        distance = (x - self.center_x) ** 2 + (y - self.center_y) ** 2
        gaussian_logits = -self.scale * distance  # 增加scale参数以提升中央权重

        # 展开成1维
        gaussian_logits = gaussian_logits.view(1, -1)
        gaussian_logits = gaussian_logits.repeat(batch_size, 1)  # 扩展为 batch_size 大小

        return gaussian_logits

# 组合策略和价值网络
class AlphaZeroNetwork(nn.Module):
    def __init__(self, board_size=9):
        super(AlphaZeroNetwork, self).__init__()
        self.policy_net = GaussianPolicyNetwork(board_size)
        self.board_size = board_size
        
    def forward(self, state):
        """
        结合策略和价值网络，返回策略和局面价值。
        - state: 当前棋盘状态 [batch_size, 7, board_size, board_size]
        返回：
        - policy_probs: 策略动作概率
        - value: 局面评估值（范围在-1到1之间）
        """
        # 先获取策略
        policy_probs = self.policy_net(state)

        # 计算局面价值
        batch_size = state.shape[0]
        values = torch.zeros(batch_size)
        for i in range(batch_size):
            board = state[i].cpu().numpy()  # 假设 state 是一个包含局面的张量
            value = evaluate_board(board)  # 根据输入格式进行评估
            values[i] = value

        return policy_probs, values

# # 示例使用
# if __name__ == "__main__":
#     # 初始化模型
#     network = AlphaZeroNetwork(board_size=9)

#     # 示例输入，形状为 [batch_size, 7, 9, 9]
#     input_state = torch.zeros((1, 7, 9, 9))  # 示例输入

#     # 获取策略和局面价值
#     policy_probs, value = network(input_state)

#     print("策略概率分布:", policy_probs)
#     print("局面价值:", value)
