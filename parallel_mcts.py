import numpy as np
from cnn import *
import torch
from torch import nn
from gomoku import Checkerboard
import math
import copy
import random
from hyperparams import *

def random_argmax(arr):
    max_value = np.max(arr) 
    max_indices = np.where(arr == max_value)[0]
    return random.choice(max_indices)

class Node:
    def __init__(self, parent, from_action, Ps, current_player):
        self.parent = parent
        self.from_action = from_action
        self.Ps = Ps
        self.Ns = np.zeros((len(Ps)))
        self.Ws = np.zeros((len(Ps)))
        self.Qs = np.zeros((len(Ps)))
        self.virtual_losses = np.zeros((len(Ps)))
        self.children = [None] * len(Ps)
        self.just_expanded = True
        self.current_player = current_player
    

class DummyNode:
    def __init__(self, parent, from_action):
        self.parent = parent
        self.from_action = from_action
        self.just_expanded = True
    
class MCTS:
    def __init__(self, game:Checkerboard, cnn:Network, device):
        self.game = game
        self.copy_game = copy.deepcopy(game)
        self.nn = cnn.to(device)
        self.nn.eval()
        self.game.reset()
        logits= self.nn(torch.Tensor(np.array([self.game.get_state()])).to(device))[0].cpu().detach()
        Ps = nn.Softmax(dim=1)(logits)[0].numpy()
        self.root = Node(parent=None, from_action= -1, Ps=Ps, current_player=self.copy_game.get_current_player())
        self.cpuct = CPUCT
        self.device = device
        self.virtual_loss = VIRTUAL_LOSS
        self.epsilon = 1e-6
        self.k = 2
        self.alpha = self.calculate_alpha_grid(game.width, game.height).flatten()


    def calculate_alpha_grid(self, width, height):
        alpha_grid = np.zeros((height, width))
        center = (height // 2, width // 2)

        for i in range(width):
            for j in range(height):
                # 计算距离中心点的距离，距离越近 alpha 值越大
                distance = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
                alpha_grid[i, j] = 1 / (1 + distance)
        
        # 归一化 alpha 值
        alpha_grid = alpha_grid / np.sum(alpha_grid)
        return alpha_grid

    def reset(self):
        self.game.reset()
        self.nn.eval()
        logits= self.nn(torch.Tensor(np.array([self.game.get_state()])).to(self.device))[0].cpu().detach()
        Ps = nn.Softmax(dim=1)(logits)[0].numpy()
        self.root = Node(parent=None, from_action= 0, Ps=Ps, current_player=self.copy_game.get_current_player())

    def xy2A(self, x, y):
        return x * self.game.width + y
    
    def A2xy(self, a):
        return a // self.game.width, a % self.game.width

    def compute_n_forced(self, node):
        return np.sqrt(self.k * node.Ps * np.sum(node.Ns))
    
    def cal_PUCT(self, node):
        N = np.sum(node.Ns)
        PUCT = self.cpuct * (node.Ps + self.epsilon)* math.sqrt(np.sum(N)) / (1 + node.Ns) + node.Qs - node.virtual_losses
        return PUCT

    def expand_to_leaf(self):
        self.copy_game.set(self.game)
        node = self.root
        while True:
            PUCT = self.cal_PUCT(node)
            if node == self.root and sum(node.Ns) > 0:
                n_forced = self.compute_n_forced(node)
                urgency_list = (node.Ns >= 1) & (n_forced > node.Ns) & (node.virtual_losses == 0)
                PUCT[urgency_list] = np.inf

            PUCT = np.where(self.copy_game.valid_actions.flatten(), PUCT, -np.inf)  # Masking invalid actions
            # if max(PUCT) == np.inf:
            #     best_a = random_argmax(PUCT)
            # else:
            #     best_a = np.argmax(PUCT)
            best_a = random_argmax(PUCT)
            x, y = self.A2xy(best_a)
            game_end, winner = self.copy_game.step(x, y)
            node.virtual_losses[best_a] += self.virtual_loss
            if game_end:
                if node.children[best_a] is None:
                    node.children[best_a] = DummyNode(parent=node, from_action=best_a)
                if winner != 0:
                    if self.copy_game.get_current_player() != winner:
                        v = -1
                    else:
                        v = 1
                else:
                    v = 0
                self.backpropagate(node.children[best_a], v, self.copy_game.get_current_player())
                return None
            if node.children[best_a] is None:
                # Create a new leaf
                node.children[best_a] = Node(parent = node, from_action=best_a, Ps=np.zeros(self.game.width * self.game.height), current_player=self.copy_game.get_current_player())
                return [node.children[best_a], self.copy_game.get_state(), self.copy_game.get_current_player()]
            node = node.children[best_a]
            
    def backpropagate(self, node:Node, v, player):
        curNode = node
        while curNode.parent is not None:
            curNode.just_expanded = False
            curNode.parent.Ns[curNode.from_action] += 1
            if curNode.parent.current_player != player:
                curNode.parent.Ws[curNode.from_action] -= v
            else:
                curNode.parent.Ws[curNode.from_action] += v
            curNode.parent.Qs[curNode.from_action] = curNode.parent.Ws[curNode.from_action] / curNode.parent.Ns[curNode.from_action]
            curNode.parent.virtual_losses[curNode.from_action] -= self.virtual_loss
            curNode = curNode.parent

    def policy_target_pruning(self, node):
        # 计算当前所有子节点的 PUCT 值
        PUCT_values = self.cal_PUCT(node)
        max_N = max(node.Ns)
        # 找到拥有最大 PUCT 值的子节点 c*
        PUCT_star = np.max(PUCT_values)
        # star_index = np.argmax(PUCT_values)

        n_forced = self.compute_n_forced(node).astype(np.uint8)
        
        # 对每个其他子节点执行 pruning
        for i in range(len(node.Ns)):
            if node.Ns[i] == max_N:
                continue  # 跳过拥有最大 PUCT 的子节点
            
            # 逐步减少 n_forced
            while node.Ns[i] > 0 and n_forced[i] > 0:
                node.Ns[i] -= 1
                n_forced[i] -= 1
                N_new = np.sum(node.Ns)
                U_new = self.cpuct * (node.Ps[i] + self.epsilon) * math.sqrt(N_new) / (1 + node.Ns[i])
                PUCT_new = node.Qs[i] + U_new
                
                if PUCT_new >= PUCT_star:
                    node.Ns[i] += 1  # 如果减少过多，则回退一步
                    break
            if node.Ns[i] <= 1:
                node.Ns[i] = 0


    def search(self, batch_size = MCTS_BATCH_SIZE):
        states = []
        nodes = []
        players = []
        for _ in range(batch_size):
            info = self.expand_to_leaf()
            if info is not None:
                states.append(info[1])
                nodes.append(info[0])
                players.append(info[2])

        if len(states) == 0:
            return
        states = np.array(states)
        with torch.no_grad():
            self.nn.eval()
            logits, vs = self.nn(torch.Tensor(states).to(self.device))
        Pss = nn.Softmax(dim=1)(logits)
        for i in range(len(states)):
            Ps = Pss[i].cpu().numpy()
            v = float(vs[i])
            node = nodes[i]
            node.Ps = Ps
            self.backpropagate(node, v, players[i])

    def step(self, x, y):
        gameEnd, winner = self.game.step(x, y)
        a = self.xy2A(x, y)
        if self.root.children[a] is not None:
            self.root = self.root.children[a]
            self.root.parent = None
        else:
            logits= self.nn(torch.Tensor(np.array([self.game.get_state()])).to(self.device))[0].cpu().detach()
            Ps = nn.Softmax(dim=1)(logits)[0].numpy()
            self.root = Node(parent=None, from_action=-1, Ps=Ps, current_player=self.copy_game.get_current_player())
        return gameEnd, winner
    
    def recommend_action(self, N_based = True):
        if N_based:
            return random_argmax(self.root.Ns)
        return random_argmax(np.where(self.game.valid_actions.flatten(), self.root.Qs, float('-inf')))
    
    def add_dirichlet_noise(self, epsilon=0.25, alpha=0.15):
        dirichlet_noise = np.random.dirichlet([alpha] * len(self.root.Ps))

        self.root.Ps = (1 - epsilon) * self.root.Ps + epsilon * dirichlet_noise


def test_MCTS():
    game = Checkerboard(3, 3, win_count=3)
    cnn = DummyNN()


    mcts = MCTS(game, cnn, 'cuda')
    mcts.reset()
    gameEnd = False


    import time
    while not gameEnd:
        while (sum(mcts.root.Ns) <= 400):
            mcts.search(8)
        best_a = mcts.recommend_action()
        # print(max(np.where(game.valid_actions.flatten(), mcts.root.Qs, float('-inf')) + 1)/2)
        print(mcts.root.Ns.reshape((mcts.game.height, mcts.game.width)))
        mcts.game.render()
        x, y = mcts.A2xy(best_a)
        gameEnd, winner = mcts.step(x, y)

        if gameEnd:
            print(f'winner is {winner}')
            mcts.game.render()


# # 初始化 MCTS 对象
# mcts = MCTS(game=Checkerboard(3,3,3), cnn=Network(2*3+1,28,2,3,3), device='cpu')

# # 执行测试
# test_mcts_implementation(mcts)

# test_MCTS()
