from parallel_mcts import *
from cnn import Network
from gomoku import *
import os
import pickle
from hyperparams import *
from utils import *


if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)

ITER_DATA_DIR = os.path.join(DATA_DIR, f"ITER_NUM_{ITER_NUM}")
existing_games = 0
if not os.path.isdir(ITER_DATA_DIR):
    os.mkdir(ITER_DATA_DIR)
if len(os.listdir(ITER_DATA_DIR)) >= NUM_GAMES:
    raise NameError('This iteration of data already exists, start a new iteration')
if len(os.listdir(ITER_DATA_DIR)) < NUM_GAMES:
    existing_games = len(os.listdir(ITER_DATA_DIR))


env = Checkerboard(HEIGHT, WIDTH, win_count=5, history_rec=HISTORY_REC)
latest_model_pth = get_latest_model(MODEL_DIR)
cnn = Network(2*HISTORY_REC+1, NUM_FILTERS, NUM_BLOCKS, WIDTH, HEIGHT)
cnn.load_state_dict(torch.load(latest_model_pth))

for e in range(existing_games, NUM_GAMES):
    game_info = {'states': [], 'policies': [], 'values': []}
    mcts = MCTS(env, cnn, DEVICE)
    mcts.reset()
    gameEnd = False
    while not gameEnd:
        game_info['states'].append(env.get_state()) # record game state info
        if env.stones > EXPLORE_COUNT:
            mcts.add_dirichlet_noise(epsilon=0.03)
            while (sum(mcts.root.Ns) <= MAX_VISITS):
                mcts.search(MCTS_BATCH_SIZE)
        else:
            while (sum(mcts.root.Ns) <= MIN_VISITS):
                mcts.search(MCTS_BATCH_SIZE)
        mcts.policy_target_pruning(mcts.root)

        policy = mcts.root.Ns / sum(mcts.root.Ns)
        if env.stones <= EXPLORE_COUNT:
            policy = mcts.root.Ns / sum(mcts.root.Ns)
            best_a = np.random.choice(range(len(policy)), p=policy)
        else:
            policy = np.zeros(len(mcts.root.Ns))
            best_a = random_argmax(mcts.root.Ns)
            policy[best_a] = 0
        game_info['policies'].append(policy)
        print(latest_model_pth)
        print(f'game_num: {e}')
        mcts.game.render()
        x, y = mcts.A2xy(best_a)
        gameEnd, winner = mcts.step(x, y)

        if gameEnd:
            for state in game_info['states']:
                if winner == 0:
                    game_info['values'].append(0)
                else:
                    # Check player turn
                    if state[-1][0][0] == 0:
                        player = env.black
                    else:
                        player = env.white
                    if player == winner:
                        game_info['values'].append(1)
                    else:
                        game_info['values'].append(-1)
            print(f'winner is {winner}')
            env.render()
            with open(os.path.join(ITER_DATA_DIR, f'game_{e}.pkl'), 'wb') as f:
                pickle.dump(game_info, f)
            

