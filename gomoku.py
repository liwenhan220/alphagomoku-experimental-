import numpy as np
import copy
import pdb
# Define the checkerboard class from the provided code
class Checkerboard:
    def __init__(self, width, height, win_count = 5, history_rec=3):
        self.width = width
        self.height = height
        self.emptyboard = np.zeros((height, width))
        self.board = np.zeros((height, width))
        # 1 as black, -1 as white
        self.black = 1
        self.white = -1
        self.currentplayer = self.black
        self.stones = 0
        self.history = [self.board]
        self.valid_actions = np.ones((height, width))
        self.win_count = win_count
        self.last_action = [-1, -1]
        self.history_rec = history_rec

    def reset(self):
        self.board = np.zeros((self.height, self.width))
        self.currentplayer = self.black
        self.stones = 0
        self.history = [self.board]
        self.valid_actions = np.ones((self.height, self.width))
        self.last_action = [-1, -1]
    
    def set(self, other):
        self.board = copy.deepcopy(other.board)
        self.currentplayer = other.currentplayer
        self.stones = other.stones
        self.history = copy.deepcopy(other.history)
        self.valid_actions = copy.deepcopy(other.valid_actions)
        self.last_action = other.last_action.copy()


    def swap_player(self):
        # Swap player
        self.currentplayer = self.white if self.currentplayer == self.black else self.black


    def step(self, x, y):
        if (x < 0 or x >= self.height or y < 0 or y >= self.width or self.board[x][y] != 0):
            print(f'Invalid actions encountered!, you cannot play ({x}, {y})')
            raise NameError()
            # return [False, 0]
        self.board[x][y] = self.currentplayer
        self.stones += 1
        self.valid_actions[x][y] = 0
        
        self.last_action = [x,y]
        self.history.append(self.board)

        # Check if anyone is winning
        for i, j in [[1, 0], [1, 1], [0, 1], [-1, 1]]:
            counter = 1
            for c in [1, -1]:
                ii = c * i
                jj = c * j
                xx = x + ii
                yy = y + jj
                while 0 <= xx < self.height and 0 <= yy < self.width and self.board[xx][yy] == self.currentplayer:
                    counter += 1
                    xx += ii
                    yy += jj
                if counter >= self.win_count:
                    player = self.currentplayer
                    self.swap_player()
                    return [True, player]
        
        # Check if draw
        if (self.stones == self.width * self.height):
            self.swap_player()
            return [True, 0]
        self.swap_player()
        return [False, 0]
    
    def get_state(self, history = None, num = None, current_player = None):
        if num is None:
            num = self.history_rec
        if history is None:
            history = self.history

        if current_player is None:
            current_player = self.currentplayer
        state = []
        for i in range(num):
            board = self.history[len(self.history) - i - 1 if len(self.history) - i - 1 > 0 else 0]
            board_state1 = np.array(board == self.black, dtype = np.uint8)
            board_state2 = np.array(board == self.white, dtype = np.uint8)
            state.append(board_state1)
            state.append(board_state2)
        state.append(np.full(board.shape, 0 if current_player == self.black else 1))
        state = np.array(state, dtype=np.uint8)
        return state
    
    def render(self):
        for i, row in enumerate(self.board):
            for j, item in enumerate(row):
                symb = ''
                if item == self.black:
                    if i == self.last_action[0] and j == self.last_action[1]:
                        symb = 'O'
                    else:
                        symb = 'o'
                elif item == self.white:
                    if i == self.last_action[0] and j == self.last_action[1]:
                        symb = 'X'
                    else:
                        symb = 'x'
                else:
                    symb = '.'
                print(symb, end = ' ')     
            print()
        print('\n\n')
    
    def get_current_player(self):
        return self.currentplayer



