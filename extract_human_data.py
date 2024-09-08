from gomoku import *
from hyperparams import *
data_source = 'Freestyle2/0_0_1_1.psq'

with open(data_source, 'r') as f:
    moves = f.read().split()
    print(moves)

for ele in moves[4:len(moves)-4]:
    info = ele.split(',')
    x = int(info[0]) - 1
    y = int(info[1]) - 1
    if x > 19 or y > 19:
        print(f"Invalid coordinate found: ({x}, {y})")
        quit()

game = Checkerboard(WIDTH, HEIGHT)
game.reset()
for move in moves[4:len(moves)-4]:
    info = move.split(',')
    x = int(info[0]) - 1
    y = int(info[1]) - 1
    print(move)
    print(x, y)
    game.step(x, y)
    game.render()
