from hyperparams import *
from cnn import *
import os
from hyperparams import *
from utils import save_model


if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)

model = Network(2 * HISTORY_REC + 1, NUM_FILTERS, NUM_BLOCKS, WIDTH, HEIGHT)
save_model(model, MODEL_DIR)

