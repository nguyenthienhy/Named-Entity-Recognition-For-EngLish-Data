import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

MAX_WORD_NUMBER = 52
WORD_VECTOR_SIZE = 300

INPUT_PADDING = '<pad>'
TARGET_PADDING = 'O'

BATCH_SIZE = 128

NUM_EPOCH = 12

HIDDEN_DIM = 42