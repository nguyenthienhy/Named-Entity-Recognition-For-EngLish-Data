from gensim.models import KeyedVectors
from tensorflow.keras.utils import *
import constant
from path import *
import numpy as np

word_vectors = KeyedVectors.load_word2vec_format(get_path(constant.ROOT_DIR + "/data/google_w2v.bin"),
                                                 binary=True, limit=None)


def generate_data_batch(X, y, batch_size, entity_to_index, num_classes, shuffle=True):

    len_data = len(X)
    start_index = 0

    while True:

        if start_index + batch_size > len_data:
            start_index = 0
            if shuffle:
                perm = np.arange(len_data)
                np.random.shuffle(perm)
                X = X[perm]
                y = y[perm]

        word2vec_array = np.zeros([batch_size, constant.MAX_WORD_NUMBER, constant.WORD_VECTOR_SIZE], np.float32)
        for batch_idx, sentence in enumerate(X[start_index:start_index + batch_size]):
            for word_idx, word in enumerate(sentence):
                # Get google word2vector
                try:
                    word2vec_array[batch_idx, word_idx] = word_vectors[word]
                except Exception as e:
                    word2vec_array[batch_idx, word_idx] = np.random.uniform(low=-0.25, high=0.25, size=(1, 300))

        y_batch = y[start_index: start_index + batch_size]
        t = [[entity_to_index[word or 'O'] for word in sentence] for sentence in y_batch]
        target = np.array([to_categorical(i, num_classes=num_classes) for i in t])

        start_index += batch_size
        yield word2vec_array, target

def get_data_feature(X):
    word2vec_array = np.zeros([1, constant.MAX_WORD_NUMBER, constant.WORD_VECTOR_SIZE], np.float32)
    for word_idx, word in enumerate(X):
        try:
            word2vec_array[0, word_idx] = word_vectors[word]
        except Exception as e:
            word2vec_array[0, word_idx] = np.random.uniform(low=-0.25, high=0.25, size=(1, 300))
    return word2vec_array