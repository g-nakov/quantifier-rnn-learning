import tensorflow as tf
import numpy as np
from state_explorer import RNNStateExplorer

A, B, C, D, P = np.identity(5, dtype=np.float64)
A[4] = B[4] = C[4] = D[4] = 1
ALPHABET = [[A], [B], [C], [D]]
WORD = np.array([A, A, C])
DICT = {A.tobytes():'a', B.tobytes():'b', C.tobytes(): 'c', D.tobytes():'d', P.tobytes():'.'}


def pad_word(word, up_to=20):
    padding_char = np.zeros(5, dtype=np.float64)
    padding_char[4] = 1
    padding_cnt = up_to - len(word)
    if padding_cnt <= 0:
        return word
    return word + padding_cnt * [padding_char]


tf.logging.set_verbosity(tf.logging.WARN)
np.set_printoptions(precision=5, linewidth=10000, floatmode='fixed', suppress=True)

ne = RNNStateExplorer("data/no_padding/use_me/", 2, 12)
me = RNNStateExplorer("data/max_padding/use_me/", 2, 12)
oe = RNNStateExplorer("data/one_padding/use_me/", 2, 12)
#RNNState.DEBUG = True
es = [ne, me, oe]

print "#### NO PADDING ####"
print ne[A, A, B]
print ne[A, A, B, P]
print ne[A, A, B, P, P]
print "#### MAX LEN PADDING ####"
print me[A, A, B]
print me[A, A, B, P]
print me[A, A, B, P, P]
print "#### ONE CHAR PADDING ####"
print oe[A, A, B]
print oe[A, A, B, P]