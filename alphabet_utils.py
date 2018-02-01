import numpy as np

A, B, C, D, P = np.identity(5, dtype=np.float64)
A[4] = B[4] = C[4] = D[4] = 1
# alphabet as seq of chars
ALPHABET_C = [A, B, C, D]
ALPHABET_CP = ALPHABET_C + [P]
# alphabet as seq of 1-letter words
ALPHABET_W = [[c] for c in [A, B, C, D]]
ALPHABET_WP = ALPHABET_W + [[P]]

TRANS_DICT = {
    A.tobytes(): 'a',
    B.tobytes(): 'b',
    C.tobytes(): 'c',
    D.tobytes(): 'd',
    P.tobytes(): '.'}


def to_pretty_str(w):
    return "".join(map(lambda x: TRANS_DICT[x.tobytes()], w))


