import tensorflow as tf
from state_explorer import RNNStateExplorer, RNNState
from alphabet_utils import *


tf.logging.set_verbosity(tf.logging.WARN)
np.set_printoptions(precision=3, linewidth=10000, floatmode='fixed', suppress=True)

ne = RNNStateExplorer("data/no_padding/use_me/", 2, 12)
me = RNNStateExplorer("data/max_padding/use_me/", 2, 12)
rnn = RNNStateExplorer("data/one_padding/use_me/", num_layers=2, hidden_size=12)
RNNState.DEBUG = True
es = [ne, me, rnn]

# print "#### ONE CHAR PADDING ####"
# word = [A,A,B,C]
# l = []
# for i in range(5, 21):
#     l += [me.run(pad_word(word, i))[0].is_accepted]
# print l

print rnn[ALPHABET_W][ALPHABET_W][ALPHABET_W][PADDING_CHAR]