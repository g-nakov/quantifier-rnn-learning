from state_explorer import RNNStateExplorer, RNNState
from alphabet_utils import ALPHABET_W, PADDING_CHAR
from quant_trainer import QuantifierRnn, PaddingMode
import quantifiers
import data_gen
import numpy as np


PRINT_TO_STDOUT = True
RNNState.DEBUG = True
np.set_printoptions(precision=5, linewidth=10000, floatmode='fixed', suppress=True)


def get_all():
    al3_single = QuantifierRnn("data_al3/one_padding/use_me", quantifiers.at_least_three,
                               padding_mode=PaddingMode.SINGLE)
    al3_none = QuantifierRnn("data_al3/no_padding/use_me", quantifiers.at_least_three,
                             padding_mode=PaddingMode.NONE)
    al3_maxlen = QuantifierRnn("data_al3/max_padding/use_me", quantifiers.at_least_three,
                               padding_mode=PaddingMode.MAXLEN)

    m_single = QuantifierRnn("data_m/one_padding/use_me", quantifiers.most,
                             padding_mode=PaddingMode.SINGLE)
    f3_single = QuantifierRnn("data_f3/one_padding/use_me", quantifiers.first_three,
                              padding_mode=PaddingMode.SINGLE)

    return al3_single, al3_none, al3_maxlen, m_single, f3_single


def data_distribution(num_data_points=200000):
    if PRINT_TO_STDOUT:
        print "Data generation distribution:"
    generator = data_gen.DataGenerator(20, [quantifiers.at_least_three], mode='g',
                                       num_data_points=num_data_points, append_padding=False)
    training_data = generator.get_training_data()
    test_data = generator.get_test_data()
    lens_training = 21*[0]
    avg_len = lambda x : sum([k * i for (k, i) in enumerate(x)]) * 1. / sum(x)
    for datum in training_data:
        lens_training[int(np.sum(datum[0]) - 20)] += 1
    print lens_training
    print "Avg length: {} over {} data points ".format(avg_len(lens_training), sum(lens_training))

    lens_test = 21*[0]
    for datum in test_data:
        lens_test[int(np.sum(datum[0]) - 20)] += 1
    print lens_test
    print "Avg length: {} over {} data points".format(avg_len(lens_test), sum(lens_test))

    return lens_training, lens_test, lens_training, lens_test


def train_quantifiers():
    rnns = get_all()
    for r in rnns:
        r.train()


def eval_different_padding(maxlen=9):
    al3_single = QuantifierRnn("data_al3/one_padding/use_me",
                               quantifiers.at_least_three, padding_mode=PaddingMode.SINGLE)
    al3_none = QuantifierRnn("data_al3/no_padding/use_me",
                             quantifiers.at_least_three, padding_mode=PaddingMode.NONE)
    al3_max = QuantifierRnn("data_al3/max_padding/use_me",
                            quantifiers.at_least_three, padding_mode=PaddingMode.MAXLEN)

    len_one = (maxlen + 1) * [0]
    len_no = (maxlen + 1) * [0]
    len_max = (maxlen + 1) * [0]

    for i in range(2, maxlen+1):
        len_no[i] += al3_none.eval(i)
        len_one[i] += al3_single.eval(i+1)
        len_max[i] += al3_max.eval(i)
        if PRINT_TO_STDOUT:
            print "Loss for words up to {} with padding [no, single, max]: {}\t{}\t{}"\
                .format(i, len_no[i], len_one[i], len_max[i])

    return len_no, len_one, len_max


def eval_different_quantifiers(maxlen=9):
    al3 = QuantifierRnn("data_al3/one_padding/use_me", quantifiers.at_least_three, padding_mode=PaddingMode.SINGLE)
    m = QuantifierRnn("data_m/one_padding/use_me", quantifiers.most, padding_mode=PaddingMode.SINGLE)
    f3 = QuantifierRnn("data_f3/one_padding/use_me", quantifiers.first_three, padding_mode=PaddingMode.SINGLE)

    len_al3 = (maxlen+1) * [0]
    len_m = (maxlen+1) * [0]
    len_f3 = (maxlen+1) * [0]

    for i in range(2, maxlen+1):
        len_al3[i] += al3.eval(i+1)
        len_m[i] += m.eval(i+1)
        len_f3[i] += f3.eval(i+1)
        if PRINT_TO_STDOUT:
            print "Loss for (at_least_three, most, first_three) with words len up to {}: {}\t{}\t{}"\
                  .format(i, len_al3[i], len_m[i], len_f3[i])
    return len_al3, len_m, len_f3


def explore_states():
    rnn = RNNStateExplorer("data/one_padding/use_me/", num_layers=2, hidden_size=12)
    states = rnn[ALPHABET_W][ALPHABET_W][ALPHABET_W][ALPHABET_W][PADDING_CHAR]
    if PRINT_TO_STDOUT:
        print states
    return states.get_states()
