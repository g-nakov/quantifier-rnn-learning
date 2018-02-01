import tensorflow as tf
import numpy as np
from alphabet_utils import to_pretty_str

TFTYPE = tf.float64
NPTYPE = np.float64
FIXED_NUM_CLS = 2

KEY_IN_DATA = "X"
KEY_IN_H = "initial_h"
KEY_IN_C = "initial_c"

KEY_OUT_PROB = "probs"
KEY_OUT_H = "last_h"
KEY_OUT_C = "last_c"


def fix_words_input(words):
    if type(words) is not np.ndarray:
        words = np.array(words, dtype=NPTYPE)
    if len(words.shape) == 1:
        words = np.array([[words]])
    elif len(words.shape) == 2:
        words = np.array([words])
    return words


class RNNState(object):
    DEBUG = False

    def __init__(self, label, state, parent=None):
        label = to_pretty_str(label)
        self.label = label if parent is None else "{}/{}".format(parent.label, label)
        self.is_accepted = state[KEY_OUT_PROB][0]
        self.h = state[KEY_OUT_H]
        self.c = state[KEY_OUT_C]
        self.state = state

    def is_accepted(self):
        return self.is_accepted > 0.5

    def __str__(self):
        head = "< {}: {},".format(self.label, self.is_accepted)
        if RNNState.DEBUG:
            return "{}\n  h:\n{},\n  c:\n{} >".format(head, self.h, self.c)
        return "{}  h: shape{}, c: shape{} >".format(head, self.h.shape, self.c.shape)

    __repr__ = __str__


# this is MUTABLE!

class RNNExplorerWithInternalState(object):

    def __init__(self, parent, states):
        self.parent = parent
        self.states = states

    def __getitem__(self, words):
        new_states = [ns for x in[self.parent.run(words, s) for s in self.states] for ns in x]
        self.states = new_states
        return self

    def get_states(self):
        return self.states

    def __str__(self):
        out = "["
        out += ", \n ".join([item.__str__() for item in self.states])
        return out + "]"


class RNNStateExplorer(object):

    def __init__(self, saved_dir, num_layers, hidden_size, alphabet=[], ):
        self.saved_dir = saved_dir
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.alphabet = alphabet

        self.char_word = map(lambda x: np.array(x), alphabet)
        self.model = self._build_model()

    def _build_model(self):
        tf.reset_default_graph()
        empty = tf.estimator.RunConfig()
        return tf.estimator.Estimator(model_fn=lambda features: self._lstm_model_fn(features),
                                      model_dir=self.saved_dir, config=empty)

    def _lstm_model_fn(self, features):
        cells = []
        for _ in range(self.num_layers):
            cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
            cells.append(cell)
        multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

        try:
            # shape [batch_size, num_layers, hidden_size]
            initial_h = features[KEY_IN_H]
            initial_c = features[KEY_IN_C]
        except KeyError:
            initial_h = initial_c = None
        states = None
        if initial_h is not None and initial_c is not None:
            # LONG LIVE THE DEBUGGER
            # shape [num_layers* (batch_size, hidden_size)]
            initial_h = tf.unstack(initial_h, axis=1)
            initial_c = tf.unstack(initial_c, axis=1)
            states = tuple([tf.nn.rnn_cell.LSTMStateTuple(*i) for i in zip(initial_c, initial_h)])

        _, last_ts_output = tf.nn.dynamic_rnn(multi_cell, features[KEY_IN_DATA], dtype=TFTYPE,
                                              initial_state=states)

        logits = tf.contrib.layers.fully_connected(inputs=last_ts_output[self.num_layers - 1].h,
                                                   num_outputs=FIXED_NUM_CLS, activation_fn=None)
        probs = tf.nn.softmax(logits)
        # -- [batch_size, num_layers, hidden_size]
        last_hs = tf.stack([layer.h for layer in last_ts_output], axis=1)
        last_cs = tf.stack([layer.c for layer in last_ts_output], axis=1)
        outputs = {KEY_OUT_PROB: probs, KEY_OUT_H: last_hs, KEY_OUT_C: last_cs}
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, predictions=outputs)

    # state is RNNState wrapped nparays: h = [num_layers, hidden_size], c = [num_layers, hidden_size]
    def run(self, words, state=None):
        words = fix_words_input(words)
        batch_size = words.shape[0]
        shape = [batch_size, self.num_layers, self.hidden_size]
        if state is not None:
            h, c = map(lambda x: np.broadcast_to(x, shape), (state.h, state.c))
        else:
            h = c = np.zeros(shape, dtype=NPTYPE)
        input_dict = {KEY_IN_DATA: words, KEY_IN_H: h, KEY_IN_C: c}
        predictor_input_fn = tf.estimator.inputs.numpy_input_fn(input_dict, shuffle=False)
        return [RNNState(words[idx], pred, state) for (idx, pred) in enumerate(self.model.predict(predictor_input_fn))]

    def __getitem__(self, item):
        return RNNExplorerWithInternalState(self, self.run(item))
