import tensorflow as tf
import numpy as np
from quant_verify import INPUT_FEATURE


def pad_word(word, total_length=20):
    padding_char = np.zeros(5, dtype=np.float64)
    padding_char[4] = 1
    padding_cnt = total_length - len(word)
    if padding_cnt <= 0:
        return word
    return word + padding_cnt * [padding_char]


def get_data(total_length=20):
    a, b, c, d, _ = np.identity(5, dtype=np.float64)
    a[4] = b[4] = c[4] = d[4] = 1
    data = [a,b,c]
    return np.array([pad_word(data, total_length)])


def lstm_model_fn(features, params):
    cells = []
    for _ in range(params['num_layers']):
        cell = tf.nn.rnn_cell.LSTMCell(params['hidden_size'])
        cells.append(cell)
    multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    _, last_output = tf.nn.dynamic_rnn(multi_cell, features[INPUT_FEATURE], dtype=tf.float64)

    logits = tf.contrib.layers.fully_connected(inputs=last_output[params['num_layers']-1].h,
                                               num_outputs=params['num_classes'], activation_fn=None)
    probs = tf.nn.softmax(logits)
    #probs = tf.Print(probs, [i.h for i in last_output], "###LAST_OUTPUT:", summarize=400)
    h_layers = tf.stack([layer.h for layer in last_output], 1)
    outputs = {'probs': probs, 'h': h_layers}
    return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, predictions=outputs)


def run_test(save_dir, max_word_len=30):
    params = {'hidden_size': 12, 'num_layers': 2, 'num_classes':2}

    tf.reset_default_graph()

    # BUILD MODEL
    run_config = tf.estimator.RunConfig()

    model = tf.estimator.Estimator(
        model_fn=lstm_model_fn,
        model_dir=save_dir,
        params=params,
        config=run_config)

    for i in range(0, max_word_len + 1):
        word = get_data(i)
        predictor_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={INPUT_FEATURE: word}, shuffle=False)
        pred = model.predict(input_fn=predictor_input_fn).next()
        print "[word-len {} : {}]".format(len(word[0]), pred['probs'])


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.WARN)
    save_dir = "data/dfa_not_real/trial_0/"
    run_test(save_dir)