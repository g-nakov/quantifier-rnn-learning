from collections import defaultdict
import tensorflow as tf
import numpy as np
import data_gen
import quantifiers
import util
from enum import Enum

INPUT_FEATURE = 'x'


class PaddingMode(Enum):
    NONE = 0
    SINGLE = 1
    MAXLEN = 2


class EvalEarlyStopHook(tf.train.SessionRunHook):
    """Evaluates estimator during training and implements early stopping.

    Writes output of a trial as CSV file.

    See https://stackoverflow.com/questions/47137061/. """

    def __init__(self, estimator, eval_input, filename,
                 num_steps=50, stop_loss=0.05):

        self._estimator = estimator
        self._input_fn = eval_input
        self._num_steps = num_steps
        self._stop_loss = stop_loss
        # store results of evaluations
        self._results = defaultdict(list)
        self._filename = filename

    def begin(self):

        self._global_step_tensor = tf.train.get_or_create_global_step()
        if self._global_step_tensor is None:
            raise ValueError("global_step needed for EvalEarlyStop")

    def before_run(self, run_context):

        requests = {'global_step': self._global_step_tensor}
        return tf.train.SessionRunArgs(requests)

    def after_run(self, run_context, run_values):

        global_step = run_values.results['global_step']
        if (global_step-1) % self._num_steps == 0:
            ev_results = self._estimator.evaluate(input_fn=self._input_fn)

            print ''
            for key, value in ev_results.items():
                self._results[key].append(value)
                print '{}: {}'.format(key, value)

            # TODO: add running total accuracy or other complex stop condition?
            if ev_results['loss'] < self._stop_loss:
                run_context.request_stop()

    def end(self, session):
        # write results to csv
        util.dict_to_csv(self._results, self._filename)


class QuantifierRnn(object):

    def __init__(self, working_dir, quantifier, num_layers=2, hidden_size=12,
                 dropout=1.0, padding_mode=PaddingMode.NONE, num_classes=2):
        self.padding_mode = padding_mode
        self.working_dir = working_dir
        self.quantifiers = [quantifier]
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_classes = num_classes

    # for variable length sequences,
    # see http://danijar.com/variable-sequence-lengths-in-tensorflow/
    def _length(self, data):
        # """Gets real length of sequences from a padded tensor.
        #
        # Args:
        #   data: a Tensor, containing sequences
        #
        # Returns:
        #   a Tensor, of shape [data.shape[0]], containing the length
        #   of each sequence
        # """
        # the if checks might be pretty slow
        if self.padding_mode != PaddingMode.MAXLEN:
            data = tf.slice(data, [0, 0, 0], [-1, -1, quantifiers.Quantifier.num_chars])
        used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        if self.padding_mode == PaddingMode.SINGLE:
            length = tf.add(1, length)
        return length

    def _lstm_model_fn(self, features, labels, mode, params):
        # how big each input will be
        num_quants = len(self.quantifiers)
        item_size = quantifiers.Quantifier.num_chars + num_quants

        # -- input_models: [batch_size, max_len, item_size]
        input_models = features[INPUT_FEATURE]
        # -- input_labels: [batch_size, num_classes]
        input_labels = labels
        # -- lengths: [batch_size], how long each input really is
        lengths = self._length(input_models)
        # input_models = tf.Print(input_models, [lengths], "###LEN", summarize=400)

        cells = []
        for _ in range(self.num_layers):
            cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)

            if mode == tf.estimator.ModeKeys.TRAIN:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, state_keep_prob=self.dropout)

            cells.append(cell)

        multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        # run on input
        # -- output: [batch_size, max_len, out_size]
        output, h_out = tf.nn.dynamic_rnn(multi_cell, input_models,dtype=tf.float64, sequence_length=lengths)

        # extract output at end of reading sequence
        # -- flat_output: [batch_size * max_len, out_size]
        flat_output = tf.reshape(output, [-1, self.hidden_size])
        # -- indices: [batch_size]
        output_length = tf.shape(output)[0]
        indices = (tf.range(0, output_length) * params['max_len'] + (lengths - 1))
        # -- final_output: [batch_size, out_size]
        final_output = tf.gather(flat_output, indices)
        tf.summary.histogram('final_output', final_output)


        # -- logits: [batch_size, num_classes]
        logits = tf.contrib.layers.fully_connected(
            inputs=final_output,
            num_outputs=self.num_classes,
            activation_fn=None)
        # -- probs: [batch_size, num_classes]
        probs = tf.nn.softmax(logits)
        # dictionary of outputs
        outputs = {'probs': probs}

        # exit before labels are used when in predict mode
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=outputs)

        # -- loss: [batch_size]
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=input_labels, logits=logits)
        # -- total_loss: scalar
        total_loss = tf.reduce_mean(loss)

        # training op
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        train_op = optimizer.minimize(total_loss, global_step=tf.train.get_global_step())

        # total accuracy
        # -- prediction: [batch_size]
        prediction = tf.argmax(probs, 1)
        # -- target: [batch_size]
        target = tf.argmax(input_labels, 1)

        # list of metrics for evaluation
        eval_metrics = {'total_accuracy': tf.metrics.accuracy(target, prediction)}

        # metrics by quantifier
        # -- flat_inputs: [batch_size * max_len, item_size]
        flat_input = tf.reshape(input_models, [-1, item_size])
        # -- final_inputs: [batch_size, item_size]
        final_inputs = tf.gather(flat_input, indices)
        # extract the portion of the input corresponding to the quantifier
        # -- quants_by_seq: [batch_size, num_quants]
        quants_by_seq = tf.slice(final_inputs,
                                 [0, quantifiers.Quantifier.num_chars],
                                 [-1, -1])
        # index, in the quantifier list, of the quantifier for each data point
        # -- quant_indices: [batch_size]
        quant_indices = tf.to_int32(tf.argmax(quants_by_seq, 1))
        # -- prediction_by_quant: a list num_quants long
        # -- prediction_by_quant[i]: Tensor of predictions for quantifier i
        prediction_by_quant = tf.dynamic_partition(
            prediction, quant_indices, num_quants)
        # -- target_by_quant: a list num_quants long
        # -- target_by_quant[i]: Tensor containing true for quantifier i
        target_by_quant = tf.dynamic_partition(
            target, quant_indices, num_quants)

        for idx in xrange(num_quants):
            key = '{}_accuracy'.format(self.quantifiers[idx]._name)
            eval_metrics[key] = tf.metrics.accuracy(
                target_by_quant[idx], prediction_by_quant[idx])

        return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op, eval_metric_ops=eval_metrics)

    def train(self, max_len=20, batch_size=8, epochs=4, eval_steps=50, stop_loss=0.05, num_data=200000):

        tf.reset_default_graph()

        csv_file = '{}.csv'.format(self.working_dir)

        # BUILD MODEL
        run_config = tf.estimator.RunConfig(save_checkpoints_steps=eval_steps,
                                            save_checkpoints_secs=None,
                                            save_summary_steps=eval_steps)

        model = tf.estimator.Estimator(model_fn=self._lstm_model_fn, model_dir=self.working_dir,
                                       config=run_config, params={'max_len': max_len})

        # GENERATE DATA
        generator = data_gen.DataGenerator(max_len, self.quantifiers, mode='g', num_data_points=num_data)

        training_data = generator.get_training_data()
        test_data = generator.get_test_data()

        def get_np_data(data):
            x_data = np.array([datum[0] for datum in data])
            y_data = np.array([datum[1] for datum in data])
            return x_data, y_data

        # input fn for training
        train_x, train_y = get_np_data(training_data)
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={INPUT_FEATURE: train_x},
            y=train_y,
            batch_size=batch_size,
            num_epochs=epochs,
            shuffle=True)

        # input fn for evaluation
        test_x, test_y = get_np_data(test_data)
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={INPUT_FEATURE: test_x},
            y=test_y,
            batch_size=len(test_x),
            shuffle=False)

        print '\n------ Runing training of {} -----'.format(self.working_dir)

        # train and evaluate model together, using the Hook
        model.train(input_fn=train_input_fn,
                    hooks=[EvalEarlyStopHook(model, eval_input_fn, csv_file, eval_steps, stop_loss)])

    def eval(self, max_len=20, num_data=200000):
        tf.reset_default_graph()

        model = tf.estimator.Estimator(
            model_fn=self._lstm_model_fn, params={'max_len': max_len}, model_dir=self.working_dir)

        # GENERATE DATA
        generator = data_gen.DataGenerator(
            max_len, self.quantifiers, mode='g', num_data_points=num_data,
            training_split=1, append_padding=self.padding_mode == PaddingMode.SINGLE)

        in_data = generator.get_training_data()

        def get_np_data(data):
            x_data = np.array([datum[0] for datum in data])
            y_data = np.array([datum[1] for datum in data])
            return x_data, y_data

        # input fn for evaluation
        test_x, test_y = get_np_data(in_data)
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={INPUT_FEATURE: test_x},
            y=test_y,
            batch_size=len(test_x),
            shuffle=False)

        ev_results = model.evaluate(input_fn=eval_input_fn)
        results = dict(ev_results.items())
        return results['loss']
        # suc = all = 0
        # for (i, out) in enumerate(ev_results):
        #     if np.sum(test_x[i])-max_len != max_len-1:
        #         continue
        #     # print to_pretty_str(test_x[i]), test_y[i], out['probs']
        #     all +=1
        #     if np.matmul(out['probs'], test_y[i]) >= 0.5:
        #         suc += 1
        #
        # return (suc*1.)/all
