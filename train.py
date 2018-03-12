import tensorflow as tf
import numpy as np
import os
import time

from model import add_layer, build_net
from data_reader import load_data, DataReader

flags = tf.flags

# data
flags.DEFINE_string('data_dir',    'data',   'data directory. Should contain train.txt/test.txt with input data')
flags.DEFINE_string('train_dir',   'cv',     'training directory (models and summaries are saved there periodically)')
flags.DEFINE_string('load_model',   None,    '(optional) filename of the model to load.')

# model parameters
flags.DEFINE_string ('input_dim',      '[0, 1]', 'the input dimension')
flags.DEFINE_string ('output_dim',     '[14]', 'the output dimension')
flags.DEFINE_string ('layers',         '[80, 80, 80, 80, 80]', 'layers of the neural net')
flags.DEFINE_string ('act_funcs', 'tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu', 'non-linear funcs')

# optimization1
flags.DEFINE_float  ('learning_rate_decay', 0.5,  'learning rate decay')
flags.DEFINE_float  ('learning_rate',       0.001,  'starting learning rate')
flags.DEFINE_float  ('decay_when',          30.0,  'decay if loss is less than the decay_when * learning_rate')
flags.DEFINE_integer('batch_size',          200,   'number of data to train on in parallel')
flags.DEFINE_integer('max_epochs',          500000,   'number of full passes through the training data')

# bookkeeping
flags.DEFINE_integer('print_every',    1000,    'how often to print current loss')

input_names  = ['P', 'T', 'C', 'N', 'O']
#                0    1    2    3    4
output_names = ['H', 'He', 'C', 'N', 'O', 'H2', 'CO', 'CO2', 'CH4', 'H2O', 'N2', 'HCN', 'NH3']
#                0    1     2    3    4    5     6      7      8      9     10    11     12

FLAGS = flags.FLAGS


def main(_):
    ''' Train model from here'''

    if not os.path.exists(FLAGS.train_dir):
        os.mkdir(FLAGS.train_dir)
        print('Created training directory', FLAGS.train_dir)

    # get data from data_reader
    actual_in_dim = eval(FLAGS.input_dim)
    actual_out_dim = eval(FLAGS.output_dim)
    data_tensors = load_data(FLAGS.data_dir)
    train_reader = DataReader(data_tensors['train'], actual_in_dim, actual_out_dim, FLAGS.batch_size)
    test_reader = DataReader(data_tensors['test'], actual_in_dim, actual_out_dim, FLAGS.batch_size)

    print('initialized all dataset readers')

    # build the model

    xs = tf.placeholder(tf.float32, [None, len(actual_in_dim)])
    ys = tf.placeholder(tf.float32, [None, len(actual_out_dim)])
    prediction = build_net(xs, len(actual_in_dim), len(actual_out_dim),
                           eval(FLAGS.layers), eval(FLAGS.act_funcs))
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    learning_rate = tf.Variable(FLAGS.learning_rate)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # initialization and train
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        epoch_start_time = time.time()

        # training starts here
        for epoch in range(FLAGS.max_epochs):
            for x, y in train_reader.iter():
                sess.run(train_step, feed_dict={xs: x, ys: y})
                cur_loss = float(sess.run(loss, feed_dict=
                    {xs: data_tensors['train'][:, actual_in_dim], ys: data_tensors['train'][:, actual_out_dim]}))
                current_learning_rate = sess.run(learning_rate)
                if abs(cur_loss) < current_learning_rate * FLAGS.decay_when:
                    print("current learningRate: ", current_learning_rate)
                    change_learn = tf.assign(learning_rate, current_learning_rate * FLAGS.learning_rate_decay)
                    sess.run(change_learn)
                    print("new learning rate is: ", sess.run(learning_rate))
            if epoch % FLAGS.print_every == 0:
                time_elapsed = time.time() - epoch_start_time
                epoch_start_time = time.time()
                print('epoch %7d: train_loss = %6.8f, secs = %.4fs' %
                      (epoch, cur_loss, time_elapsed))

        # test starts here
        sess.run(train_step, feed_dict=
            {xs: data_tensors['test'][:, actual_in_dim], ys: data_tensors['test'][:, actual_out_dim]})
        cur_loss = float(sess.run(loss, feed_dict=
            {xs: data_tensors['test'][:, actual_in_dim], ys: data_tensors['test'][:, actual_out_dim]}))
        prediction_value = sess.run(prediction, feed_dict=
            {xs: data_tensors['test'][:, actual_in_dim], ys: data_tensors['test'][:, actual_out_dim]})
        print('testing loss:', cur_loss)
        np.savetxt("cv/pred_" + output_names[actual_out_dim[0] - len(input_names)] +
                   "_np_test_loss" + str(cur_loss) + "_" + str(FLAGS.batch_size) + ".txt", prediction_value)


if __name__ == "__main__":
    tf.app.run()
