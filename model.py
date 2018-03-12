import tensorflow as tf


# add_layer
def add_layer(inputs, in_size, out_size, activation_function=None, bia=0.5):
    # add one more layer and return the output of this layer
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + bia)
    wx_plus_b = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b)
    return outputs


# build the model
def build_net(inputs, input_dim, out_dim, layer_sizes, act_funcs):
    layers = []
    first_layer = add_layer(inputs, input_dim, layer_sizes[0], activation_function=act_funcs[0])
    layers.append(first_layer)
    for idx in range(1, len(layer_sizes)):
        cur_layer = add_layer(layers[-1], layer_sizes[idx - 1], layer_sizes[idx], activation_function=act_funcs[idx])
        layers.append(cur_layer)
    last_layer = add_layer(layers[-1], layer_sizes[-1], out_dim, activation_function=None)
    return last_layer


