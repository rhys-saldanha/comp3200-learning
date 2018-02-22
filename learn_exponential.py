import argparse
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

parser = argparse.ArgumentParser()

parser.add_argument('--alpha', type=float, default=889)
parser.add_argument('--a', type=float, default=4164)
parser.add_argument('--b', type=float, default=667)

args = parser.parse_args()

with tf.device('/cpu:0'):
    # with tf.device('/device:gpu:0'):

    # all data
    # alpha: 72502.10095009,    a: 339540.3056056,      b: 54449.33459445,      loss: 619784.3227130468
    # alpha: 889.2980868261652, a: 4164.7414410585925,  b: 667.8660128354211,   loss: 619784.3227130426
    # alpha: 3579.692612228826, a: 16764.340195841032,  b: 2688.361829989068,   loss: 619784.3227130463

    # alpha: 2955.356560847476, a: 13840.462801104484, b: 2219.4832443860337, loss: 44814.461528813496
    # alpha: 3020.3095022586476, a: 14144.649042933394, b: 2268.263133434123, loss: 44814.46152881349

    alpha = tf.Variable(args.alpha, dtype=tf.float64)
    a = tf.Variable(args.a, dtype=tf.float64)
    b = tf.Variable(args.b, dtype=tf.float64)

    c = tf.placeholder(dtype=tf.float64)
    X = tf.placeholder(dtype=tf.float64)

    T = tf.constant(1800, dtype=tf.float64)

    death = tf.add(a, tf.multiply(b, c))
    model = tf.multiply(tf.div(alpha, death), tf.add(T, tf.div(tf.exp(-tf.multiply(death, T)) - 1.0, death)))

    squared_deltas = tf.square(model - X)
    loss = tf.reduce_sum(squared_deltas)

    optimiser = tf.train.MomentumOptimizer(1, .5, use_nesterov=True)
    train = optimiser.minimize(loss=loss)

    init = tf.global_variables_initializer()

# with open('data_wt.csv', 'r') as f:
#     c_train = f.readline().strip().split(',')
#     X_train = f.readline().strip().split(',')

c_train = []
X_train = []
# with open('data_wt.csv', 'r') as f:
with open('data_wt_avg.csv', 'r') as f:
    c_values = f.readline().strip().split(',')
    for line in f:
        c_train += c_values
        X_train += line.strip().split(',')

c_train = list(map(lambda x: float(x), c_train))
X_train = list(map(lambda x: float(x), X_train))

# print(c_train, '\n', X_train)
# print(len(c_train), '\n', len(X_train))

with tf.Session() as sess:
    sess.run(init)

    curr_error = 1
    past_error = 0
    n = -1

    try:
        # while not math.isclose(past_error, curr_error):
        while True:
            past_error = curr_error
            sess.run(train, {c: c_train, X: X_train})
            curr_error = sess.run(loss, {c: c_train, X: X_train})
            n += 1
            if n % (10 ** 4) == 0:
                n = 0
                curr = sess.run([alpha, a, b, loss], {c: c_train, X: X_train})
                print('alpha:{}, a:{}, b:{}, loss:{}'.format(curr[0], curr[1], curr[2], curr[3]))
    except KeyboardInterrupt:
        pass
    finally:
        curr_alpha, curr_a, curr_b, curr_loss = sess.run([alpha, a, b, loss], {c: c_train, X: X_train})
        print('alpha:{}, a:{}, b:{}, loss:{}'.format(curr_alpha, curr_a, curr_b, curr_loss))
