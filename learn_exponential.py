import argparse
import os
import sys

parser = argparse.ArgumentParser()

parser.add_argument('--alpha', type=float, default=5270.252396499989)
parser.add_argument('--a', type=float, default=13069.504470797903)
parser.add_argument('--b', type=float, default=6467.040372893062)
parser.add_argument('--cuda', type=bool, default=False)
parser.add_argument('--nesterov', type=bool, default=True)

args = parser.parse_args()

if not args.cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    use_device = '/cpu:0'
else:
    use_device = '/job:localhost/replica:0/task:0/device:GPU:0 '

import tensorflow as tf

alpha = tf.Variable(args.alpha, dtype=tf.double, name='alpha')
# alpha = tf.constant(20, dtype=tf.double, name='alpha')
a = tf.Variable(args.a, dtype=tf.double, name='a')
b = tf.Variable(args.b, dtype=tf.double, name='b')

c = tf.placeholder(dtype=tf.double, name='c')
X = tf.placeholder(dtype=tf.double, name='X')

T = tf.constant(1800, dtype=tf.double, name='T')

death = tf.add(a, tf.multiply(b, c), name='k')
model = tf.multiply(tf.div(alpha, death), tf.add(T, tf.div(tf.exp(-tf.multiply(death, T)) - 1.0, death)), name='model')

squared_deltas = tf.square(model - X)

with tf.device(use_device):
    loss = tf.reduce_mean(squared_deltas)
    # optimiser = tf.train.MomentumOptimizer(1, .5, use_nesterov=args.nesterov)
    # optimiser = tf.train.MomentumOptimizer(.01, .9, use_nesterov=args.nesterov)
    optimiser = tf.train.AdadeltaOptimizer()
    train = optimiser.minimize(loss=loss)

init = tf.global_variables_initializer()

# with open('data_wt.csv', 'r') as f:
#     c_train = f.readline().strip().split(',')
#     X_train = f.readline().strip().split(',')

c_train = []
X_train = []
# with open('data_wt.csv', 'r') as f:
# with open('data_wt_avg.csv', 'r') as f:
with open('data_wt_avg_short.csv', 'r') as f:
    c_values = f.readline().strip().split(',')
    for line in f:
        c_train += c_values
        X_train += line.strip().split(',')

c_train = list(map(float, c_train))
X_train = list(map(float, X_train))

# print(c_train, '\n', X_train)
# print(len(c_train), '\n', len(X_train))

tf_config = tf.ConfigProto(allow_soft_placement=False)

if __name__ == '__main__':
    print('start')
    with tf.Session(config=tf_config) as sess:
        writer = tf.summary.FileWriter('/tmp/log/', sess.graph)
        sess.run(init)

        # curr_error = 1
        # past_error = 0
        n = -1

        # while True:
        #     sess.run(train, {c: c_train, X: X_train})

        try:
            # while not math.isclose(past_error, curr_error):
            while True:
                # past_error = curr_error
                sess.run(train, {c: c_train, X: X_train})
                n += 1
                if n % (10 ** 4) == 0:
                    n = 0
                    curr_alpha, curr_a, curr_b, curr_loss = sess.run([alpha, a, b, loss], {c: c_train, X: X_train})
                    print('alpha:{}, a:{}, b:{}, loss:{}'.format(curr_alpha, curr_a, curr_b, curr_loss))
        except KeyboardInterrupt:
            pass
        finally:
            curr_alpha, curr_a, curr_b, curr_loss = sess.run([alpha, a, b, loss], {c: c_train, X: X_train})
            print('\nalpha:{}, a:{}, b:{}, loss:{}'.format(curr_alpha, curr_a, curr_b, curr_loss))
            writer.close()
