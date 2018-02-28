import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--alpha', type=float, default=5271.088785691099)
parser.add_argument('--a', type=float, default=13067.982889160061)
parser.add_argument('--b', type=float, default=6469.432355220724)
parser.add_argument('--cuda', type=bool, default=False)
parser.add_argument('--nesterov', type=bool, default=True)

args = parser.parse_args()

if not args.cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

# with tf.device('/device:gpu:0'):
with tf.device('/cpu:0'):
    alpha = tf.Variable(args.alpha, dtype=tf.float64)
    a = tf.Variable(args.a, dtype=tf.float64)
    b = tf.Variable(args.b, dtype=tf.float64)

    c = tf.placeholder(dtype=tf.float64)
    X = tf.placeholder(dtype=tf.float64)

    T = tf.constant(1800, dtype=tf.float64)

    death = tf.add(a, tf.multiply(b, c))
    model = tf.multiply(tf.div(alpha, death), tf.add(T, tf.div(tf.exp(-tf.multiply(death, T)) - 1.0, death)))

    squared_deltas = tf.square(model - X)
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

c_train = list(map(lambda x: float(x), c_train))
X_train = list(map(lambda x: float(x), X_train))

# print(c_train, '\n', X_train)
# print(len(c_train), '\n', len(X_train))

if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(init)

        curr_error = 1
        past_error = 0
        n = -1

        # answer = sess.run(model, {c:0})
        # print('c=0, X={}'.format(answer))

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
