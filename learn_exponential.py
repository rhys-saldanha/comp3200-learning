import tensorflow as tf

# Just wild type

alpha = tf.Variable([602.83564367], dtype=tf.float64)
a = tf.Variable([181784.02022971], dtype=tf.float64)
b = tf.Variable([61808.97738014], dtype=tf.float64)

c = tf.placeholder(dtype=tf.float64)
X = tf.placeholder(dtype=tf.float64)

T = tf.constant(1800, dtype=tf.float64)

with open('data_wt_row1.csv', 'r') as f:
    c_train = f.readline().strip().split(',')
    X_train = f.readline().strip().split(',')

c_train = list(map(lambda x: float(x), c_train))
X_train = list(map(lambda x: float(x), X_train))

death = tf.add(a, tf.multiply(b, c))
model = tf.multiply(tf.div(alpha, death), tf.add(T, tf.div(tf.exp(-tf.multiply(death, T)) - 1.0, death)))

squared_deltas = tf.square(model - X)
loss = tf.reduce_sum(squared_deltas)

optimiser = tf.train.AdamOptimizer(epsilon=0.1)
train = optimiser.minimize(loss=loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    past = (0, 0, 0)
    # Evaluate values of parameters to learn
    curr = sess.run([alpha, a, b])

    error = 1
    target = 1e-20
    n = 0

    while past != curr and n < 10 ** 6 and error > target:
        # Loop till variables are unchanged or 10**6 iterations have been done or loss is small enough
        past = curr
        sess.run(train, feed_dict={c: c_train, X: X_train})
        curr = sess.run([alpha, a, b], {c: c_train, X: X_train})
        error = sess.run(loss, {c: c_train, X: X_train})
        n += 1
        if n % 10**4 == 0:
            print('alpha:{}, a:{}, b:{}, loss:{}'.format(curr[0], curr[1], curr[2], error))

    curr_alpha, curr_a, curr_b, curr_loss = sess.run([alpha, a, b, loss], {c: c_train, X: X_train})
    print('alpha:{}, a:{}, b:{}, loss:{}'.format(curr_alpha, curr_a, curr_b, curr_loss))
