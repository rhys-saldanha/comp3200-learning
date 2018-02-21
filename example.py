import tensorflow as tf

W = tf.Variable([.6], dtype=tf.float64)
b = tf.Variable([.8], dtype=tf.float64)

x = tf.placeholder(dtype=tf.float64)
y = tf.placeholder(dtype=tf.float64)

x_train = [1, 2, 3, 4]
y_train = [0, 1, 2, 3]

linear_model = tf.multiply(W, x) + b

squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

optimiser = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimiser.minimize(loss=loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    past = (0, 0)
    curr = sess.run([W, b])

    error = 1
    target = 1e-20
    n = 0

    while past != curr:  # Loop till variables are unchanged
        past = curr
        sess.run(train, feed_dict={x: x_train, y: y_train})
        curr = sess.run([W, b], {x: x_train, y: y_train})

    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    print("W : ", curr_W, ", b : ", curr_b, ", loss : ", curr_loss)
