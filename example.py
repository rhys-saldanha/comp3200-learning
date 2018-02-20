import tensorflow as tf

W = tf.Variable([.6], dtype=tf.float64)
b = tf.Variable([.8], dtype=tf.float64)

x = tf.placeholder(dtype=tf.float64)
y = tf.placeholder(dtype=tf.float64)

x_train = [1, 2, 3, 4]
y_train = [0, 1, 2, 3]

linear_model = W * x + b

squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

optimiser = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimiser.minimize(loss=loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    error = 1
    target = 1e-20
    n = 0

    while error > target:
        _, error = sess.run([train, loss], feed_dict={x: x_train, y: y_train})
        n += 1
        if n == 1000:
            curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
            print("W : ", curr_W, ", b : ", curr_b, ", loss : ", curr_loss)
            n = 0

    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    print("W : ", curr_W, ", b : ", curr_b, ", loss : ", curr_loss)
