import tensorflow as tf

with open('optima.txt', 'r') as f:
    optima = []
    for l in f:
        if l[0] != '#':
            optima.append(eval(l))

with open('data_wt_avg.csv', 'r') as f:
    c_train = f.readline().strip().split(',')
    c_train = list(map(lambda x: float(x), c_train))

with tf.device('/cpu:0'):
    alpha = tf.Variable(0, dtype=tf.float64)
    a = tf.Variable(0, dtype=tf.float64)
    b = tf.Variable(0, dtype=tf.float64)

    c = tf.placeholder(dtype=tf.float64)

    T = tf.constant(1800, dtype=tf.float64)

    death = tf.add(a, tf.multiply(b, c))
    model = tf.multiply(tf.div(alpha, death), tf.add(T, tf.div(tf.exp(-tf.multiply(death, T)) - 1.0, death)))

    init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for optimum in optima:
        assignments = [alpha.assign(tf.constant(optimum['alpha'], dtype=tf.float64)),
                       a.assign(tf.constant(optimum['a'], dtype=tf.float64)),
                       b.assign(tf.constant(optimum['b'], dtype=tf.float64))]
        sess.run(assignments)
        answer = []
        for c_val in c_train:
            answer.append(str(sess.run(model, {c: c_val})))
        # print(list(zip(c_train, answer)))
        print(' '.join(answer))