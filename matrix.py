import time
import sys
import tensorflow as tf

if len(sys.argv) != 4:
    raise Exception("%s m n k" % sys.argv[0])

m = int(sys.argv[1])
n = int(sys.argv[2])
k = int(sys.argv[3])

tf.random.set_seed(5)

  # TensorFlow initializes a GPU the first time it's used, exclude from timing.
t = time.time()
tf.matmul(tf.random.normal((32, 32)), tf.random.normal((32, 32)))
print("initialized GPU %ss" % (time.time() - t))
t = time.time()

m1 = tf.random.uniform(shape=(m, n), minval=-1.0, maxval=1.0)
m2 = tf.random.uniform(shape=(n, k), minval=-1.0, maxval=1.0)
m3 = tf.random.uniform(shape=(m, k), minval=-1.0, maxval=1.0)

print("initialized random matrices in %ss" % (time.time() - t))
t = time.time()

mult = tf.matmul(m1, m2)
answer = tf.add(mult, m3)

print("matmuladd (eager) in %ss" % (time.time() - t))
t = time.time()

# tf.Function does not result in improvement.

# @tf.function
# def mult_add(x, w, b):
#   return tf.add(tf.matmul(x, w), b)

# m4 = mult_add(m1, m2, m3)
# print("matmuladd (tf.function) in %ss" % (time.time() - t))
# t = time.time()

# m4 = mult_add(m1, m2, m3)
# print("matmuladd (tf.function) in %ss" % (time.time() - t))
# t = time.time()
