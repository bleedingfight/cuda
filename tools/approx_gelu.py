import tensorflow as tf
import matplotlib.pyplot as plt
from math import pi


def approx_erf(x):
    const_co = 0.044715
    return tf.tanh(2 / tf.sqrt(pi) * (x + const_co * 2 * tf.pow(x, 3)))


def approx_gelu(x):
    return 0.5 * tf.constant(x) * (1 + approx_erf(x / tf.sqrt(2.)))


t = tf.range(5000, dtype=tf.float32)
gelu_value = tf.nn.gelu(t)
agelu_value = approx_gelu(t)
plt.plot(t.numpy(), gelu_value.numpy(), label='gelu')
plt.plot(t.numpy(), agelu_value.numpy(), label='agelu')
plt.legend()
plt.grid()
plt.savefig('gelu.png', dpi=600)
