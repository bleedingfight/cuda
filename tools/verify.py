import matplotlib.pyplot as plt
import tensorflow as tf
import json
import numpy as np


def cu_approx(a):
    a = np.array(a)
    return 1.1283791670955126 * (1 + 0.08943 * a * a) * a


with open("test.json", 'r') as f:
    d = json.loads(f.read())
current_in = d['test']
with open('data.json', 'r') as f:
    data = json.loads(f.read())
erf_output = tf.math.erf(tf.constant(current_in)).numpy()
aerf_output = cu_approx(current_in)
#print(erf_output, data['erf'])
for index in range(len(current_in)):
    print("Python Approx:{:.6f},TensorRT Approx:{:.6f}".format(
        aerf_output[index], data['custom'][index]))
