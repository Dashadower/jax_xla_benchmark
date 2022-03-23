import jax
import jax.numpy
import jax.scipy.stats
import random
from numpy.random import rand
import numpy
import timeit

def jax_full_replace(dict, indices):
    dict["arr"] = numpy.zeros(indices.shape)

def jax_index_replace(dict, indices):
    dict["arr"][indices] = numpy.zeros(indices.shape)




test_dict = {"arr": jax.numpy.ones(100)}
test_indices = list(range(100))
random.shuffle(test_indices)
#test_indices = numpy.array(test_indices)
test_indices = numpy.array(range(100))

timer = timeit.Timer(lambda: jax_full_replace(test_dict, test_indices))
number, total_time = timer.autorange()
time_per_iteration_loop_inplace = timer.timeit(number=number) / number

print("inplace", time_per_iteration_loop_inplace)

timer = timeit.Timer(lambda: jax_index_replace(test_dict, test_indices))
number, total_time = timer.autorange()
time_per_iteration_loop = timer.timeit(number=number) / number

print("indexed", time_per_iteration_loop)

print(time_per_iteration_loop_inplace < time_per_iteration_loop, time_per_iteration_loop_inplace/time_per_iteration_loop)



