import jax
import jax.numpy
import jax.scipy.stats

from numpy.random import rand

import timeit

def jax_normal_lp(variate, mean, std):
    return jax.scipy.stats.norm.logpdf(variate, mean, std)


def jax_normal_lp_unfolded(arr_a, arr_b, arr_c, mean, std):
    variate = arr_a + arr_b * arr_c
    return jax.scipy.stats.norm.logpdf(variate, mean, std)


def test_unfolded(compiled_func):
    a = jax.numpy.array(rand(1000))
    b = jax.numpy.array(rand(1000))
    c = jax.numpy.array(rand(1000))

    return compiled_func(a, b, c, jax.numpy.zeros(1000), jax.numpy.ones(1000)).block_until_ready()

def test_folded(compiled_func):
    a = jax.numpy.array(rand(1000))
    b = jax.numpy.array(rand(1000))
    c = jax.numpy.array(rand(1000))
    variate = a + b * c
    return compiled_func(variate, jax.numpy.zeros(1000), jax.numpy.ones(1000)).block_until_ready()


compiled_unfolded = jax.jit(jax_normal_lp_unfolded)
compiled_folded = jax.jit(jax_normal_lp)
timer = timeit.Timer(lambda: test_unfolded(compiled_unfolded))
number, total_time = timer.autorange()
time_per_iteration_loop = timer.timeit(number=number) / number

print("unfolded", time_per_iteration_loop)

timer = timeit.Timer(lambda: test_folded(compiled_folded))
number, total_time = timer.autorange()
time_per_iteration_loop = timer.timeit(number=number) / number

print("folded  ", time_per_iteration_loop)



