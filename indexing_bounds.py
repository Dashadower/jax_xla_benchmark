import jax
import jax.numpy
from jax.scipy.stats.norm import pdf

ar1 = jax.numpy.array([1, 2, 3, 4, 5])

mean = jax.numpy.array([0, 0, 0, 0])
std = jax.numpy.array([1, 1, 1, 1])

print(pdf(ar1[:-1], mean, std))
print(ar1[:-1])