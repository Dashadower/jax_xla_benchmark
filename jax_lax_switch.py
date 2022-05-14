import jax
import jax.numpy as jnp

arr = jnp.zeros([10])

cond_arr = jnp.hstack([jnp.zeros([5]), jnp.ones([5])])


print(jax.lax.select(
    cond_arr.astype(int),
    0, 1
))