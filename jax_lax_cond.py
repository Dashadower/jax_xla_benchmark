import jax
import jax.numpy as jnp

arr = jnp.zeros([10])

cond_arr = jnp.hstack([jnp.zeros([5]), jnp.ones([5])])

print(jax.lax.cond(
    cond_arr > 0,
    lambda x: x + 1,
    lambda x: x + 2,
    cond_arr
))