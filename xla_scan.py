import jax
import jax.numpy as jnp

def lax_select_scalar(pred, on_true, on_false):
    # For jax.lax.select, the dimensions pred, on_true, on_false must all match.
    # This may not be true for values return by directly generated code(ex. select([True, False], 0.0, 1.0).
    # This function is a wrapper to support scalar expressions with array preds by expanding scalars to an array,
    # making them compatible with select()
    if isinstance(pred, jax.numpy.ndarray):
        if not isinstance(on_true, jax.numpy.ndarray):
            # if pred is an array but on_true is a scalar, convert on_true to array by filling
            on_true = jax.numpy.full(pred.shape, on_true)
        elif isinstance(on_true, jax.numpy.ndarray) and on_true.size == 1 and len(pred.shape) > 0:
            # do the same for "arrays" pretending to be an array (array with no dims)
            on_true = jax.numpy.tile(on_true, pred.size)

        # do the exact same thing for on_false
        if not isinstance(on_false, jax.numpy.ndarray):
            on_false = jax.numpy.full(pred.shape, on_false)
        elif isinstance(on_false, jax.numpy.ndarray) and on_false.size == 1 and len(pred.shape) > 0:
            on_false = jax.numpy.tile(on_false, pred.size)

    return jax.lax.select(pred, on_true, on_false)

data = {"school": jnp.array([1.0, 2.0, 3.0, 4.0])}
parameters = {"epsilon": jnp.array([1.0, 2.0, 3.0, 4.0])}
subscripts = {"subscript__3": jnp.array([0, 1, 2, 3])}

def scan_function_1(carry, index):
    (carry1) = carry
    next_value = lax_select_scalar(data['school'][index] == 1, 10.0, carry1 + 1.0)
    return (next_value), next_value


_, parameters['mu'] = jax.lax.scan(scan_function_1, (0.0), jax.numpy.arange(4))

print(parameters["mu"])
