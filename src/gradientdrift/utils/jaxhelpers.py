import jax

def makeRandomPytree(key, template):
    treeDef = jax.tree_util.tree_structure(template)
    leaves = jax.tree_util.tree_leaves(template)
    keys = jax.random.split(key, len(leaves))
    keysPytree = jax.tree_util.tree_unflatten(treeDef, keys)

    return jax.tree_util.tree_map(lambda k, t: jax.random.normal(k, shape=t.shape), keysPytree, template)
