import numpy as np


def ensure_seed(seed):
    if seed is None:
        return 0
    return int(seed)


def spawn_rngs(base_seed, count):
    seed = ensure_seed(base_seed)
    rngs = []
    for idx in range(count):
        child_seed = seed + idx
        rngs.append((child_seed, np.random.default_rng(child_seed)))
    return rngs
