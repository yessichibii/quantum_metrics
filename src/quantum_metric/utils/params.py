from autograd import numpy as anp
from typing import Optional, Union

def init_params(n: int, init: Optional[Union[anp.ndarray, list]] = None, rng=0.5) -> anp.ndarray:
    if init is None or init == []:
        return anp.array(anp.random.uniform(-rng, rng, n))
    return anp.array(init)

def cross_entropy(labels, predictions):
    epsilon = 1e-15
    predictions = anp.clip(predictions, epsilon, 1 - epsilon)
    loss = -anp.sum(labels * anp.log(predictions)) / len(labels)
    return loss