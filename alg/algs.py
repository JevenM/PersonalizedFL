from alg.fedavg import fedavg
from alg.fedprox import fedprox
from alg.fedbn import fedbn
from alg.base import base
from alg.fedap import fedap
from alg.metafed import metafed
from alg.fedlp_my import fedlp
from alg.fedlp_my_clip import fedlp_clip

ALGORITHMS = [
    'fedavg',
    'fedprox',
    'fedbn',
    'base',
    'fedap',
    'metafed',
    'fedlp'
    'fedlp_clip'
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError(
            "Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
