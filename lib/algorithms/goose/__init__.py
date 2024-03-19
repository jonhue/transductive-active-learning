from jaxtyping import Array, Float
from lib.algorithms import DiscreteGreedyAlgorithm
from lib.algorithms.baselines import UCB
from lib.algorithms.goose.model import GoOSEModel


class GoOSE(DiscreteGreedyAlgorithm[GoOSEModel]):
    r"""
    **GoOSE** [1]

    [1] Turchetta, Matteo, Felix Berkenkamp, and Andreas Krause. "Safe exploration for interactive machine learning." Advances in Neural Information Processing Systems 32 (2019).
    """

    def F(self) -> Float[Array, "n"]:
        oracle = UCB(model=self.model).F()
        return self.model.objective(oracle)
