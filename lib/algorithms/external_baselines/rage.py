# Adapted from: https://github.com/fiezt/Transductive-Linear-Bandit-Code/blob/master/RAGE.py


import jax.numpy as np
import jax.random as jr
import logging
from lib.algorithms import Algorithm
from lib.function import Function


class RAGE(Algorithm):
    def __init__(
        self, key, n, X, theta_star, factor, delta, Z=None
    ):  # X=X, Z=Z, theta_star=THETA.reshape(-1, 1), factor=10, delta=0.05
        self.n = n
        self.X = X
        if Z is None:
            self.Z = X
        else:
            self.Z = Z
        self.K = len(X)
        self.K_Z = len(self.Z)
        self.d = X.shape[1]
        self.theta_star = theta_star
        self.opt_arm = np.argmax(self.Z @ theta_star)
        self.delta = delta
        self.factor = factor

        self.init(key)

    def init(self, key, var=True):
        self.var = var
        self.key = key

        self.active_arms = list(range(len(self.Z)))
        self.arm_counts = np.zeros(self.K)
        self.N = 0
        self.phase_index = 1
        self.finished = False

        self.remaining_pulls = []

    def step(self, f: Function):
        if len(self.active_arms) == 0:
            if not self.finished:
                self.stop()

        if len(self.remaining_pulls) == 0:
            self.delta_t = self.delta / (self.phase_index**2)

            self.build_Y()
            design, rho = self.optimal_allocation()
            support = np.sum((design > 0).astype(int))
            n_min = 2 * self.factor * support
            eps = 1 / self.factor

            num_samples = np.maximum(
                np.ceil(
                    8
                    * (2 ** (self.phase_index - 1)) ** 2
                    * rho
                    * (1 + eps)
                    * np.log(2 * self.K_Z**2 / self.delta_t)
                ),
                n_min,
            ).astype(int)
            self.allocation = self.rounding(design, num_samples)

            # pulls = np.vstack([np.tile(self.X[i], (num, 1)) for i, num in enumerate(allocation) if num > 0])
            self.remaining_pulls = [
                self.X[i] for i, num in enumerate(self.allocation) for _ in range(num)
            ]
            self.key, key = jr.split(self.key)
            self.remaining_pulls = list(
                jr.permutation(
                    key, np.array(self.remaining_pulls), axis=0, independent=True
                )
            )
            self.completed_pulls = []
            self.completed_rewards = []

        pull = self.remaining_pulls.pop(0)
        self.completed_pulls.append(pull)
        pulls = pull.reshape(1, -1)

        self.key, key = jr.split(self.key)
        reward = pulls @ self.theta_star + jr.normal(key, (1,), dtype=np.float64)
        self.completed_rewards.append(reward)

        pulls = np.vstack(self.completed_pulls)
        rewards = np.vstack(self.completed_rewards)
        self.A_inv = np.linalg.pinv(pulls.T @ pulls)
        self.theta_hat = np.linalg.pinv(pulls.T @ pulls) @ pulls.T @ rewards

        self.N += 1
        if len(self.remaining_pulls) == 0:
            self.drop_arms()
            self.phase_index += 1
            self.arm_counts += self.allocation

    def stop(self):
        del self.Yhat
        del self.idxs
        del self.X
        del self.Z
        self.success = self.opt_arm in self.active_arms
        logging.critical("Succeeded? %s" % str(self.success))
        logging.critical("Sample complexity %s" % str(self.N))

    ### METRICS

    def transductive_regret(self):
        aa = np.array(self.active_arms).reshape(-1)
        best_arm = np.argmax(self.Z[aa] @ self.theta_hat)
        return (self.Z[self.opt_arm] - self.Z[aa][best_arm]) @ self.theta_star

    def transductive_convergence(self):
        return len(self.active_arms)

    def metrics(self):
        return {
            "entropy_in_roi": np.nan,
            "max_stddev_in_roi": np.nan,
            "avg_stddev_in_roi": np.nan,
            "transductive_regret": self.transductive_regret(),
            "stddev_at_transductive_optimum": np.nan,
            "transductive_convergence": self.transductive_convergence(),
        }

    ### UTILS

    def build_Y(self):
        k = len(self.active_arms)
        idxs = np.zeros((k * k, 2))
        Zhat = self.Z[np.array(self.active_arms)]
        Y = np.zeros((k * k, self.d))
        rangeidx = np.array(list(range(k)))

        for i in range(k):
            idxs = idxs.at[k * i : k * (i + 1), 0].set(rangeidx)
            idxs = idxs.at[k * i : k * (i + 1), 1].set(i)
            Y = Y.at[k * i : k * (i + 1), :].set(Zhat - Zhat[i, :])

        self.Yhat = Y
        self.idxs = idxs

    def optimal_allocation(self):
        design = np.ones(self.K)
        design /= design.sum()

        max_iter = 5000

        for count in range(1, max_iter):
            A_inv = np.linalg.pinv(self.X.T @ np.diag(design) @ self.X)
            U, D, V = np.linalg.svd(A_inv)
            Ainvhalf = U @ np.diag(np.sqrt(D)) @ V.T

            newY = (self.Yhat @ Ainvhalf) ** 2
            rho = newY @ np.ones((newY.shape[1], 1))

            idx = np.argmax(rho)
            y = self.Yhat[idx, :, None]
            g = ((self.X @ A_inv @ y) * (self.X @ A_inv @ y)).flatten()
            g_idx = np.argmax(g)

            gamma = 2 / (count + 2)
            design_update = -gamma * design
            design_update = design_update.at[g_idx].set(design_update[g_idx] + gamma)

            relative = np.linalg.norm(design_update) / (np.linalg.norm(design))

            design += design_update

            if relative < 0.01:
                break

        idx_fix = np.where(design < 1e-5)[0]
        design = design.at[idx_fix].set(0)

        return design, np.max(rho)  # type: ignore

    def rounding(self, design, num_samples):
        num_support = (design > 0).sum()
        support_idx = np.where(design > 0)[0]
        support = design[support_idx]
        n_round = np.ceil((num_samples - 0.5 * num_support) * support)

        while n_round.sum() - num_samples != 0:
            if n_round.sum() < num_samples:
                idx = np.argmin(n_round / support)
                n_round = n_round.at[idx].set(n_round[idx] + 1)
            else:
                idx = np.argmax((n_round - 1) / support)
                n_round = n_round.at[idx].set(n_round[idx] - 1)

        allocation = np.zeros(len(design))
        allocation = allocation.at[support_idx].set(n_round)

        return allocation.astype(int)

    def drop_arms(self):
        if not self.var:
            active_arms = self.active_arms.copy()
            removes = set()
            scores = self.Yhat @ self.theta_hat
            # gap = 2**(-(self.phase_index+2))
            gap = 2 ** (-(self.phase_index))

            for t, s in enumerate(scores):
                if gap <= s[0]:
                    arm_idx = int(self.idxs[t][1])
                    removes.add(self.active_arms[arm_idx])

            for r in removes:
                self.active_arms.remove(r)

        else:
            active_arms = self.active_arms.copy()

            for arm_idx in active_arms:
                arm = self.Z[arm_idx, :, None]

                for arm_idx_prime in active_arms:
                    if arm_idx == arm_idx_prime:
                        continue

                    arm_prime = self.Z[arm_idx_prime, :, None]
                    y = arm_prime - arm

                    if (
                        np.sqrt(
                            2
                            * y.T
                            @ self.A_inv
                            @ y
                            * np.log(2 * self.K**2 / self.delta_t)
                        )
                        <= y.T @ self.theta_hat
                    ):
                        self.active_arms.remove(arm_idx)
                        break
