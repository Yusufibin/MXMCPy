from abc import abstractmethod

import numpy as np
import torch

from .generic_numerical_optimization import perform_slsqp_then_nelder_mead
from .optimizer_base import OptimizerBase, OptimizationResult
from .acv_constraints import ACVConstraints

TORCHDTYPE = torch.double


class ACVOptimizer(OptimizerBase, ACVConstraints):

    def __init__(self, model_costs, covariance=None, *args, **kwargs):
        super().__init__(model_costs, covariance, *args, **kwargs)
        self._covariance_tensor = torch.tensor(self._covariance,
                                               dtype=TORCHDTYPE)
        self._model_costs_tensor = torch.tensor(self._model_costs,
                                                dtype=TORCHDTYPE)

    def optimize(self, target_cost):
        if target_cost < np.sum(self._model_costs):
            return self._get_invalid_result()
        if self._num_models == 1:
            return self._get_monte_carlo_result(target_cost)

        ratios = self._solve_opt_problem(target_cost )

        sample_nums = self._compute_sample_nums_from_ratios(ratios,
                                                            target_cost)
        sample_nums = np.floor(sample_nums)
        ratios = self._compute_ratios_from_sample_nums(sample_nums)
        actual_cost = self._compute_total_cost(sample_nums)

        variance = self._compute_objective_function(ratios, actual_cost,
                                                    gradient=False)
        allocation = self._make_allocation(sample_nums)

        return OptimizationResult(actual_cost, variance, allocation)

    def _compute_total_cost(self, sample_nums):
        cost = np.dot(sample_nums, self._model_costs)
        return cost

    def _solve_opt_problem(self, target_cost):
        initial_guess = self._get_initial_guess()
        bounds = self._get_bounds()
        constraints = self._get_constraints(target_cost)

        def obj_func(rat):
            return self._compute_objective_function(rat, target_cost,
                                                    gradient=False)

        def obj_func_and_grad(rat):
            return self._compute_objective_function(rat, target_cost,
                                                    gradient=True)

        ratios = perform_slsqp_then_nelder_mead(bounds, constraints,
                                                initial_guess, obj_func,
                                                obj_func_and_grad)

        return ratios

    def _get_initial_guess(self):
        return self._model_costs[0] / self._model_costs[1:]

    def _get_bounds(self):
        return [(1 + 1e-12, np.inf)] * (self._num_models - 1)

    def _get_constraints(self, target_cost):
        constraints = self._constr_n_greater_than_1(target_cost)
        nr_constraints = \
            self._constr_ratios_result_in_samples_1_greater_than_n(target_cost)
        constraints.extend(nr_constraints)
        return constraints

    def _compute_objective_function(self, ratios, target_cost, gradient):
        ratios_tensor = torch.tensor(ratios, requires_grad=gradient,
                                     dtype=TORCHDTYPE)
        N = self._calculate_n_autodiff(ratios_tensor, target_cost)
        variance = \
            self._compute_acv_estimator_variance(self._covariance_tensor,
                                                 ratios_tensor, N)
        if not gradient:
            return variance.detach().numpy()

        variance.backward()
        result = (variance.detach().numpy(),
                  ratios_tensor.grad.detach().numpy())
        return result

    @staticmethod
    def _get_eval_ratios(ratios_tensor):
        full_ratios = np.ones(len(ratios_tensor) + 1)
        full_ratios[1:] = ratios_tensor
        return full_ratios

    @staticmethod
    def _get_eval_ratios_autodiff(ratios_tensor):
        full_ratios = torch.ones(len(ratios_tensor) + 1, dtype=TORCHDTYPE)
        full_ratios[1:] = ratios_tensor
        return full_ratios

    def _calculate_n(self, ratios, target_cost):
        eval_ratios = self._get_eval_ratios(ratios)
        N = target_cost / (np.dot(self._model_costs, eval_ratios))
        return N

    def _calculate_n_autodiff(self, ratios_tensor, target_cost):
        eval_ratios = self._get_eval_ratios_autodiff(ratios_tensor)
        N = target_cost / (torch.dot(self._model_costs_tensor, eval_ratios))
        return N

    def _compute_acv_estimator_variance(self, covariance, ratios, N):
        big_C = covariance[1:, 1:]
        c_bar = covariance[0, 1:] / torch.sqrt(covariance[0, 0])

        F, F0 = self._compute_acv_F_and_F0(ratios)
        a = (F0 * c_bar).reshape((-1, 1))

        alpha, _ = torch.solve(a, big_C * F)
        R_squared = torch.dot(a.flatten(), alpha.flatten())
        variance = covariance[0, 0] / N * (1 - R_squared)
        return variance

    @staticmethod
    def _compute_ratios_from_sample_nums(sample_nums):
        ratios = sample_nums[1:] / sample_nums[0]
        return ratios

    def _compute_sample_nums_from_ratios(self, ratios, target_cost):
        N = self._calculate_n(ratios, target_cost)
        sample_nums = N * np.array([1] + list(ratios))
        return sample_nums

    @abstractmethod
    def _compute_acv_F_and_F0(self, ratios):
        raise NotImplementedError

    @abstractmethod
    def _make_allocation(self, sample_nums):
        raise NotImplementedError
