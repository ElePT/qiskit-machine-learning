# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Fidelity Quantum Kernel"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.algorithms.state_fidelities import BaseStateFidelity, ComputeUncompute
from qiskit.primitives import Sampler, BaseSampler
from qiskit.quantum_info import Statevector
from .base_kernel import BaseKernel

KernelIndices = List[Tuple[int, int]]


class StatevectorKernel(BaseKernel):
    r"""
    An implementation of the quantum kernel interface based on the
    :class:`~qiskit.algorithms.state_fidelities.BaseStateFidelity` algorithm.

    The general task of machine learning is to find and study patterns in data. For many
    algorithms, the datapoints are better understood in a higher dimensional feature space,
    through the use of a kernel function:

    .. math::

        K(x, y) = \langle f(x), f(y)\rangle.

    Here K is the kernel function, x, y are n dimensional inputs. f is a map from n-dimension
    to m-dimension space. :math:`\langle x, y \rangle` denotes the dot product.
    Usually m is much larger than n.

    The quantum kernel algorithm calculates a kernel matrix, given datapoints x and y and feature
    map f, all of n dimension. This kernel matrix can then be used in classical machine learning
    algorithms such as support vector classification, spectral clustering or ridge regression.

    Here, the kernel function is defined as the overlap of two quantum states defined by a
    parametrized quantum circuit (called feature map):

    .. math::

        K(x,y) = |\langle \phi(x) | \phi(y) \rangle|^2
    """

    def __init__(
        self,
        *,
        feature_map: QuantumCircuit | None = None,
    ) -> None:
        """
        Args:
            feature_map: Parameterized circuit to be used as the feature map. If ``None`` is given,
                :class:`~qiskit.circuit.library.ZZFeatureMap` is used with two qubits. If there's
                a mismatch in the number of qubits of the feature map and the number of features
                in the dataset, then the kernel will try to adjust the feature map to reflect the
                number of features.
        """
        super().__init__(feature_map=feature_map)
        self._statevector_cache = {}

    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray | None = None) -> np.ndarray:

        x_vec, y_vec = self._validate_input(x_vec, y_vec)

        if y_vec is None:
            y_vec = x_vec

        kernel_shape = (x_vec.shape[0], y_vec.shape[0])

        x_svs = [self._get_sv(x) for x in x_vec]
        y_svs = [self._get_sv(y) for y in y_vec]

        kernel_matrix = np.zeros(kernel_shape)
        for i, x in enumerate(x_svs):
            for j, y in enumerate(y_svs):
                kernel_matrix[i, j] = np.abs(np.conj(x) @ y)**2

        return kernel_matrix

    def _get_sv(self, param_values):
        param_values = tuple(param_values)
        sv = self._statevector_cache.get(param_values, None)

        if sv is None:
            qc = self._feature_map.bind_parameters(param_values)
            sv = Statevector(qc).data
            self._statevector_cache[param_values] = sv

        return sv


