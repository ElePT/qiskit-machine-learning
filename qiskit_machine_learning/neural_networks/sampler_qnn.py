# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A Neural Network implementation based on the Sampler primitive."""

from __future__ import annotations
import logging
from numbers import Integral
from typing import Optional, Union, List, Tuple, Callable, cast, Iterable

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import BaseSampler
from qiskit.algorithms.gradients import BaseSamplerGradient
from qiskit_machine_learning.exceptions import QiskitMachineLearningError, QiskitError
from .neural_network import NeuralNetwork

logger = logging.getLogger(__name__)


class SamplerQNN(NeuralNetwork):
    """A Neural Network implementation based on the Sampler primitive."""

    def __init__(
        self,
        sampler: BaseSampler,
        circuit: QuantumCircuit,
        input_params: List[Parameter] | None = None,
        weight_params: List[Parameter] | None = None,
        interpret: Callable[[int], int | Tuple[int, ...]] | None = None,
        output_shape: int | Tuple[int, ...] | None = None,
        gradient: BaseSamplerGradient | None = None,
        input_gradients: bool = False
    )-> None:
        """

        Args:
            circuit: The parametrized quantum circuit that generates the samples of this network.
            input_params: The parameters of the circuit corresponding to the input.
            weight_params: The parameters of the circuit corresponding to the trainable weights.
            interpret: A callable that maps the measured integer to another unsigned integer or
                tuple of unsigned integers. These are used as new indices for the (potentially
                sparse) output array. If no interpret function is
                passed, then an identity function will be used by this neural network.
            output_shape: The output shape of the custom interpretation
            sampler_factory: Factory for sampler primitive
            gradient_factory: String indicating pre-implemented gradient method or factory for
                gradient class
            input_gradients: to be added
        Raises:
            QiskitMachineLearningError: Invalid parameter values.
        """
        # set sampler --> make property?
        self.sampler = sampler
        # use given gradient or default
        self.gradient = gradient

        self._input_params = list(input_params or [])
        self._weight_params = list(weight_params or [])

        # initialize gradient properties
        self.input_gradients = input_gradients # TODO

        # sparse = False --> Hard coded TODO: look into sparse

        self._circuit = circuit.copy()
        if len(self._circuit.clbits) == 0:
            self._circuit.measure_all()
        # self._circuit_transpiled = False TODO: look into transpilation

        # these original values may be re-used when a quantum instance is set,
        # but initially it was None
        self._original_output_shape = output_shape
        self._output_shape = output_shape

        self.set_interpret(interpret, output_shape)
        # next line is required by pylint only
        # self._interpret = interpret
        self._original_interpret = interpret

        # init super clas
        super().__init__(
            len(self._input_params),
            len(self._weight_params),
            False, # sparse
            self._output_shape,
            self._input_gradients,
        )

    def set_interpret(
        self,
        interpret: Callable[[int], int| Tuple[int, ...]] | None = None,
        output_shape: int | Tuple[int, ...] | None = None
    ) -> None:
        """Change 'interpret' and corresponding 'output_shape'.

        Args:
            interpret: A callable that maps the measured integer to another unsigned integer or
                tuple of unsigned integers. See constructor for more details.
            output_shape: The output shape of the custom interpretation, only used in the case
                where an interpret function is provided. See constructor
                for more details.
        """

        # save original values
        self._original_output_shape = output_shape
        self._original_interpret = interpret

        # derive target values to be used in computations
        self._output_shape = self._compute_output_shape(interpret, output_shape)
        self._interpret = interpret if interpret is not None else lambda x: x
        # self.output_shape = self._compute_output_shape(self._interpret, output_shape)

    def _compute_output_shape(
        self,
        interpret: Callable[[int], int | Tuple[int, ...]] | None = None,
        output_shape: int | Tuple[int, ...] | None = None
    ) -> Tuple[int, ...]:
        """Validate and compute the output shape."""

        # this definition is required by mypy
        output_shape_: Tuple[int, ...] = (-1,)

        if interpret is not None:
            if output_shape is None:
                raise QiskitMachineLearningError(
                    "No output shape given, but required in case of custom interpret!"
                )
            if isinstance(output_shape, Integral):
                output_shape = int(output_shape)
                output_shape_ = (output_shape,)
            else:
                output_shape_ = output_shape
        else:
            if output_shape is not None:
                # Warn user that output_shape parameter will be ignored
                logger.warning(
                    "No interpret function given, output_shape will be automatically "
                    "determined as 2^num_qubits."
                )

            output_shape_ = (2**self._circuit.num_qubits,)

        return output_shape_

    def _preprocess(
        self,
        input_data: List[float] | np.ndarray | float | None,
        weights: List[float] | np.ndarray | float | None,
    ) -> Tuple[List[float], int]:
        """
        Pre-processing during forward pass of the network.
        """
        if len(input_data.shape) == 1:
            input_data = np.expand_dims(input_data, 0)
        num_samples = input_data.shape[0]
        # quick fix for 0 inputs
        if num_samples == 0:
            num_samples = 1

        parameters = []
        for i in range(num_samples):
            param_values = [input_data[i, j] for j, input_param in enumerate(self._input_params)]
            param_values += [weights[j] for j, weight_param in enumerate(self._weight_params)]
            parameters.append(param_values)

        return parameters, num_samples

    def _postprocess(self, num_samples, result):
        """
        Post-processing during forward pass of the network.
        """
        prob = np.zeros((num_samples, *self._output_shape))

        for i in range(num_samples):
            counts = result[i]
            print(counts)
            shots = sum(counts.values())

            # evaluate probabilities
            for b, v in counts.items():
                key = (i, int(self._interpret(b)))  # type: ignore
                prob[key] += v / shots

        return prob

    def _forward(
        self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Forward pass of the network.
        """
        parameter_values, num_samples = self._preprocess(input_data, weights)

        # sampler allows batching
        job = self.sampler.run([self._circuit] * num_samples, parameter_values)
        results = job.result().quasi_dists

        result = self._postprocess(num_samples, results)

        return result

    def _preprocess_gradient(self, input_data, weights):
        """
        Pre-processing during backward pass of the network.
        """
        if len(input_data.shape) == 1:
            input_data = np.expand_dims(input_data, 0)

        num_samples = input_data.shape[0]
        # quick fix for 0 inputs
        if num_samples == 0:
            num_samples = 1

        parameters = []
        for i in range(num_samples):
            param_values = [input_data[i, j] for j, input_param in enumerate(self._input_params)]
            param_values += [weights[j] for j, weight_param in enumerate(self._weight_params)]
            parameters.append(param_values)

        return parameters, num_samples

    def _postprocess_gradient(self, num_samples, results):
        """
        Post-processing during backward pass of the network.
        """
        input_grad = np.zeros((num_samples, *self._output_shape, self._num_inputs)) if self._input_gradients else None
        weights_grad = np.zeros((num_samples, *self._output_shape, self._num_weights))

        if self._input_gradients:
            num_grad_vars = self._num_inputs + self._num_weights
        else:
            num_grad_vars = self._num_weights

        for sample in range(num_samples):

            for i in range(num_grad_vars):
                grad = results.gradients[sample][i]
                for k in grad.keys():
                    val = results.gradients[sample][i][k]
                    # get index for input or weights gradients
                    if self._input_gradients:
                        grad_index = i if i < self._num_inputs else i - self._num_inputs
                    else:
                        grad_index = i
                    # interpret integer and construct key
                    key = self._interpret(k)
                    if isinstance(key, Integral):
                        key = (sample, int(key), grad_index)
                    else:
                        # if key is an array-type, cast to hashable tuple
                        key = tuple(cast(Iterable[int], key))
                        key = (sample, *key, grad_index)
                    # store value for inputs or weights gradients
                    if self._input_gradients:
                        # we compute input gradients first
                        if i < self._num_inputs:
                            input_grad[key] += np.real(val)
                        else:
                            weights_grad[key] += np.real(val)
                    else:
                        weights_grad[key] += np.real(val)

        return input_grad, weights_grad

    def _backward(
        self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],]:

        """Backward pass of the network.
        """
        # prepare parameters in the required format
        parameter_values, num_samples = self._preprocess_gradient(input_data, weights)

        if self._input_gradients:
            job = self.gradient.run([self._circuit] * num_samples, parameter_values)
        else:
            job = self.gradient.run([self._circuit] * num_samples, parameter_values,
                                    parameters=[self._circuit.parameters[self._num_inputs:]])

        results = job.result()

        input_grad, weights_grad = self._postprocess_gradient(num_samples, results)

        return input_grad, weights_grad  # `None` for gradients wrt input data, see TorchConnector
