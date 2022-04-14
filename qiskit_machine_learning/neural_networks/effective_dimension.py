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
"""An implementation of the effective dimension algorithm."""

import time
from typing import Optional, Union, List, Callable, Tuple

import numpy as np
from scipy.special import logsumexp

from qiskit.utils import algorithm_globals

from .opflow_qnn import OpflowQNN
from .neural_network import NeuralNetwork


class EffectiveDimension:
    """
    This class computes the global effective dimension for Qiskit
    :class:`~qiskit_machine_learning.neural_networks.NeuralNetwork`.
    following the definition used in [1].

        **References**
        [1]: Abbas et al., The power of quantum neural networks.
        `The power of QNNs <https://arxiv.org/pdf/2011.00027.pdf>`__.
    """

    def __init__(
        self,
        qnn: NeuralNetwork,
        params: Union[List[float], np.ndarray, int] = 1,
        samples: Union[List[float], np.ndarray, int] = 1,
        callback: Optional[Callable[[str], None]] = None,
    ) -> None:

        """
        Args:
            qnn: A Qiskit :class:`~qiskit_machine_learning.neural_networks.NeuralNetwork`,
                with a specific dimension ``(num_weights)`` that will determine the shape of the
                Fisher Information Matrix ``(num_samples * num_params, num_weights, num_weights)``
                used to compute the global effective dimension for a set of ``samples``,
                of shape ``(num_samples, qnn_input_size)``,
                and ``params``, of shape ``(num_params, num_weights)``.
            params: An array of neural network parameters (weights), of shape
                ``(num_params, num_weights)``, or an ``int`` to indicate the number of parameter
                sets to sample randomly from a uniform distribution. By default, ``params = 1``.
            samples: An array of samples to the neural network, of shape
                ``(num_samples, qnn_input_size)``, or an ``int`` to indicate the number of input
                sets to sample randomly from a normal distribution. By default, ``samples = 1``.
            callback: A callback function for the Monte Carlo sampling.
        """

        # Store arguments
        self._params = None
        self._samples = None
        self._num_params = 1
        self._num_samples = 1
        self._model = qnn
        self._callback = callback

        # Define samples and parameters
        self.params = params  # type: ignore
        # input setter uses self._model
        self.samples = samples  # type: ignore

    def num_weights(self) -> int:
        """Returns the dimension of the model according to the definition
        from [1]."""
        return self._model.num_weights

    @property
    def params(self) -> np.ndarray:
        """Returns network parameters."""
        return self._params

    @params.setter
    def params(self, params: Union[List[float], np.ndarray, int]) -> None:
        """Sets network parameters."""
        if isinstance(params, int):
            # random sampling from uniform distribution
            self._params = algorithm_globals.random.uniform(
                0, 1, size=(params, self._model.num_weights)
            )
        else:
            self._params = np.asarray(params)

        self._num_params = len(self._params)

    @property
    def samples(self) -> np.ndarray:
        """Returns network samples."""
        return self._samples

    @samples.setter
    def samples(self, samples: Union[List[float], np.ndarray, int]) -> None:
        """Sets network samples."""
        if isinstance(samples, int):
            # random sampling from normal distribution
            self._samples = algorithm_globals.random.normal(
                0, 1, size=(samples, self._model.num_inputs)
            )
        else:
            self._samples = np.asarray(samples)

        self._num_samples = len(self._samples)

    def run_monte_carlo(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method computes the model's Monte Carlo sampling for a set of
        samples and parameters (params).

        Returns:
             grads: QNN gradient vector, result of backward passes, of shape
                ``(num_samples * num_params, output_size, num_weights)``.
             outputs: QNN output vector, result of forward passes, of shape
                ``(num_samples * num_params, output_size)``.
        """
        grads = np.zeros(
            (
                self._num_samples * self._num_params,
                self._model.output_shape[0],
                self._model.num_weights,
            )
        )
        outputs = np.zeros((self._num_samples * self._num_params, self._model.output_shape[0]))

        for (i, param_set) in enumerate(self.params):
            t_before_forward = time.time()
            forward_pass = np.asarray(
                self._model.forward(input_data=self.samples, weights=param_set)
            )
            t_after_forward = time.time()

            if self._callback is not None:
                msg = f"iteration {i}, time forward pass: {t_after_forward - t_before_forward}"
                self._callback(msg)

            backward_pass = np.asarray(
                self._model.backward(input_data=self.samples, weights=param_set)[1]
            )
            t_after_backward = time.time()

            if self._callback is not None:
                msg = f"iteration {i}, time backward pass: {t_after_backward - t_after_forward}"
                self._callback(msg)

            grads[self._num_samples * i : self._num_samples * (i + 1)] = backward_pass
            outputs[self._num_samples * i : self._num_samples * (i + 1)] = forward_pass

        # post-processing in the case of OpflowQNN output, to match the CircuitQNN output format
        if isinstance(self._model, OpflowQNN):
            grads = np.concatenate([grads / 2, -1 * grads / 2], 1)
            outputs = np.concatenate([(outputs + 1) / 2, (1 - outputs) / 2], 1)

        return grads, outputs

    def get_fisher_information(
        self, gradients: np.ndarray, model_outputs: np.ndarray
    ) -> np.ndarray:

        """
        This method computes the average Jacobian for every set of gradients and
        model output as shown in Abbas et al.

        Args:
            gradients: A numpy array, result of the neural network's backward pass, of
                shape ``(num_samples * num_params, output_size, num_weights)``.
            model_outputs: A numpy array, result of the neural networks' forward pass,
                of shape ``(num_samples * num_params, output_size)``.
        Returns:
            fisher: A numpy array of shape
                ``(num_samples * num_params, num_weights, num_weights)``
                with the average Jacobian  for every set of gradients and model output given.
        """

        if model_outputs.shape < gradients.shape:
            # add dimension to model outputs for broadcasting
            model_outputs = np.expand_dims(model_outputs, axis=2)

        # get grad-vectors (gradient_k/model_output_k)
        # multiply by sqrt(model_output) so that the outer product cross term is correct
        # after Einstein summation
        gradvectors = np.sqrt(model_outputs) * gradients / model_outputs

        # compute the sum of matrices obtained from outer product of grad-vectors
        fisher_information = np.einsum("ijk,lji->ikl", gradvectors, gradvectors.T)

        return fisher_information

    def get_normalized_fisher(self, normalized_fisher: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        This method computes the normalized Fisher Information Matrix
        and extracts its trace.
        Args:
            normalized_fisher: The Fisher Information Matrix to be normalized.
        Returns:
             normalized_fisher: The normalized Fisher Information Matrix, a numpy array
                 of size ``(num_samples, num_weights, num_weights)``.
             fisher_trace: The trace of the Fisher Information Matrix
                            (before normalizing).
        """

        # compute the trace with all normalized_fisher
        fisher_trace = np.trace(np.average(normalized_fisher, axis=0))

        # average the normalized_fisher over the num_samples to get the empirical normalized_fisher
        fisher_avg = np.average(
            np.reshape(
                normalized_fisher,
                (
                    self._num_params,
                    self._num_samples,
                    self._model.num_weights,
                    self._model.num_weights,
                ),
            ),
            axis=1,
        )

        # calculate normalized_normalized_fisher for all the empirical normalized_fisher
        normalized_fisher = self._model.num_weights * fisher_avg / fisher_trace
        return normalized_fisher, fisher_trace

    def _get_effective_dimension(
        self,
        normalized_fisher: Union[List[float], np.ndarray],
        num_data: Union[List[int], np.ndarray, int],
    ) -> Union[np.ndarray, int]:

        if not isinstance(num_data, int) and len(num_data) > 1:
            # expand dims for broadcasting
            normalized_fisher = np.expand_dims(normalized_fisher, axis=0)
            n_expanded = np.expand_dims(np.asarray(num_data), axis=(1, 2, 3))
            logsum_axis = 1
        else:
            n_expanded = np.asarray(num_data)
            logsum_axis = None

        # calculate effective dimension for each data sample size "n" out
        # of normalized normalized_fisher
        f_mod = normalized_fisher * n_expanded / (2 * np.pi * np.log(n_expanded))
        one_plus_fmod = np.eye(self._model.num_weights) + f_mod
        # take log. of the determinant because of overflow
        dets = np.linalg.slogdet(one_plus_fmod)[1]
        # divide by 2 because of square root
        dets_div = dets / 2
        effective_dims = (
            2
            * (logsumexp(dets_div, axis=logsum_axis) - np.log(self._num_params))
            / np.log(num_data / (2 * np.pi * np.log(num_data)))
        )

        return np.squeeze(effective_dims)

    def get_effective_dimension(
        self, num_data: Union[List[int], np.ndarray, int]
    ) -> Union[np.ndarray, int]:
        """
        This method compute the effective dimension for a data sample size ``num_data``.

        Args:
            num_data: array of data sizes
        Returns:
             effective_dim: array of effective dimensions for each sample size in ``num_data``.
        """

        # step 1: Monte Carlo sampling
        grads, output = self.run_monte_carlo()

        # step 2: compute as many fisher info. matrices as (input, params) sets
        fisher = self.get_fisher_information(gradients=grads, model_outputs=output)

        # step 3: get normalized fisher info matrices
        normalized_fisher, _ = self.get_normalized_fisher(fisher)

        # step 4: compute eff. dim
        effective_dimensions = self._get_effective_dimension(normalized_fisher, num_data)

        return effective_dimensions


class LocalEffectiveDimension(EffectiveDimension):
    """
    This class computes the local effective dimension for Qiskit
    :class:`~qiskit_machine_learning.neural_networks.NeuralNetwork`
    following the definition used in [1].

        **References**
        [1]: Abbas et al., The power of quantum neural networks.
        `The power of QNNs <https://arxiv.org/pdf/2011.00027.pdf>`__.
    """

    def __init__(
        self,
        qnn: NeuralNetwork,
        params: Union[List[float], np.ndarray, int] = 1,
        samples: Union[List[float], np.ndarray, int] = 1,
        callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Args:
            params: An array of neural network parameters (weights), of shape
                ``(1, qnn.num_weights)``, or ``None`` to sample one parameter set
                randomly from a uniform distribution.
            samples: An array of samples to the neural network, of shape
                ``(num_samples, qnn.num_samples)``, or an ``int`` to indicate the number of input
                sets to sample randomly from a normal distribution. By default, ``samples = 1``.
            callback: A callback function for the Monte Carlo sampling.

        Raises:
            QiskitMachineLearningError: If more than 1 set of parameters is inputted.
        """

        super().__init__(qnn, params, samples, callback)

    # override setter to enforce 1 set of parameters
    @property
    def params(self) -> np.ndarray:
        """Returns network parameters."""
        return self._params

    @params.setter
    def params(self, params: Union[List[float], np.ndarray, float]) -> None:
        """Sets network parameters."""
        if params is not None:
            params = np.asarray(params)
            if params.shape[0] > 1:
                if len(params.shape) > 1:
                    raise ValueError(
                        f"The local effective dimension algorithm uses only 1 set of parameters, "
                        f"got {params.shape[0]}"
                    )
                params = np.expand_dims(params, 0)
            self._params = params

        else:
            # random sampling from uniform distribution
            self._params = algorithm_globals.random.uniform(0, 1, size=(1, self._model.num_weights))

        self._num_params = 1
