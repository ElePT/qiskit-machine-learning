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

"""Trainable Quantum Kernel"""

from __future__ import annotations

import numpy as np
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.parameterexpression import ParameterValueType


class TrainableKernelMixin:
    """A mixin that adds ability to train kernel."""

    def __init__(
        self, *args, training_parameters: ParameterVector | list[Parameter] | None = None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        if training_parameters is None:
            training_parameters = []
        self._training_parameters = training_parameters

        self._num_trainable_parameters = len(self._training_parameters)

        self._parameter_dict = {parameter: None for parameter in training_parameters}

    def assign_training_parameters(
        self, parameter_values: dict[Parameter, ParameterValueType] | list[ParameterValueType]
    ) -> None:
        """
        Fix the training parameters to numerical values.
        """
        if not isinstance(parameter_values, dict):
            if len(parameter_values) != self._num_trainable_parameters:
                raise ValueError(
                    f"The number of given parameters is wrong: {len(parameter_values)}, "
                    f"expected {self._num_trainable_parameters}."
                )
            self._parameter_dict.update(
                {
                    parameter: parameter_values[i]
                    for i, parameter in enumerate(self._training_parameters)
                }
            )
        else:
            for key in parameter_values:
                if key not in self._training_parameters:
                    raise ValueError(
                        f"Parameter {key} is not a trainable parameter of the feature map and "
                        f"thus cannot be bound. Make sure {key} is provided in the the trainable "
                        "parameters when initializing the kernel."
                    )
                self._parameter_dict[key] = parameter_values[key]

    @property
    def parameter_values(self):
        """
        Numerical values assigned to the training parameters.
        """
        return np.asarray([self._parameter_dict[param] for param in self._training_parameters])

    @property
    def training_parameters(self) -> ParameterVector | list[Parameter] | None:
        """
        Return the vector of training parameters.
        """
        return self._training_parameters
