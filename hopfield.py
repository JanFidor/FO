from typing import NamedTuple
import numpy as np
import numpy.typing as npt


class PredictResult(NamedTuple):
    states: npt.NDArray[np.int8]
    energies: npt.NDArray[np.float32]


class HopfieldNet:
    """
    This is a fully connected HopfieldNet with the following properties:
    - inputs are assuemd to be 2-dimensional.
    - the number of units is the same as the dimension of an input.
    - the state is an 1-dimensional vector with size of the number of units.
    - a unit of the state is either -1 or 1.
    - the weight matrix is constructed by Hebbian rule.
    - the weight matrix is symmetric with zero-diagonal elements.
    - the bias term is the same for all units.
    - the activation function is sign function.
    """

    def __init__(self, inputs: npt.NDArray[np.int8]) -> None:
        self.dim = len(inputs[0])
        self.patterns = len(inputs)
        self.W = np.zeros((self.dim, self.dim))
        self.W_history = []
        mean = np.sum([np.sum(i) for i in inputs]) / (self.patterns * self.dim)

        for i in range(self.patterns):
            t = inputs[i] - mean  # standardization
            self.W += np.outer(t, t)

            current_W = self.W.copy()
            np.fill_diagonal(current_W, 0)
            current_W /= (i+1)
            self.W_history.append(current_W)

        np.fill_diagonal(self.W, 0)
        self.W /= self.patterns
    
    def energy(self, x: npt.NDArray[np.int8], bias: float) -> float:
        return -0.5 * np.dot(x.T, np.dot(self.W, x)) + np.sum(bias * x)

    def sync_predict(self, x: npt.NDArray[np.int8], bias: float) -> PredictResult:
        es = [self.energy(x, bias)]
        xs = [x]
        for i in range(100):
            x_prev = xs[-1]
            e_prev = es[-1]
            x_new = np.sign(np.dot(self.W, x_prev) - bias)
            e_new = self.energy(x_new, bias)
            # if abs(e_new - e_prev) < 1e-7:
            #     return PredictResult(states=xs, energies=es)
            xs.append(x_new)
            es.append(e_new)
        return PredictResult(states=xs, energies=es)

    def async_predict(self, x: npt.NDArray[np.int8], bias: float) -> PredictResult:
        es = [self.energy(x, bias)]
        xs = [x]
        for i in range(len(x)):
            state = xs[-1].copy()
            state_i_new = np.sign(np.dot(self.W[i,:], state) - bias)
            state[i] = state_i_new
            xs.append(state)
            es.append(self.energy(state, bias))
        return PredictResult(states=xs, energies=es)
    
    def predict(self, x: npt.NDArray[np.int8], bias: float, sync: bool) -> PredictResult:
        return self.sync_predict(x, bias) if sync else self.async_predict(x, bias)
