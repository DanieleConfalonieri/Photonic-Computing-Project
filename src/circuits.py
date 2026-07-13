import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from interferometer.interferometer.main import Interferometer, Beamsplitter, triangle_decomposition, square_decomposition, random_unitary
from nodes import MZI, SGMZI

from abc import ABC, abstractmethod

class Circuit(ABC):

    @property
    @abstractmethod
    def experimentalMatrix(self):
        pass

    @property
    @abstractmethod
    def fidelity(self):
        pass

    @property
    def numPhaseShifters(self):
        return self._numPhaseShifters

    @property
    def numCouplers(self):
        return self._numCouplers
 
    def setPhaseGaussianNoise(self, averageNoise, stdNoise):
        errors = np.random.normal(averageNoise, stdNoise, len(self._nodes) * 2)
        for i, node in self._nodes:
            node.setNoise(errors[2*i], errors[2*i + 1])
    
    # This method set an equal loss to each node of the circuit. The loss can be set to the node or to the coupler.
    def setUniformLoss(self, loss, lossUnity = "Node"):
        for node in self.nodes:
            node.loss = (loss, lossUnity)

# To implement
class ClementsCircuit(Circuit):
    def __init__(self, U_id):
        self._U_id = U_id
        self._N = U_id.shape[0]
        self._I = square_decomposition(U_id)
        self._nodes = {}
        i = 0
        for bs in self._I.BS_list:
            self._nodes[i] = MZI(bs.mode1, bs.mode2, bs.theta, bs.phi)
            i = i + 1

    @property
    def experimentalMatrix(self):
        return self._I.calculate_transformation()

    @property
    def fidelity(self):
        return np.square(np.abs(np.trace(self.experimentalMatrix.conj().T @ self._U_id)/ (np.sqrt(self._N * np.trace(self.experimentalMatrix.conj().T @ self.experimentalMatrix)))))


class ReckCircuit(Circuit):
    def __init__(self, U_id):
        self._U_id = U_id
        self._N = U_id.shape[0]
        self._I = triangle_decomposition(U_id)
        self._nodes = {}
        i = 0
        for bs in self._I.BS_list:
            self._nodes[i] = MZI(bs.mode1, bs.mode2, bs.theta, bs.phi)
            i = i + 1

    @property
    def experimentalMatrix(self):
        return self._I.calculate_transformation()

    @property
    def fidelity(self):
        return np.square(np.abs(np.trace(self.experimentalMatrix.conj().T @ self._U_id)/ (np.sqrt(self._N * np.trace(self.experimentalMatrix.conj().T @ self.experimentalMatrix)))))


class UGMZICircuit(Circuit):
    def __init__(self, U_id):
        pass
    
    @property
    def experimentalMatrix(self):
        pass

    @property
    def fidelity(self):
        pass

    def customUGMZIDecomposition(self,):
        pass

class SingularValuesCircuit(Circuit):
    def __init__(self, U_id):
        pass

    @property
    def experimentalMatrix(self):
        pass

    @property
    def fidelity(self):
        pass
