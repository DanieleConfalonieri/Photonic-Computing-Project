import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from interferometer.interferometer.main import Interferometer, Beamsplitter, triangle_decomposition, square_decomposition, random_unitary

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

    @abstractmethod
    def setPhaseGaussianNoise(self, averageNoise, stdNoise):
        pass
    
    # This method set an equal loss to each node of the circuit. The loss can be set to the node or to the coupler.
    def setUniformLoss(self, loss, lossUnity = "Node"):
        for node in self.nodes:
            node.loss = (loss, lossUnity)

# To implement
class ClementsCircuit(Circuit):
    def __init__(self, U_id):
        pass

    @property
    def experimentalMatrix(self):
        pass

    @property
    def fidelity(self):
        pass

    def setPhaseGaussianNoise(self, averageNoise, stdNoise):
        pass

class ReckCircuit(Circuit):
    def __init__(self, U_id):
        pass

    @property
    def experimentalMatrix(self):
        pass

    @property
    def fidelity(self):
        pass

    def setPhaseGaussianNoise(self, averageNoise, stdNoise):
        pass

class UGMZICircuit(Circuit):
    def __init__(self, U_id):
        pass
    @property
    def experimentalMatrix(self):
        pass

    @property
    def fidelity(self):
        pass

    def setPhaseGaussianNoise(self, averageNoise, stdNoise):
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

    def setPhaseGaussianNoise(self, averageNoise, stdNoise):
        pass