import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import interferometer as inf
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
    def loss(self):
        return self._node._lossPerNode

    @property
    def numPhaseShifters(self):
        return self._numPhaseShifters

    @property
    def numCouplers(self):
        return self._numCouplers

    @loss.setter
    def loss(self, values: tuple):
        self._node.loss = values

    @abstractmethod
    def setPhaseGaussianNoise(self, averageNoise, stdNoise):
        pass

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
        inf.Interferomete
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