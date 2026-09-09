import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from interferometer.interferometer.main import Interferometer, Beamsplitter, triangle_decomposition, square_decomposition, random_unitary

from abc import ABC, abstractmethod

class Node(ABC):

    @property
    @abstractmethod
    def loss(self):
        pass

    # This method set the loss to the node or to the coupler. We check the second element of the tuple to do so.
    @loss.setter
    @abstractmethod
    def loss(self, lossTupla: tuple):
        pass

    @property
    @abstractmethod
    def transmission_matrix(self):
        pass

    @abstractmethod
    def setNoise(self, val1, val2):
        pass

class MZI(Node):
    # MZI definition expoliting the Beamsplitter class from the interferometer package.
    def __init__(self, mode1, mode2, theta, phi, lossTupla: tuple = (0.0, "Node"), bs_instance = None):
        if bs_instance is not None:
            self._BS = bs_instance
        else:
            self._BS = Beamsplitter(mode1, mode2, theta, phi)
        self.loss = lossTupla

    @property
    def loss(self):
        return self._BS.alpha

    @loss.setter
    def loss(self, lossTupla: tuple):
        if(lossTupla[1] == "Node"):
            self._BS.alpha = lossTupla[0]
        elif(lossTupla[1] == "Coupler"):
            self._BS.alpha = 2*lossTupla[0]

    @property
    def transmission_matrix(self):
        phi = self._BS.phi
        theta = self._BS.theta
        return np.exp(-self._BS.alpha) * np.array([[np.exp(1j * phi) * np.cos(theta), -np.sin(theta)], 
                                          [np.exp(1j * phi) * np.sin(theta),  np.cos(theta)]])
    
    def setNoise(self, val1, val2):
        self._BS.theta = self._BS.theta + val1
        self._BS.phi = self._BS.phi + val2


class SGMZI(Node):
    # S_N definition according to the paper, section 3
    def __init__(self, startMode, endMode, thetas, omegas, lossTupla: tuple = (0.0, "Node"), phi0 = 0.0):
        self.alpha = 0.0
        self.loss = lossTupla
        self._endMode = endMode
        self._startMode = startMode
        self._thetas = thetas
        self._omegas = omegas
        self._phi0 = phi0

    @property
    def loss(self):
        return self.alpha
    
    @property
    def N(self):
        return self._endMode - self._startMode + 1

    @loss.setter
    def loss(self, lossTupla: tuple):
        if(lossTupla[1] == "Node"):
            self.alpha = lossTupla[0]
        elif(lossTupla[1] == "Coupler"):
            self.alpha = 2*lossTupla[0]

    # Helper function to compute phis
    @staticmethod
    def compute_phi(N ,row, col, phi0 = 0.0):
        if (N + row - col) % 2 != 0:
            phi = phi0 + np.pi + (np.pi/(4 *N))*(N + 1 - row - col)*(N - 1 + row + col)
        else: 
            phi = phi0 + (np.pi/(4 *N))*(N + row - col)*(N - row + col)
        return phi

    # T_N definition according to the paper, section 3
    @staticmethod
    def mmi_transmission_matrix(N, phi0 = 0.0):
        T = np.zeros((N, N), dtype=complex)
        for row in range(N):
            for col in range(N):
                phi = SGMZI.compute_phi(N, row+1, col+1, phi0)
                T[row, col] = np.exp(1j * phi)
        return T * (1/np.sqrt(N))
    
    @property
    def transmission_matrix(self):
        T = SGMZI.mmi_transmission_matrix(self.N, self._phi0)
        thetaMatrix = np.diag(np.exp(1j * self._thetas))
        omegaMatrix = np.diag(np.exp(1j * self._omegas))
        return np.exp(-self.alpha) * omegaMatrix @ T @ thetaMatrix @ T
    
    def setNoise(self, val1, val2):
        print("Phase noise is not implemented for SGMZI nodes yet.")