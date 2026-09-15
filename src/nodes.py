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
        self._thetas = np.asarray(thetas, dtype=float)
        self._omegas = np.asarray(omegas, dtype=float)
        self._phi0 = phi0
        self.thetaMatrix = np.diag(np.exp(1j * self._thetas))
        self.omegaMatrix = np.diag(np.exp(1j * self._omegas))
        self._thetaMatrix = self.thetaMatrix
        self._omegaMatrix = self.omegaMatrix

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
        return np.exp(-self.alpha) * self.omegaMatrix @ T @ self.thetaMatrix @ T
    
    def setNoise(self, thetas_noise, omegas_noise):
        thetas_noise = np.asarray(thetas_noise)
        omegas_noise = np.asarray(omegas_noise)

        # --- Gestione THE TAS ---
        # Se passano una matrice 2D (di angoli), prendiamo la diagonale
        if thetas_noise.ndim == 2:
            thetas_noise = np.diag(thetas_noise)
        
        # Broadcasting 0D e Padding (N-1)
        if thetas_noise.ndim == 0:
            thetas_noise = np.full(self.N, thetas_noise)
        elif len(thetas_noise) == self.N - 1:
            thetas_noise = np.append(thetas_noise, 0.0)

        # Aggiornamento additivo e wrapping tra -pi e pi (opzionale ma pulito)
        self._thetas = (self._thetas + thetas_noise + np.pi) % (2 * np.pi) - np.pi
        self.thetaMatrix = np.diag(np.exp(1j * self._thetas))


        # --- Gestione OMEGAS ---
        if omegas_noise.ndim == 2:
            omegas_noise = np.diag(omegas_noise)
            
        if omegas_noise.ndim == 0:
            omegas_noise = np.full(self.N, omegas_noise)

        self._omegas = (self._omegas + omegas_noise + np.pi) % (2 * np.pi) - np.pi
        self.omegaMatrix = np.diag(np.exp(1j * self._omegas))

        # Allinea le variabili private se proprio ti servono altrove
        self._thetaMatrix = self.thetaMatrix
        self._omegaMatrix = self.omegaMatrix