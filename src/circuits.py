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
    def fidelity(self):
        return np.square(np.abs(np.trace(self.experimentalMatrix.conj().T @ self._U_id)/ (np.sqrt(self._N * np.trace(self.experimentalMatrix.conj().T @ self.experimentalMatrix)))))
    

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


class UGMZICircuit(Circuit):
    def __init__(self, U_id):
        self._U_id = U_id
        self._N = U_id.shape[0]
        self._nodes = self.customUGMZIDecomposition(U_id)
    
    @property
    def experimentalMatrix(self):
        res = np.eye(self._N, dtype=complex)
        #for node in self._nodes:
            # Extend the node's transmission matrix to the full size of the circuit
            # res = res @ extended_matrix

    def customUGMZIDecomposition(self, U_id):

        # The main (recursive) decomposition is:
        #   U = Q_n @ Q_block
        # Where:
        # - U is the recursion engine and starts as U = U_id
        # - Q_n is the SGMZI matrix that will be built starting from u_1, first column of U
        # - Q_block is the block matrix that has e1 as first row and column; the right bottom block will be the new U in the recursive step
        # To build Q_n in a deterministic way and guarantee orthonormality, we use the Householder reflection. 
        # Therefore we are never going to build explicitly Q_n; the function sp.linalg.larf applies it directly. Remember it is hermitian, so Q_n^H = Q_n^{-1} = Q_n.

        U = U_id.copy()

        nodes = {}

        for n in (self._N, 0, -1):
            # Extract the first column of U_n
            u_1 = U[:, 0]
            # Compute the Householder reflection and build a unique orthonormal matrix.
            # The firts colomn of H (therefore U_n) is not exactly u1, but is that vector scaled by a complex number alpha (|alpha| = 1).
            # To guarantee the property we are looking for, we apply a rescaling.
            alpha, v, tau = sp.linalg.zlarf(u_1)
            # Compute Q_n^H @ U = Q_block_to_scale; remember that Q_n = H and H is hermitian
            Q_block = sp.linalg.larf(v, tau, U) # <- compute the Householder reflection and apply it to U
            # Scale the [1,1] element of Q_block to guarantee that the first column is exactly e1
            Q_block[0,0] = 1.0 # <- rescale the result to guarantee the property we are looking for
            # Extract SGMZI parameters
            # 1) Compute Q_n = D_n @ Q_tilde_n. D_n is a diagonal matrix obtained by removing the phase from u1.
            u1_phases = np.exp(-1j * np.angle(u_1))
            # 2) Compute thetas from Q_tilde_n
            u1_real_positive = np.abs(u_1.real)
            thetas = self.computeThetas(u1_real_positive)
            # 3) Compute omegas from Q_tilde_n

            # Build the SGMZI and add to the nodes list

            if n > 1: 
                U = Q_block[1:, 1:] # <- reduce the size of U_n for the next iteration

    def computeThetas(self, u1_real_positive):
        pass


class SingularValuesCircuit(Circuit):
    def __init__(self, U_id):
        pass

    @property
    def experimentalMatrix(self):
        pass

