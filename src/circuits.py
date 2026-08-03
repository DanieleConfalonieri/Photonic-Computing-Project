import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import pyswarms as ps
from scipy.optimize import least_squares

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

        nodes = []

        for n in range(self._N, 0, -1):
            # Extract the first column of U_n
            u_1 = U[:, 0]
            # Compute the Householder reflection and build a unique orthonormal matrix.
            # The firts colomn of H (therefore U_n) is not exactly u1, but is that vector scaled by a complex number alpha (|alpha| = 1).
            # To guarantee the property we are looking for, we apply a rescaling.
            alpha, v, tau = sp.linalg.zlarfg(u_1)
            # Compute Q_n^H @ U = Q_block_to_scale; remember that Q_n = H and H is hermitian
            Q_block = sp.linalg.zlarf(v, tau, U) # <- compute the Householder reflection and apply it to U
            # Scale the [1,1] element of Q_block to guarantee that the first column is exactly e1
            # Rescale the result to guarantee the property we are looking for. We impose the condition to avoid numerical instabilities.
            Q_block[0, 0] = 1.0 + 0.0j
            Q_block[0, 1:] = 0.0 + 0.0j  
            Q_block[1:, 0] = 0.0 + 0.0j
            # Extract SGMZI parameters
            # 1) Compute thetas from Q_tilde_n
            u1_real_positive = np.abs(u_1)
            thetas = self.computeThetas(u1_real_positive)
            # 2) Compute omegas from Q_tilde_n
            q = [] 
            for r in range(n):
                res = 0.0
                for k in range(n): 
                    res += np.exp(1j * thetas[k]) * np.exp(1j * (SGMZI.compute_phi(n, r+1, k+1)+SGMZI.compute_phi(n, k+1, 1)))
                q.append(res*1/n)
            omegas = np.angle(u_1) -  np.angle(q)
            # Build the SGMZI and add to the nodes list
            nodes.insert(0,SGMZI(0, n-1, thetas, omegas)) # <- insert at the beginning of the list to guarantee the order of the nodes
            if n > 1: 
                U = Q_block[1:, 1:] # <- reduce the size of U_n for the next iteration

        return nodes
            


    def computeThetas(self, u1_magnitudes):
        n = len(u1_magnitudes)
        
        # Base case: if n == 1, return an array with a single zero (no phase shifters needed)
        if n == 1:
            return np.array([0.0])
            
        # Precompute the phi matrix
        phi_matrix = np.zeros((n, n))
        for r in range(n):
            for k in range(n):
                phi_matrix[r, k] = SGMZI.compute_phi(n, r+1, k+1) + SGMZI.compute_phi(n, k+1, 1)
                
        exp_phi = np.exp(1j * phi_matrix)
        
        # 1) Swarm Particle Optimization (PSO) to find a good initial guess for the phase shifters
        def pso_cost_function(particles):
            n_particles = particles.shape[0]
            
            # Last phase set to zero (as mentioned in the paper)
            zeros = np.zeros((n_particles, 1))
            theta_full = np.hstack((particles, zeros)) 
            
            # Transpose for matrix multiplication
            exp_theta = np.exp(1j * theta_full).T 
            
            predicted_mags = np.abs(exp_phi @ exp_theta) / n
            
            # MSE for each particle
            target_mags = u1_magnitudes.reshape(n, 1)
            errors = np.sum((predicted_mags - target_mags)**2, axis=0)
            
            return errors

        # Standard PSO parameters
        options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}
        # Search space
        bounds = (np.zeros(n-1), np.ones(n-1) * 2 * np.pi)
        
        optimizer = ps.single.GlobalBestPSO(n_particles=150, dimensions=n-1, options=options, bounds=bounds)
        
        best_cost, best_pos = optimizer.optimize(pso_cost_function, iters=200, verbose=False)
        
        # 2) Levenberg-Marquardt (LM) optimization to refine the solution found by PSO
        def lm_residuals(theta_free):
            theta_full = np.append(theta_free, 0.0)
            exp_theta = np.exp(1j * theta_full)
            predicted_mags = np.abs(exp_phi @ exp_theta) / n
            return predicted_mags - u1_magnitudes

        result = least_squares(lm_residuals, best_pos, method='lm')
        
        if not result.success or result.cost > 1e-10:
            print(f"Warning (n={n}): Possible instability. Final LM cost: {result.cost:.2e}")
            
        return np.append(result.x, 0.0)


class SingularValuesCircuit(Circuit):
    def __init__(self, U_id):
        pass

    @property
    def experimentalMatrix(self):
        pass

