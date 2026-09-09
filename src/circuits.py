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
from src.nodes import MZI, SGMZI

from abc import ABC, abstractmethod

class Circuit(ABC):

    @property
    @abstractmethod
    def experimentalMatrix(self):
        pass

    @property
    def fidelity(self):
        # We use the formula in the paper (13):
        # F = |Tr(U^dag U_exp) / sqrt((Tr(U^dag U) * Tr(U_exp^dag U_exp))) | ^ 2
        U = self._U_id
        U_exp = self.experimentalMatrix
        
        trace_num = np.trace(U.conj().T @ U_exp)
        
        # We use np.real to avoid numerical issues with very small imaginary parts due to floating point errors.
        trace_U_norm = np.real(np.trace(U.conj().T @ U))
        trace_U_exp_norm = np.real(np.trace(U_exp.conj().T @ U_exp))
        
        den = np.sqrt(trace_U_norm * trace_U_exp_norm)
        
        # Preventing division by zero
        if den < np.finfo(float).eps:
            return 0.0
            
        return np.square(np.abs(trace_num) / den)

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
        
        # We save the nodes in "circuit" order, so the first element of the list is S_1, the second is S_2, and so on.
        # We multiply the matrices in reverse order to get the final transformation matrix.
        for node in reversed(self._nodes):
            n = node.N
            extended_matrix = np.eye(self._N, dtype=complex)
            # Insertion in the bottom right corner of the extended matrix, according to the dimensions of the SGMZI node.
            extended_matrix[self._N - n :, self._N - n :] = node.transmission_matrix
            res = res @ extended_matrix
        return res
    

    def customUGMZIDecomposition(self, U_id):

        # The main (recursive) decomposition is:
        #   U = Q_n @ Q_block
        # Where:
        # - U is the recursion engine and starts as U = U_id
        # - Q_n is the SGMZI matrix that will be built starting from u_1, first column of U
        # - Q_block is the block matrix that has e1 as first row and column; the right bottom block will be the new U in the recursive step
        # Q_n has to be built in a deterministic way and guarantee orthonormality: a theoretical solution is using the Householder reflection.
        # Here, we use the fact that given the first column of U (2n - 1 free parameters), the SGMZI transmission matrix is uniquely determined!

        U = U_id.copy()
        nodes = [] 

        for n in range(self._N, 0, -1):
            u_1 = U[:, 0].copy()
            
            # 1) Extracting MMI parameters
            u1_magnitudes = np.abs(u_1)
            thetas = self.computeThetas(u1_magnitudes)
            
            # 2) Computing Omegas to compensate the phase of the first column of U.
            q = [] 
            for r in range(n):
                res = 0.0
                for k in range(n): 
                    res += np.exp(1j * thetas[k]) * np.exp(1j * (SGMZI.compute_phi(n, r+1, k+1) + SGMZI.compute_phi(n, k+1, 1)))
                q.append(res/n)
                
            omegas = np.angle(u_1) - np.angle(q)
            
            # 3) Creation of the SGMZI node
            sgmzi_node = SGMZI(0, n-1, thetas, omegas)
            nodes.insert(0, sgmzi_node) 
            
            # 4) Recursive step: if n > 1, we compute the new U for the next iteration.
            if n > 1: 
                # Sn (Qn in our premises) is the SGMZI transmission matrix that we just built.
                Sn = sgmzi_node.transmission_matrix
                
                # Compute Sn^H @ U = Q_block, where Q_block is the block matrix that has e1 as first row and column; the right bottom block will be the new U in the recursive step.
                Q_block = Sn.conj().T @ U
                
                # We impose the first row and column of Q_block to be e1 to avoid numerical instabilities.
                Q_block[0, 0] = 1.0 + 0.0j
                Q_block[0, 1:] = 0.0 + 0.0j
                Q_block[1:, 0] = 0.0 + 0.0j
                
                # Use the right bottom block in the next step.
                U = Q_block[1:, 1:] 

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
        
        if not result.success or result.cost > 1e-6:
            print(f"Warning (n={n}): Possible instability. Final LM cost: {result.cost:.2e}")
            
        return np.append(result.x, 0.0)        

class SingularValuesCircuit(Circuit):
    def __init__(self, U_id):
        self._U_id = U_id
        self._N = U_id.shape[0]
        self._nodes = []

        # Extract the singular values from the diagonal of U_id
        singular_values = np.diag(U_id)

        for mode in range(self._N):
            # We analize the direct transmission ([0,0] element of the MZI transmission matrix)
            sigma = singular_values[mode]
            theta = np.arccos(sigma)
            phi = 0.0  # No phase shift needed for singular values

            mzi = MZI(mode, mode+self._N, theta, phi)
            self._nodes.append(mzi)

    @property
    def experimentalMatrix(self):
        # The experimental matrix is a diagonal NxN
        exp = np.eye(self._N, dtype=complex)

        for mode, node in enumerate(self._nodes):
            exp[mode, mode] = node.transmission_matrix[0, 0]  # Direct transmission

        return exp

class SVDCircuit(Circuit):
    # unitary_circuit_class is one of the above classes (ClementsCircuit, ReckCircuit, UGMZICircuit) that implement the unitary decomposition.
    def __init__(self, U_id, unitary_circuit_class):
        self._U_id = U_id
        self._N = U_id.shape[0]

        # Perform SVD decomposition
        U, S, Vh = np.linalg.svd(U_id)

        # Global normalization: since our circuit is passive,
        # we normalize the singular values between 0 and 1.
        eps = np.finfo(float).eps # machine epsilon for float type
        sigma_max = np.max(S)

        if sigma_max > eps: 
            S = S / sigma_max

        # Safety clipping to ensure singular values are within [0, 1] even with numerical noise.
        S = np.clip(S, 0.0, 1.0)

        S_diag = np.diag(S)
        self.U_circuit = unitary_circuit_class(U)
        self.Sigma_circuit = SingularValuesCircuit(S_diag)
        self.Vh_circuit = unitary_circuit_class(Vh)

    @property
    def experimentalMatrix(self):
        # The light travels through Vh, then Sigma, then U
        U_exp = self.U_circuit.experimentalMatrix
        Sigma_exp = self.Sigma_circuit.experimentalMatrix
        Vh_exp = self.Vh_circuit.experimentalMatrix

        return U_exp @ Sigma_exp @ Vh_exp

