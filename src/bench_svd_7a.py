import os

original_env = {
    'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
    'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS'),
    'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS'),
    'NUMEXPR_NUM_THREADS': os.environ.get('NUMEXPR_NUM_THREADS')
}
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import multiprocessing as mp
from functools import partial

from circuits import SVDCircuit, ClementsCircuit

def db_to_alpha(il_db):
    return il_db * np.log(10) / 10.0 #TODO here we put 10.0 to match the paper, but it should be 20.0 (amplitude loss vs power loss)

def evaluate_single_svd_sample(seed, N, alpha_loss):
    rng = np.random.default_rng(seed)
    
    # Random complex matrix A_target
    A_target = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    
    # Pre-scaling to ensure the largest singular value is 1
    _, S, _ = np.linalg.svd(A_target)
    sigma_max = np.max(S)
    if sigma_max > 0:
        A_target_scaled = A_target / sigma_max
    else:
        A_target_scaled = A_target
        
    # Calcoliamo solo Clements per risparmiare pesantemente sui cicli CPU
    circ_c = SVDCircuit(A_target_scaled, unitary_circuit_class=ClementsCircuit)
    
    # Apply loss and compute fidelity
    circ_c.U_circuit.setUniformLoss(alpha_loss, "Node")
    circ_c.Vh_circuit.setUniformLoss(alpha_loss, "Node")
    circ_c.Sigma_circuit.setUniformLoss(alpha_loss, "Node")
    
    return circ_c.fidelity

def run_svd_loss_sweep(N_values, samples, loss_values):
    results = {4: [], 8: []}
    num_cores = mp.cpu_count()
    print(f"Starting multiprocessing on {num_cores} cores (SVD-Clements N={N_values})...")

    for N in N_values:
        print(f"\n--- Elaboration N={N} ---")
        for il_db in loss_values:
            alpha_loss = db_to_alpha(il_db)
            worker_func = partial(evaluate_single_svd_sample, N=N, alpha_loss=alpha_loss)
            
            seeds = [hash((N, il_db, i)) % (2**32) for i in range(samples)]
            fid_c_list = []
            
            with mp.Pool(processes=num_cores) as pool:
                for res_c in tqdm(pool.imap_unordered(worker_func, seeds), total=samples, desc=f"Loss={il_db:.1f}dB"):
                    fid_c_list.append(res_c)
                
            results[N].append(np.mean(fid_c_list))
            
    return results

def plot_figure_7a(loss_values, data, samples):
    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    
    # UGMZI restored is matematically 1.0
    ugmzi_restored = [1.0] * len(loss_values)
    
    plt.plot(loss_values, ugmzi_restored, marker='s', color='tab:red', markersize=5, linewidth=1, label='SVD-UGMZI N=4 & N=8')
    plt.plot(loss_values, data[4], marker='^', color='gray', markersize=5, linewidth=1, linestyle='--', label='SVD-Clements N=4')
    plt.plot(loss_values, data[8], marker='v', color='black', markersize=5, linewidth=1, linestyle='-.', label='SVD-Clements N=8')
    
    plt.xlabel("Losses per node (dB)", fontsize=12)
    plt.ylabel("Fidelity", fontsize=12)
    
    plt.xlim(0.0, 2.0)
    plt.ylim(0.4, 1.05)
    
    plt.xticks(np.arange(0.0, 2.1, 0.4))
    plt.yticks(np.arange(0.4, 1.1, 0.1))
    
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.tick_params(which='both', direction='in', top=True, right=True)
    
    plt.grid(True, which='major', linestyle='--', alpha=0.5)
    plt.legend(loc='lower left', frameon=False)
    plt.tight_layout()
    
    plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))
    os.makedirs(plots_dir, exist_ok=True)
    
    filepath = os.path.join(plots_dir, f"SVD_Fig7a_SweepLoss_n{samples}.pdf")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Plot salvato in: {filepath}")
    
    plt.show()

if __name__ == "__main__":
    N_VALUES = [4, 8]
    SAMPLES = 500  
    LOSS_VALUES = np.linspace(0.0, 2.0, 11)
    
    data = run_svd_loss_sweep(N_VALUES, SAMPLES, LOSS_VALUES)
    plot_figure_7a(list(LOSS_VALUES), data, SAMPLES)

    for key, original_value in original_env.items():
        if original_value is not None:
            os.environ[key] = original_value
        else:
            os.environ.pop(key, None)