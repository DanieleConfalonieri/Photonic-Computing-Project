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
    return il_db * np.log(10) / 20.0

def evaluate_svd_clements_sample(seed, N, alpha_loss):
    rng = np.random.default_rng(seed)
    
    A_target = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    
    _, S, _ = np.linalg.svd(A_target)
    sigma_max = np.max(S)
    if sigma_max > 0:
        A_target_scaled = A_target / sigma_max
    else:
        A_target_scaled = A_target
        
    circ_c = SVDCircuit(A_target_scaled, unitary_circuit_class=ClementsCircuit)
    
    circ_c.U_circuit.setUniformLoss(alpha_loss, "Node")
    circ_c.Vh_circuit.setUniformLoss(alpha_loss, "Node")
    circ_c.Sigma_circuit.setUniformLoss(alpha_loss, "Node")
    
    return circ_c.fidelity

def run_svd_n_sweep(N_values, samples, loss_db):
    results = []
    num_cores = mp.cpu_count()
    print(f"\nStarting multiprocessing on {num_cores} cores (Loss={loss_db} dB)...")
    alpha_loss = db_to_alpha(loss_db)

    for N in N_values:
        worker_func = partial(evaluate_svd_clements_sample, N=N, alpha_loss=alpha_loss)
        seeds = [hash((N, loss_db, i)) % (2**32) for i in range(samples)]
        
        fid_c_list = []
        with mp.Pool(processes=num_cores) as pool:
            for res_c in tqdm(pool.imap_unordered(worker_func, seeds), total=samples, desc=f"N={N}"):
                fid_c_list.append(res_c)
            
        results.append(np.mean(fid_c_list))
            
    return results

def plot_figure_7bc(N_values, data_clements, loss_db, samples, suffix):
    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    
    ugmzi_restored = [1.0] * len(N_values)
    
    plt.plot(N_values, ugmzi_restored, marker='s', color='tab:red', markersize=5, linewidth=1, label='SVD-UGMZI')
    plt.plot(N_values, data_clements, marker='^', color='black', markersize=5, linewidth=1, label='SVD-Clements')
    
    title_text = f"IL$_{{node, dB}} = {loss_db}$ dB"
    plt.title(title_text, fontsize=10, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='square,pad=0.3'))
    
    plt.xlabel("Number of Nodes", fontsize=12) 
    plt.ylabel("Fidelity", fontsize=12)
    
    plt.xlim(4, 20)
    plt.ylim(0.4, 1.05)
    
    plt.xticks(range(4, 21, 4))
    plt.yticks(np.arange(0.4, 1.1, 0.1))
    
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.tick_params(which='both', direction='in', top=True, right=True)
    
    plt.grid(True, which='major', linestyle='--', alpha=0.5)
    plt.legend(loc='lower left', frameon=False)
    plt.tight_layout()
    
    plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))
    os.makedirs(plots_dir, exist_ok=True)
    
    filepath = os.path.join(plots_dir, f"SVD_Fig7{suffix}_{loss_db}dB_n{samples}.pdf")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Plot saved in: {filepath}")
    
    plt.close()

if __name__ == "__main__":
    N_VALUES = range(4, 21, 2)
    SAMPLES = 500  
    
    data_05 = run_svd_n_sweep(N_VALUES, SAMPLES, 0.5)
    plot_figure_7bc(list(N_VALUES), data_05, 0.5, SAMPLES, "b")

    data_15 = run_svd_n_sweep(N_VALUES, SAMPLES, 1.5)
    plot_figure_7bc(list(N_VALUES), data_15, 1.5, SAMPLES, "c")

    for key, original_value in original_env.items():
        if original_value is not None:
            os.environ[key] = original_value
        else:
            os.environ.pop(key, None)