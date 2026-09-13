import os

# Save the current environment variables related to threading to restore them later
original_env = {
    'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
    'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS'),
    'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS'),
    'NUMEXPR_NUM_THREADS': os.environ.get('NUMEXPR_NUM_THREADS')
}

# Force single-threaded execution for numerical libraries to avoid oversubscription in multiprocessing
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
from scipy.stats import unitary_group
from tqdm import tqdm 
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import multiprocessing as mp
from functools import partial

from circuits import UGMZICircuit, ClementsCircuit, ReckCircuit

def db_to_alpha(il_db):
    return il_db * np.log(10) / 20.0

def evaluate_single_sample(seed, N, alpha_loss):
    rng = np.random.default_rng(seed)
    U_target = unitary_group.rvs(N, random_state=rng)
    
    circ_u = UGMZICircuit(U_target)
    circ_u.setUniformLoss(alpha_loss, "Node")
    fid_u = circ_u.fidelity
    
    circ_c = ClementsCircuit(U_target)
    circ_c.setUniformLoss(alpha_loss, "Node")
    fid_c = circ_c.fidelity
    
    circ_r = ReckCircuit(U_target)
    circ_r.setUniformLoss(alpha_loss, "Node")
    fid_r = circ_r.fidelity
    
    return fid_u, fid_c, fid_r

def run_loss_sweep_benchmark(N_fixed, samples, loss_values):
    results = {'UGMZI': [], 'Clements': [], 'Reck': []}
    
    num_cores = mp.cpu_count()
    print(f"Starting multiprocessing with {num_cores} cores...fixed N={N_fixed}\n")

    for il_db in loss_values:
        print(f"\n--- Benchmarking Loss={il_db:.2f} dB (Samples: {samples}) ---")
        alpha_loss = db_to_alpha(il_db)
        
        worker_func = partial(evaluate_single_sample, N=N_fixed, alpha_loss=alpha_loss)
        seeds = [hash((il_db, i)) % (2**32) for i in range(samples)]
        
        fid_u_list, fid_c_list, fid_r_list = [], [], []
        
        with mp.Pool(processes=num_cores) as pool:
            for res_u, res_c, res_r in tqdm(pool.imap_unordered(worker_func, seeds), total=samples):
                fid_u_list.append(res_u)
                fid_c_list.append(res_c)
                fid_r_list.append(res_r)
            
        results['UGMZI'].append(np.mean(fid_u_list))
        results['Clements'].append(np.mean(fid_c_list))
        results['Reck'].append(np.mean(fid_r_list))
        
    return results

def plot_figure_4c(loss_values, data, N_fixed, samples):
    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    
    ugmzi_restored = [1.0] * len(loss_values)
    
    plt.plot(loss_values, ugmzi_restored, marker='s', color='tab:red', markersize=5, linewidth=1, label='UGMZI restored')
    plt.plot(loss_values, data['UGMZI'], marker='o', color='darkred', markersize=5, linewidth=1, label='UGMZI')
    plt.plot(loss_values, data['Clements'], marker='^', color='black', markersize=5, linewidth=1, label='Clements')
    plt.plot(loss_values, data['Reck'], marker='v', color='tab:blue', markersize=5, linewidth=1, label='Reck')
    
    title_text = f"N = {N_fixed}"
    plt.title(title_text, fontsize=10, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='square,pad=0.3'))
    
    plt.xlabel("Loss per node (dB)", fontsize=12)
    plt.ylabel("Fidelity", fontsize=12)
    
    plt.xlim(0.0, 2.0)
    plt.ylim(0.2, 1.05)
    
    plt.xticks(np.arange(0.0, 2.1, 0.4))
    plt.yticks(np.arange(0.2, 1.1, 0.1))
    
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.tick_params(which='both', direction='in', top=True, right=True)
    
    plt.grid(True, which='major', linestyle='--', alpha=0.5)
    plt.legend(loc='lower left', frameon=False)
    plt.tight_layout()
    
    plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))
    os.makedirs(plots_dir, exist_ok=True)
    
    filename = f"IL_sweep_N{N_fixed}_n{samples}.pdf"
    filepath = os.path.join(plots_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Plot salvato con successo in: {filepath}")
    
    plt.show()

if __name__ == "__main__":
    N_FIXED = 20
    SAMPLES = 500  
    
    LOSS_VALUES = np.linspace(0.0, 2.0, 11)
    
    data = run_loss_sweep_benchmark(N_FIXED, SAMPLES, LOSS_VALUES)
    print("\nRisultati medi:", data)
    
    plot_figure_4c(list(LOSS_VALUES), data, N_FIXED, SAMPLES)

    # Restore the original environment variables to their previous state
    for key, original_value in original_env.items():
        if original_value is not None:
            os.environ[key] = original_value
        else:
            # If the original value was None, remove the environment variable if it exists
            os.environ.pop(key, None)