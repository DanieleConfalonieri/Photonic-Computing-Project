import os
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
    # change the random seed at each iteration to ensure different random unitaries for each sample
    rng = np.random.default_rng(seed)
    U_target = unitary_group.rvs(N, random_state=rng)
    
    # UGMZI
    circ_u = UGMZICircuit(U_target)
    circ_u.setUniformLoss(alpha_loss, "Node")
    fid_u = circ_u.fidelity
    
    # Clements
    circ_c = ClementsCircuit(U_target)
    circ_c.setUniformLoss(alpha_loss, "Node")
    fid_c = circ_c.fidelity
    
    # Reck
    circ_r = ReckCircuit(U_target)
    circ_r.setUniformLoss(alpha_loss, "Node")
    fid_r = circ_r.fidelity
    
    return fid_u, fid_c, fid_r

def run_loss_benchmark(N_values, samples, il_db):
    results = {'UGMZI': [], 'Clements': [], 'Reck': []}
    alpha_loss = db_to_alpha(il_db)
    
    # Look for the number of available CPU cores for multiprocessing
    num_cores = mp.cpu_count()
    print(f"Starting multiprocessing with {num_cores} cores...")

    for N in N_values:
        print(f"\n--- Benchmarking N={N} (Loss: {il_db} dB, Samples: {samples}) ---")
        
        worker_func = partial(evaluate_single_sample, N=N, alpha_loss=alpha_loss)
        
        seeds = [hash((N, i)) % (2**32) for i in range(samples)]
        
        fid_u_list, fid_c_list, fid_r_list = [], [], []
        
        # Map the execution of the worker function across multiple processes
        with mp.Pool(processes=num_cores) as pool:
            for res_u, res_c, res_r in tqdm(pool.imap_unordered(worker_func, seeds), total=samples):
                fid_u_list.append(res_u)
                fid_c_list.append(res_c)
                fid_r_list.append(res_r)
            
        results['UGMZI'].append(np.mean(fid_u_list))
        results['Clements'].append(np.mean(fid_c_list))
        results['Reck'].append(np.mean(fid_r_list))
        
    return results

def plot_figure_4a(N_values, data, il_db, samples):
    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    
    ugmzi_restored = [1.0] * len(N_values)
    
    plt.plot(N_values, ugmzi_restored, marker='s', color='tab:red', markersize=5, linewidth=1, label='UGMZI restored')
    plt.plot(N_values, data['UGMZI'], marker='o', color='darkred', markersize=5, linewidth=1, label='UGMZI')
    plt.plot(N_values, data['Clements'], marker='^', color='black', markersize=5, linewidth=1, label='Clements')
    plt.plot(N_values, data['Reck'], marker='v', color='tab:blue', markersize=5, linewidth=1, label='Reck')
    
    title_text = f"IL$_{{node, dB}} = {il_db}$ dB"
    plt.title(title_text, fontsize=10, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='square,pad=0.3'))
    
    plt.xlabel("Number of Inputs", fontsize=12)
    plt.ylabel("Fidelity", fontsize=12)
    
    plt.xlim(4, 20)
    plt.ylim(0.2, 1.05)
    plt.xticks(range(4, 21, 4))
    plt.yticks(np.arange(0.2, 1.1, 0.1))
    
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.tick_params(which='both', direction='in', top=True, right=True)
    
    plt.grid(True, which='major', linestyle='--', alpha=0.5)
    plt.legend(loc='lower left', frameon=False)
    plt.tight_layout()
    
    # Store the plot in a dedicated 'plots' directory
    plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))
    os.makedirs(plots_dir, exist_ok=True)
    
    filename = f"IL_{il_db}dB_n{samples}_N{N_values[0]}-{N_values[-1]}.pdf"
    filepath = os.path.join(plots_dir, filename)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Plot salvato con successo in: {filepath}")
    # --------------------------------------------------------
    
    plt.show()

if __name__ == "__main__":
    N_range = range(4, 21, 2)  # [4, 6, 8, 10, 12, 14, 16, 18, 20]
    SAMPLES = 500              
    LOSS_DB = 1.5
    
    
    data = run_loss_benchmark(N_range, SAMPLES, LOSS_DB)
    print("\nRisultati medi:", data)
    
    plot_figure_4a(list(N_range), data, LOSS_DB, SAMPLES)