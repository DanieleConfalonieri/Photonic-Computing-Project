import numpy as np
from scipy.stats import unitary_group
from tqdm import tqdm 
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from circuits import UGMZICircuit, ClementsCircuit, ReckCircuit

def db_to_alpha(il_db):
    return il_db * np.log(10) / 20.0

def run_loss_benchmark(N_values, samples, il_db):
    results = {'UGMZI': [], 'Clements': [], 'Reck': []}
    alpha_loss = db_to_alpha(il_db)

    for N in N_values:
        print(f"\n--- Benchmarking N={N} (Loss: {il_db} dB) ---")
        fid_u, fid_c, fid_r = [], [], []
        
        for _ in tqdm(range(samples)):
            U_target = unitary_group.rvs(N)
            
            # UGMZI
            circ_u = UGMZICircuit(U_target)
            circ_u.setUniformLoss(alpha_loss, "Node")
            fid_u.append(circ_u.fidelity)
            
            # Clements
            circ_c = ClementsCircuit(U_target)
            circ_c.setUniformLoss(alpha_loss, "Node")
            fid_c.append(circ_c.fidelity)
            
            # Reck
            circ_r = ReckCircuit(U_target)
            circ_r.setUniformLoss(alpha_loss, "Node")
            fid_r.append(circ_r.fidelity)
            
        results['UGMZI'].append(np.mean(fid_u))
        results['Clements'].append(np.mean(fid_c))
        results['Reck'].append(np.mean(fid_r))
        
    return results

def plot_figure_4a(N_values, data, il_db):
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
    plt.show()

if __name__ == "__main__":
    N_range = range(4, 21) 
    
    SAMPLES = 5 
    LOSS_DB = 1.5
    
    data = run_loss_benchmark(N_range, SAMPLES, LOSS_DB)
    print("\nAverage results:", data)
    
    plot_figure_4a(list(N_range), data, LOSS_DB)