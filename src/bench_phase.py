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


def evaluate_single_sample(seed, N, std_noise, mean_noise=0.0):
    """
    Evaluates fidelity of UGMZI, Clements, and Reck circuits for a single random unitary matrix
    with Gaussian phase noise applied to all phase shifters.
    """
    # Seed both numpy legacy RNG and Generator for reproducibility across processes
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    U_target = unitary_group.rvs(N, random_state=rng)

    # UGMZI Circuit
    circ_u = UGMZICircuit(U_target)
    circ_u.setPhaseGaussianNoise(mean_noise, std_noise)
    fid_u = circ_u.fidelity

    # Clements Circuit
    circ_c = ClementsCircuit(U_target)
    circ_c.setPhaseGaussianNoise(mean_noise, std_noise)
    fid_c = circ_c.fidelity

    # Reck Circuit
    circ_r = ReckCircuit(U_target)
    circ_r.setPhaseGaussianNoise(mean_noise, std_noise)
    fid_r = circ_r.fidelity

    return fid_u, fid_c, fid_r


def run_phase_noise_sweep(N_fixed, samples, std_values, mean_noise=0.0):
    """
    Runs benchmark for a fixed N while sweeping the standard deviation of phase noise.
    """
    results = {'UGMZI': [], 'Clements': [], 'Reck': []}
    num_cores = mp.cpu_count()
    print(f"Starting multiprocessing with {num_cores} cores... (Fixed N = {N_fixed}, mean = {mean_noise})\n")

    for std in std_values:
        print(f"--- Benchmarking N={N_fixed}, Phase Noise Std={std:.4f} rad (Samples: {samples}) ---")
        worker_func = partial(evaluate_single_sample, N=N_fixed, std_noise=std, mean_noise=mean_noise)
        seeds = [hash((N_fixed, std, i)) % (2**32) for i in range(samples)]

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


def run_phase_N_sweep(N_values, samples, std_noise=0.1, mean_noise=0.0):
    """
    Runs benchmark for a fixed phase noise standard deviation while sweeping N.
    """
    results = {'UGMZI': [], 'Clements': [], 'Reck': []}
    num_cores = mp.cpu_count()
    print(f"Starting multiprocessing with {num_cores} cores... (Fixed Std = {std_noise:.4f} rad, mean = {mean_noise})\n")

    for N in N_values:
        print(f"--- Benchmarking N={N}, Phase Noise Std={std_noise:.4f} rad (Samples: {samples}) ---")
        worker_func = partial(evaluate_single_sample, N=N, std_noise=std_noise, mean_noise=mean_noise)
        seeds = [hash((N, std_noise, i)) % (2**32) for i in range(samples)]

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


def plot_phase_noise_fixed_N(std_values, data, N_fixed, samples, mean_noise=0.0):
    """
    Plots fidelity vs phase noise standard deviation for a fixed N.
    """
    plt.figure(figsize=(6, 5))
    ax = plt.gca()

    plt.plot(std_values, data['UGMZI'], marker='o', color='darkred', markersize=5, linewidth=1, label='UGMZI')
    plt.plot(std_values, data['Clements'], marker='^', color='black', markersize=5, linewidth=1, label='Clements')
    plt.plot(std_values, data['Reck'], marker='v', color='tab:blue', markersize=5, linewidth=1, label='Reck')

    title_text = f"N = {N_fixed} ($\mu = {mean_noise}$ rad)"
    plt.title(title_text, fontsize=10, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='square,pad=0.3'))

    plt.xlabel(r"Phase noise standard deviation $\sigma$ (rad)", fontsize=12)
    plt.ylabel("Fidelity", fontsize=12)

    plt.xlim(min(std_values), max(std_values))
    plt.ylim(0.0, 1.05)

    y_ticks = np.arange(0.0, 1.1, 0.1)
    plt.yticks(y_ticks)

    std_span = max(std_values) - min(std_values)
    if std_span > 0:
        ax.xaxis.set_minor_locator(MultipleLocator(std_span / 10))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.tick_params(which='both', direction='in', top=True, right=True)

    plt.grid(True, which='major', linestyle='--', alpha=0.5)
    plt.legend(loc='lower left', frameon=False)
    plt.tight_layout()

    plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))
    os.makedirs(plots_dir, exist_ok=True)

    filename = f"PhaseNoise_sweep_N{N_fixed}_n{samples}.pdf"
    filepath = os.path.join(plots_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Plot salvato con successo in: {filepath}")

    plt.show()
    plt.close()


def plot_phase_noise_sweep_N(N_values, data, std_noise, samples, mean_noise=0.0):
    """
    Plots fidelity vs N for a fixed phase noise standard deviation.
    """
    plt.figure(figsize=(6, 5))
    ax = plt.gca()

    plt.plot(N_values, data['UGMZI'], marker='o', color='darkred', markersize=5, linewidth=1, label='UGMZI')
    plt.plot(N_values, data['Clements'], marker='^', color='black', markersize=5, linewidth=1, label='Clements')
    plt.plot(N_values, data['Reck'], marker='v', color='tab:blue', markersize=5, linewidth=1, label='Reck')

    title_text = f"$\sigma = {std_noise}$ rad ($\mu = {mean_noise}$ rad)"
    plt.title(title_text, fontsize=10, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='square,pad=0.3'))

    plt.xlabel("Number of Inputs $N$", fontsize=12)
    plt.ylabel("Fidelity", fontsize=12)

    plt.xlim(min(N_values), max(N_values))
    plt.ylim(0.0, 1.05)

    plt.xticks(range(min(N_values), max(N_values) + 1, 2 if len(N_values) <= 8 else 4))
    plt.yticks(np.arange(0.0, 1.1, 0.1))

    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.tick_params(which='both', direction='in', top=True, right=True)

    plt.grid(True, which='major', linestyle='--', alpha=0.5)
    plt.legend(loc='lower left', frameon=False)
    plt.tight_layout()

    plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))
    os.makedirs(plots_dir, exist_ok=True)

    filename = f"PhaseNoise_std{std_noise}_n{samples}_N{N_values[0]}-{N_values[-1]}.pdf"
    filepath = os.path.join(plots_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Plot salvato con successo in: {filepath}")

    plt.show()
    plt.close()


if __name__ == "__main__":
    # Benchmark parameters
    SAMPLES = int(os.environ.get('BENCH_SAMPLES', 500))  # Numero di matrici unitarie casuali per punto
    MEAN_NOISE = 0.0    # Media del rumore gaussiano di fase

    # Toggles per decidere quali diagnostiche eseguire
    RUN_N4_SWEEP = True
    RUN_N8_SWEEP = True
    RUN_N_SWEEP = True

    # -------------------------------------------------------------------------
    # 1) N = 4: Fidelity al variare della standard deviation (media = 0)
    # -------------------------------------------------------------------------
    if RUN_N4_SWEEP:
        N_4 = 4
        STD_VALUES_N4 = np.linspace(0.0, 0.2, 11)  # Sweep standard deviation da 0.0 a 0.2 rad
        print(f"\n==========================================")
        print(f"Benchmark 1: N={N_4}, Sweep Phase Noise Std")
        print(f"==========================================")
        data_n4 = run_phase_noise_sweep(N_fixed=N_4, samples=SAMPLES, std_values=STD_VALUES_N4, mean_noise=MEAN_NOISE)
        print("\nRisultati medi N=4:", data_n4)
        plot_phase_noise_fixed_N(STD_VALUES_N4, data_n4, N_fixed=N_4, samples=SAMPLES, mean_noise=MEAN_NOISE)

    # -------------------------------------------------------------------------
    # 2) N = 8: Fidelity al variare della standard deviation (media = 0)
    # -------------------------------------------------------------------------
    if RUN_N8_SWEEP:
        N_8 = 8
        STD_VALUES_N8 = np.linspace(0.0, 0.2, 11)  # Sweep standard deviation da 0.0 a 0.2 rad
        print(f"\n==========================================")
        print(f"Benchmark 2: N={N_8}, Sweep Phase Noise Std")
        print(f"==========================================")
        data_n8 = run_phase_noise_sweep(N_fixed=N_8, samples=SAMPLES, std_values=STD_VALUES_N8, mean_noise=MEAN_NOISE)
        print("\nRisultati medi N=8:", data_n8)
        plot_phase_noise_fixed_N(STD_VALUES_N8, data_n8, N_fixed=N_8, samples=SAMPLES, mean_noise=MEAN_NOISE)

    # -------------------------------------------------------------------------
    # 3) Standard deviation = 0.1 rad: Fidelity al variare di N (media = 0)
    # -------------------------------------------------------------------------
    if RUN_N_SWEEP:
        STD_FIXED = 0.1
        N_RANGE = range(4, 21, 2)  # [4, 6, 8, 10, 12, 14, 16, 18, 20]
        print(f"\n==========================================")
        print(f"Benchmark 3: Std={STD_FIXED} rad, Sweep N")
        print(f"==========================================")
        data_n_sweep = run_phase_N_sweep(N_values=N_RANGE, samples=SAMPLES, std_noise=STD_FIXED, mean_noise=MEAN_NOISE)
        print(f"\nRisultati medi Std={STD_FIXED}:", data_n_sweep)
        plot_phase_noise_sweep_N(list(N_RANGE), data_n_sweep, std_noise=STD_FIXED, samples=SAMPLES, mean_noise=MEAN_NOISE)

    # Restore the original environment variables
    for key, original_value in original_env.items():
        if original_value is not None:
            os.environ[key] = original_value
        else:
            os.environ.pop(key, None)
