import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Modifica l'importazione se i file si chiamano diversamente nel tuo environment
from circuits import SVDCircuit, UGMZICircuit, ReckCircuit, ClementsCircuit

def run_svd_test():
    N = 3 
    print(f"--- INIZIO TEST SVD (Non-Unitario) {N}x{N} ---")
    
    # 1. Generazione di una matrice generica complessa casuale
    np.random.seed(42) 
    # Generiamo parti reali e immaginarie in modo indipendente
    A_target = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    
    print("\n[Target A (Matrice Ideale Originale)]")
    print(np.round(A_target, 3))
    
    # 2. Inizializzazione del circuito (esegue SVD + Decomposizioni Unitarie)
    print("\nEsecuzione decomposizione hardware (U, Sigma, Vh) tramite UGMZI...")
    # Qui iniettiamo la classe per la parte unitaria
    circuit = SVDCircuit(A_target, unitary_circuit_class=UGMZICircuit)
    
    # 3. Ricostruzione della matrice sperimentale
    A_exp = circuit.experimentalMatrix
    
    # [CRITICITÀ] Scaliamo il target per il confronto diretto degli elementi
    _, S, _ = np.linalg.svd(A_target)
    sigma_max = np.max(S)
    A_target_scaled = A_target / sigma_max if sigma_max > 0 else A_target

    print("\n[Experimental A (Uscita hardware limitata dalla passività)]")
    print(np.round(A_exp, 3))
    
    # 4. Verifica dell'errore (Modulo della differenza sul target scalato)
    diff = np.abs(A_target_scaled - A_exp)
    print("\n[Errore assoluto |A_target_scaled - A_exp|]")
    print(np.round(diff, 5))
    
    # 5. Verifica della Fidelity globale (passiamo il target originale, la formula compensa)
    fidelity = circuit.fidelity
    print(f"\n=> FIDELITY DEL CIRCUITO (Hilbert-Schmidt): {fidelity:.6f}")

    # Plot visuale per confrontare le magnitudini
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    # Mostriamo il target scalato altrimenti i colori della heatmap non sarebbero comparabili
    sns.heatmap(np.abs(A_target_scaled), annot=True, cmap="Blues", cbar=False)
    plt.title("Magnitudini Target (Scalate)")
    
    plt.subplot(1, 2, 2)
    sns.heatmap(np.abs(A_exp), annot=True, cmap="Reds", cbar=False)
    plt.title("Magnitudini Ricostruite (Exp)")
    plt.show()

if __name__ == "__main__":
    run_svd_test()
