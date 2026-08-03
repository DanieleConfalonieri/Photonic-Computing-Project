import numpy as np
from scipy.stats import unitary_group
import matplotlib.pyplot as plt
import seaborn as sns

from circuits import UGMZICircuit

def run_ugmzi_test():
    N = 3  # Dimensione del circuito di test (3x3 è sufficiente per validare la logica)
    print(f"--- INIZIO TEST UGMZI {N}x{N} ---")
    
    # 1. Generazione di una matrice unitaria target casuale
    np.random.seed(42) # Fissiamo il seed per riproducibilità
    U_target = unitary_group.rvs(N)
    
    print("\n[Target U]")
    print(np.round(U_target, 3))
    
    # 2. Inizializzazione del circuito (esegue la decomposizione e PSO+LM)
    print("\nEsecuzione della decomposizione UGMZI (estrazione thetas e omegas)...")
    circuit = UGMZICircuit(U_target)
    
    # 3. Ricostruzione della matrice sperimentale
    U_exp = circuit.experimentalMatrix
    
    print("\n[Experimental U (Ricostruita dall'hardware)]")
    print(np.round(U_exp, 3))
    
    # 4. Verifica dell'errore (Modulo della differenza)
    diff = np.abs(U_target - U_exp)
    print("\n[Errore assoluto |U_target - U_exp|]")
    print(np.round(diff, 5))
    
    # 5. Verifica della Fidelity globale
    fidelity = circuit.fidelity
    print(f"\n=> FIDELITY DEL CIRCUITO: {fidelity:.6f}")
    
    # 6. Analisi dei nodi generati
    print("\n[Dettaglio Nodi Hardware SGMZI]")
    for i, node in enumerate(circuit._nodes):
        print(f"  Nodo {i}: dimensione {node.N}x{node.N}")
        print(f"    Thetas: {np.round(node._thetas, 3)}")
        print(f"    Omegas: {np.round(node._omegas, 3)}")

    # Plot visuale (opzionale) per confrontare le magnitudini
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.heatmap(np.abs(U_target), annot=True, cmap="Blues", cbar=False)
    plt.title("Magnitudini Target")
    
    plt.subplot(1, 2, 2)
    sns.heatmap(np.abs(U_exp), annot=True, cmap="Reds", cbar=False)
    plt.title("Magnitudini Ricostruite (Exp)")
    plt.show()


if __name__ == "__main__":
    run_ugmzi_test()