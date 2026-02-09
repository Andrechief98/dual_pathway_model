import rosbag
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re

def extract_mean_latency(bag_path, topic='/vlm/inference/time'):
    """Estrae i messaggi da un bag file e calcola la media dei tempi di inferenza."""
    latencies = []
    try:
        with rosbag.Bag(bag_path, 'r') as bag:
            for _, msg, _ in bag.read_messages(topics=[topic]):
                try:
                    latencies.append(msg.data)
                except AttributeError:
                    import json
                    data = json.loads(msg.data)
                    latencies.append(data.get('inference_time', 0))
        
        if not latencies:
            return None
        
        return np.mean(latencies)
    except Exception as e:
        print(f"Errore nella lettura di {bag_path}: {e}")
        return None

def plot_mpc_performance(results):
    """
    Genera il grafico con:
    - Marker quadrati e linee (s-)
    - Notazione matematica Mathtext (stile LaTeX)
    - Ordinamento numerico basato sul nome del file
    """
    filenames = list(results.keys())
    means = list(results.values())

    # Estrazione numero di oggetti per l'ordinamento dell'asse X
    x_values = []
    for f in filenames:
        nums = re.findall(r'\d+', f)
        # Se trova un numero lo usa, altrimenti usa un indice progressivo
        x_values.append(int(nums[0]) if nums else 0)

    # Ordinamento dei dati per evitare linee incrociate
    sorted_indices = np.argsort(x_values)
    x_plot = np.array(x_values)[sorted_indices]
    y_plot = np.array(means)[sorted_indices]

    plt.figure(figsize=(10, 6))
    
    # Plot: quadrati ('s') e linea continua ('-')
    plt.plot(x_plot, y_plot, 's-', color='tab:red', linewidth=2, 
             markersize=8, label=r'$\mathrm{Inference\ Mean}$')

    # Titoli e Assi con sintassi Mathtext (simil-LaTeX)
    # \mathrm{} serve per evitare il corsivo tipico delle formule
    plt.title(r'$\mathrm{Inference\ Time\ vs.\ Number\ of\ Objects}$', fontsize=16)
    plt.xlabel(r'$\mathrm{N_{obs}}$', fontsize=14)
    plt.ylabel(r'$\mathrm{t\ [s]}$', fontsize=14)
    
    # Griglia e stile
    plt.xticks(x_plot)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    plt.show()

def main():
    # Prende la cartella dove si trova lo script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bag_files = glob.glob(os.path.join(script_dir, "*.bag"))
    
    if not bag_files:
        print(f"Nessun file .bag trovato in: {script_dir}")
        return

    results = {}

    for path in bag_files:
        filename = os.path.basename(path)
        print(f"Elaborazione di {filename}...")
        
        mean_val = extract_mean_latency(path)
        if mean_val is not None:
            results[filename] = mean_val

    if results:
        # Richiamo della funzione di plot all'interno del main
        plot_mpc_performance(results)
    else:
        print("Dati insufficienti per generare il grafico.")

if __name__ == "__main__":
    main()