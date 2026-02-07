import rosbag
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def extract_mean_latency(bag_path, topic='/vlm/inference/time'):
    """Estrae i messaggi da un bag file e calcola la media."""
    latencies = []
    try:
        with rosbag.Bag(bag_path, 'r') as bag:
            for _, msg, _ in bag.read_messages(topics=[topic]):
                # Assumiamo che il messaggio sia di tipo std_msgs/Float32 o simile
                # Se il messaggio è una stringa o ha un campo specifico, adattalo (es. msg.data)
                try:
                    latencies.append(msg.data)
                except AttributeError:
                    # Se il messaggio è una stringa JSON, caricalo
                    import json
                    data = json.loads(msg.data)
                    latencies.append(data['inference_time'])
        
        if not latencies:
            print(f"Attenzione: Nessun dato trovato nel topic {topic} in {bag_path}")
            return None
        
        return np.mean(latencies)
    except Exception as e:
        print(f"Errore nella lettura di {bag_path}: {e}")
        return None

def main():
    # Configurazione percorsi
    bag_folder = "./" # Cambia con il tuo percorso
    bag_files = glob.glob(os.path.join(bag_folder, "*.bag"))
    bag_files.sort() # Ordina i file alfabeticamente

    results = {}
    print(bag_files)

    for path in bag_files:
        filename = os.path.basename(path)
        print(f"Elaborazione di {filename}...")
        
        mean_val = extract_mean_latency(path)
        if mean_val is not None:
            results[filename] = mean_val

    if not results:
        print("Nessun dato valido estratto. Esco.")
        return

    # Preparazione Plot
    names = list(results.keys())
    means = list(results.values())

    plt.figure(figsize=(12, 6))
    
    # Grafico a barre
    bars = plt.bar(names, means, color='skyblue', edgecolor='navy')
    
    # Aggiungi i valori sopra le barre
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 4), ha='center', va='bottom')

    plt.title('Media Tempi di Inferenza VLM per Configurazione', fontsize=14)
    plt.xlabel('Bag File (Configurazione)', fontsize=12)
    plt.ylabel('Tempo Medio di Inferenza (secondi)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Salva e mostra
    plt.savefig('vlm_latency_comparison.png')
    print("\nGrafico salvato come 'vlm_latency_comparison.png'")
    plt.show()

if __name__ == "__main__":
    main()