#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# Configurazione Goal e Start
GOAL = np.array([9.0, 9.0])
START = np.array([0.0, 0.0])
DIST_TOL = 0.5
MAX_SPEED = 0.5  # Per normalizzazione radar

def analyze():
    df = pd.read_csv('raw_navigation_data.csv')
    processed_runs = []

    # Ordiniamo gli algoritmi per coerenza visiva
    df['algo'] = pd.Categorical(df['algo'], categories=sorted(df['algo'].unique(), key=len, reverse=True), ordered=True)

    for (algo, run), group in df.groupby(['algo', 'run']):
        group = group.sort_values('t')
        
        # 1. Trova l'istante in cui arriva al goal
        group['dist_to_goal'] = np.sqrt((group['x'] - GOAL[0])**2 + (group['y'] - GOAL[1])**2)
        arrival_mask = group['dist_to_goal'] < DIST_TOL
        
        if arrival_mask.any():
            idx_arrival = arrival_mask.idxmax()
            clean_data = group.loc[:idx_arrival] # Tronca dopo l'arrivo
            success = 1
        else:
            clean_data = group # Non è mai arrivato
            success = 0

        # 2. Calcolo Metriche condizionali al successo
        # Se non ha avuto successo, path_ratio e speed non sono metriche valide di performance
        if success == 1:
            path_len = np.sum(np.sqrt(np.diff(clean_data['x'])**2 + np.diff(clean_data['y'])**2))
            # d_euclidea / lunghezza (Path Efficiency)
            path_ratio = np.linalg.norm(GOAL - START) / path_len
            
            # Calcolo velocità
            dt = clean_data['t'].diff().fillna(0)
            dist_step = np.sqrt(clean_data['x'].diff()**2 + clean_data['y'].diff()**2).fillna(0)
            avg_speed = (dist_step / dt).replace([np.inf, -np.inf], 0).mean()
        else:
            # Usiamo NaN per non inquinare le medie e i boxplot
            path_ratio = np.nan
            avg_speed = np.nan

        processed_runs.append({
            'Algorithm': algo,
            'Success': success,
            'Path_Ratio': path_ratio,
            'Avg_Speed': avg_speed,
            'Min_Dist': clean_data['min_dist'].min()
        })

    results_df = pd.DataFrame(processed_runs)
    
    # Print Media e Varianza (Ignora i NaN automaticamente)
    print("\n--- STATISTICHE FINALI (Medie calcolate solo su Successi) ---")
    summary = results_df.groupby('Algorithm').agg(['mean', 'var']).round(4)
    print(summary)

    # --- PLOTTING ---
    algos = sorted(results_df['Algorithm'].unique())
    metrics = ['Success', 'Path_Ratio', 'Avg_Speed', 'Min_Dist']

    # 1. BOX PLOTS (3 Subplots, uno per ciascun algoritmo)
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
    fig1.suptitle('Distribuzione Metriche per Algoritmo', fontsize=16)
    
    for i, algo in enumerate(algos):
        algo_data = results_df[results_df['Algorithm'] == algo][metrics]
        sns.boxplot(data=algo_data, ax=axes1[i], palette="Set3", showmeans=True)
        axes1[i].set_title(f"Algoritmo: {algo}")
        axes1[i].set_xticklabels(metrics, rotation=45)
        axes1[i].grid(True, alpha=0.3)
        # Fissiamo i limiti Y per rendere i boxplot confrontabili tra subplots
        axes1[i].set_ylim(-0.1, 1.2 if i < 3 else None) # Success, Ratio e Speed stanno tra 0 e 1

    # 2. RADAR CHARTS (3 Subplots, uno per algoritmo con scala unificata)
    fig2 = plt.figure(figsize=(18, 6))
    fig2.suptitle('Radar Charts - Performance Profile (Medie)', fontsize=16)
    
    # Normalizzazione per Radar
    norm_df = results_df.copy()
    norm_df['Avg_Speed'] = norm_df['Avg_Speed'] / MAX_SPEED
    # Per Min_Dist normalizziamo rispetto al massimo globale trovato per scalarlo 0-1
    max_min_dist = results_df['Min_Dist'].max()
    norm_df['Min_Dist'] = norm_df['Min_Dist'] #/ max_min_dist
    
    means_norm = norm_df.groupby('Algorithm').mean()
    angles = np.linspace(0, 2*pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    for i, algo in enumerate(algos):
        ax = fig2.add_subplot(1, 3, i+1, polar=True)
        values = means_norm.loc[algo].tolist()
        values += values[:1]
        
        ax.plot(angles, values, color='teal', linewidth=2, marker='o')
        ax.fill(angles, values, color='teal', alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_title(f"Media: {algo}", pad=20)
        
        # Scala Unificata
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    analyze()