import rosbag
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import re  # Per estrarre il numero dal nome del file
from collections import defaultdict

# --- Configurazione ---
script_dir = os.path.dirname(os.path.abspath(__file__))

ALGO_NAMES = ["MPC", "MPC_lr", "MPC_hr", "MPC_dp", "APF"]
X_LIMITS = [-1.5, 12]
Y_LIMITS = [-4.5, 4.5]

model_mapping = {
    "mir": "MiR",
    "cylinder": "Cylinder",
    "rover": "Rover",
    "person": "Person",
    "cardboard_box": "Cardboard box"
}

radius_mapping = {
    "Cylinder": 0.3,
    "Rover": 1.0,
    "Person": 0.3,
    "Cardboard box": 0.3
}

def identify_algorithm(filename):
    sorted_algos = sorted(ALGO_NAMES, key=len, reverse=True)
    for algo in sorted_algos:
        if filename.startswith(algo + "_"):
            return algo
    return None

def extract_dynamic_data(bag_file_path):
    data = {'robot': [], 'obstacles': defaultdict(list)}
    try:
        with rosbag.Bag(bag_file_path, 'r') as bag:
            start_t = None
            for topic, msg, t in bag.read_messages(topics=['/gazebo/model_states']):
                if start_t is None: start_t = t.to_sec()
                rel_t = t.to_sec() - start_t
                for i, name in enumerate(msg.name):
                    if name in model_mapping:
                        label = model_mapping[name]
                        p = msg.pose[i].position
                        if name == "mir":
                            data['robot'].append({'t': rel_t, 'x': p.x, 'y': p.y})
                        else:
                            data['obstacles'][label].append({'t': rel_t, 'x': p.x, 'y': p.y})
        return pd.DataFrame(data['robot']), {lab: pd.DataFrame(pts) for lab, pts in data['obstacles'].items()}
    except Exception as e:
        print(f"Errore: {e}")
        return None, None

def plot_dynamic_comparison_grid():
    all_files = [f for f in os.listdir(script_dir) if f.endswith('.bag')]
    grouped_files = defaultdict(list)
    for f in all_files:
        algo = identify_algorithm(f)
        if algo: grouped_files[algo].append(f)

    active_algos = [a for a in ALGO_NAMES if a in grouped_files]
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 6)
    slots = [gs[0, 0:2], gs[0, 2:4], gs[0, 4:6], gs[1, 1:3], gs[1, 3:5]]
    cmap = plt.get_cmap('tab10')

    for idx, algo in enumerate(active_algos):
        ax = fig.add_subplot(slots[idx])
        algo_color = cmap(idx % 10)
        
        # Ordiniamo i file per essere sicuri della sequenza
        for file_idx, bag_file in enumerate(sorted(grouped_files[algo])):
            df_r, obs_dfs = extract_dynamic_data(os.path.join(script_dir, bag_file))
            if df_r is None or df_r.empty: continue

            r_x, r_y = df_r['x'].to_numpy(), df_r['y'].to_numpy()
            ax.plot(r_x, r_y, color=algo_color, alpha=0.4, linewidth=1.2, zorder=5)
            
            run_index = bag_file.split('_')[-1].replace('.bag', '')

            # Posizionamento a metà traiettoria
            mid_idx = len(r_x) // 2
            ax.text(r_x[mid_idx], r_y[mid_idx], run_index, 
                    fontsize=10, color=algo_color, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1),
                    zorder=12)

            # Ostacoli (disegnati una sola volta per subplot)
            if file_idx == 0:
                for obs_label, df_o in obs_dfs.items():
                    if df_o.empty: continue
                    o_x, o_y = df_o['x'].to_numpy(), df_o['y'].to_numpy()
                    ax.plot(o_x, o_y, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)
                    radius = radius_mapping.get(obs_label, 0.3)
                    ax.add_patch(plt.Circle((o_x[-1], o_y[-1]), radius, color='red', fill=False, alpha=0.3))
                    ax.text(o_x[-1], o_y[-1] - 0.6, obs_label, fontsize=7, color='red', ha='center')

        ax.set_xlim(X_LIMITS)
        ax.set_ylim(Y_LIMITS)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_title(rf"$\mathrm{{{algo}}}$", fontsize=16)

    plt.suptitle(r"$\mathrm{Dynamic\ Trajectories\ -\ Filename-based\ Indexing}$", fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    plot_dynamic_comparison_grid()