# #!/usr/bin/env python3

# import rosbag
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches 
# import numpy as np
# import math
# from collections import defaultdict

# # --- Configuration ---
# script_dir = os.path.dirname(os.path.abspath(__file__))

# # Ordine specifico richiesto
# ALGO_NAMES = ["MPC", "MPC_lr", "MPC_hr", "MPC_dp", "APF"]

# # Limiti assi fissi per comparazione uniforme
# X_LIMITS = [-1.5, 11.5]
# Y_LIMITS = [-4.5, 4.5]

# model_mapping = {
#     "mir"           : "MiR",
#     "cylinder"      : "Cylinder", 
#     "rover"         : "Rover",
#     "person"        : "Person",
#     "cardboard_box" : "Cardboard box"
# }

# radius_mapping = {
#     "Cylinder"      : 0.3,
#     "Rover"         : 1.0,
#     "Person"        : 0.3,
#     "Cardboard box" : 0.3
# }

# def identify_algorithm(filename):
#     """Identifica l'algoritmo verificando il prefisso del file."""
#     sorted_algos = sorted(ALGO_NAMES, key=len, reverse=True)
#     for algo in sorted_algos:
#         if filename.startswith(algo + "_"):
#             return algo
#     return None

# def extract_data_simple(bag_file_path, model_map):
#     data = {'robot_path': [], 'obstacles': {}}
#     try:
#         with rosbag.Bag(bag_file_path, 'r') as bag:
#             for topic, msg, t in bag.read_messages(topics=['/gazebo/model_states']):
#                 for i, name in enumerate(msg.name):
#                     if name in model_map:
#                         label = model_map[name]
#                         pose = msg.pose[i]
#                         if name == "mir":
#                             data['robot_path'].append({'x': pose.position.x, 'y': pose.position.y})
#                         elif label not in data['obstacles']:
#                             data['obstacles'][label] = {'x': pose.position.x, 'y': pose.position.y}
#         data['robot_path'] = pd.DataFrame(data['robot_path'])

#         return data
    
    
#     except Exception: return None

# def format_latex_name(name):
#     """Formatta i nomi per la visualizzazione LaTeX sopra il plot."""
#     if "_" in name:
#         base, sub = name.split("_")
#         return rf"$\mathrm{{{base}_{{{sub}}}}}$"
#     return rf"$\mathrm{{{name}}}$"

# def plot_top_titles_layout():
#     all_files = [f for f in os.listdir(script_dir) if f.endswith('.bag')]
#     grouped_files = defaultdict(list)
#     for f in all_files:
#         algo = identify_algorithm(f)
#         if algo: grouped_files[algo].append(f)

#     active_algos = [a for a in ALGO_NAMES if a in grouped_files]
#     if not active_algos: 
#         print("Nessun file trovato!")
#         return

#     # Setup Figura (2 righe, 3 colonne)
#     fig = plt.figure(figsize=(18, 10))
#     gs = fig.add_gridspec(2, 6) # Griglia a 6 colonne per centrare i 2 sotto
    
#     # Mappatura slot: 
#     # Sopra: (0,0-1), (0,2-3), (0,4-5) 
#     # Sotto: (1,1-2), (1,3-4) -> per averli centrati
#     slots = [
#         gs[0, 0:2], gs[0, 2:4], gs[0, 4:6],
#         gs[1, 1:3], gs[1, 3:5]
#     ]
    
#     cmap = plt.get_cmap('tab10')
#     global_obstacles = None

#     for idx, algo in enumerate(active_algos):
#         ax = fig.add_subplot(slots[idx])
#         color = cmap(idx % 10)
        
#         # 1. Plot Traiettorie
#         for bag_file in grouped_files[algo]:
#             data = extract_data_simple(os.path.join(script_dir, bag_file), model_mapping)
#             if data is None or data['robot_path'].empty: continue
#             if global_obstacles is None: global_obstacles = data['obstacles']

#             x = data['robot_path']['x'].to_numpy()
#             y = data['robot_path']['y'].to_numpy()
#             ax.plot(x, y, color=color, alpha=0.3, linewidth=1.0, zorder=2)
#             ax.scatter(x[-1], y[-1], color=color, s=15, alpha=0.4, zorder=3)

#         # 2. Plot Ostacoli (Stile Originale)
#         if global_obstacles:
#             for label, pos in global_obstacles.items():
#                 ox, oy = pos['x'], pos['y']
#                 r = radius_mapping.get(label, 0.3)
#                 ax.scatter(ox, oy, color='black', marker='X', s=90, zorder=5)
#                 ax.add_patch(plt.Circle((ox, oy), r, color='red', fill=True, alpha=0.1, zorder=1))
#                 ax.add_patch(plt.Circle((ox, oy), r, color='red', fill=False, linewidth=1.5, zorder=3))
#                 ax.text(ox + 0.1, oy + 0.1, label, fontsize=9, fontweight='bold')

#         # --- Formattazione Subplot ---
#         ax.set_xlim(X_LIMITS)
#         ax.set_ylim(Y_LIMITS)
#         ax.set_aspect('equal')
#         ax.grid(True, linestyle=':', alpha=0.6)
        
#         # Titolo Superiore in LaTeX
#         ax.set_title(format_latex_name(algo), fontsize=20, pad=15)
        
#         ax.set_xlabel(r'$X \ [\mathrm{m}]$', fontsize=10)
#         ax.set_ylabel(r'$Y \ [\mathrm{m}]$', fontsize=10)

#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     plot_top_titles_layout()


#!/usr/bin/env python3

import rosbag
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
import numpy as np
import math
from collections import defaultdict

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# Ordine specifico richiesto
ALGO_NAMES = ["MPC", "MPC_lr", "MPC_hr", "MPC_dp", "APF"]

# Limiti assi fissi per comparazione uniforme
X_LIMITS = [-1.5, 11.5]
Y_LIMITS = [-4.5, 4.5]

model_mapping = {
    "mir"           : "MiR",
    "cylinder"      : "Cylinder", 
    "rover"         : "Rover",
    "person"        : "Person",
    "cardboard_box" : "Cardboard box"
}

radius_mapping = {
    "Cylinder"      : 0.3,
    "Rover"         : 1.0,
    "Person"        : 0.3,
    "Cardboard box" : 0.3
}

def identify_algorithm(filename):
    """Identifica l'algoritmo verificando il prefisso del file."""
    sorted_algos = sorted(ALGO_NAMES, key=len, reverse=True)
    for algo in sorted_algos:
        if filename.startswith(algo + "_"):
            return algo
    return None

def extract_data_simple(bag_file_path, model_map):
    data = {'robot_path': [], 'obstacles': {}}
    try:
        with rosbag.Bag(bag_file_path, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=['/gazebo/model_states']):
                for i, name in enumerate(msg.name):
                    if name in model_map:
                        label = model_map[name]
                        pose = msg.pose[i]
                        if name == "mir":
                            data['robot_path'].append({'x': pose.position.x, 'y': pose.position.y})
                        elif label not in data['obstacles']:
                            data['obstacles'][label] = {'x': pose.position.x, 'y': pose.position.y}
        data['robot_path'] = pd.DataFrame(data['robot_path'])
        return data
    except Exception: return None

def format_latex_name(name):
    """Formatta i nomi per la visualizzazione LaTeX sopra il plot."""
    if "_" in name:
        base, sub = name.split("_")
        return rf"$\mathrm{{{base}_{{{sub}}}}}$"
    return rf"$\mathrm{{{name}}}$"

def plot_top_titles_layout():
    all_files = [f for f in os.listdir(script_dir) if f.endswith('.bag')]
    grouped_files = defaultdict(list)
    for f in all_files:
        algo = identify_algorithm(f)
        if algo: grouped_files[algo].append(f)

    active_algos = [a for a in ALGO_NAMES if a in grouped_files]
    if not active_algos: 
        print("Nessun file trovato!")
        return

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 6) 
    
    slots = [
        gs[0, 0:2], gs[0, 2:4], gs[0, 4:6],
        gs[1, 1:3], gs[1, 3:5]
    ]
    
    cmap = plt.get_cmap('tab10')
    global_obstacles = None

    for idx, algo in enumerate(active_algos):
        ax = fig.add_subplot(slots[idx])
        color = cmap(idx % 10)
        
        # 1. Plot Traiettorie
        for bag_file in grouped_files[algo]:
            data = extract_data_simple(os.path.join(script_dir, bag_file), model_mapping)
            if data is None or data['robot_path'].empty: continue
            if global_obstacles is None: global_obstacles = data['obstacles']

            x = data['robot_path']['x'].to_numpy()
            y = data['robot_path']['y'].to_numpy()
            
            # Plot linea
            ax.plot(x, y, color=color, alpha=0.3, linewidth=1.0, zorder=2)
            
            # --- AGGIUNTA INDICE RUN ---
            # Estraiamo il numero dal nome del file (es: "MPC_1.bag" -> "1")
            try:
                run_index = bag_file.split('_')[-1].replace('.bag', '')
                mid_point = len(x) // 2 # Metà della traiettoria
                # Inseriamo il testo con un piccolo offset per leggibilità
                ax.text(x[mid_point], y[mid_point], run_index, 
                        fontsize=8, color=color, fontweight='bold',
                        alpha=0.8, zorder=4)
            except Exception:
                pass 
            
            # Punto finale
            ax.scatter(x[-1], y[-1], color=color, s=15, alpha=0.4, zorder=3)

        # 2. Plot Ostacoli
        if global_obstacles:
            for label, pos in global_obstacles.items():
                ox, oy = pos['x'], pos['y']
                r = radius_mapping.get(label, 0.3)
                ax.scatter(ox, oy, color='black', marker='X', s=90, zorder=5)
                ax.add_patch(plt.Circle((ox, oy), r, color='red', fill=True, alpha=0.1, zorder=1))
                ax.add_patch(plt.Circle((ox, oy), r, color='red', fill=False, linewidth=1.5, zorder=3))
                ax.text(ox + 0.1, oy + 0.1, label, fontsize=9, fontweight='bold')

        # --- Formattazione ---
        ax.set_xlim(X_LIMITS)
        ax.set_ylim(Y_LIMITS)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_title(format_latex_name(algo), fontsize=20, pad=15)
        ax.set_xlabel(r'$X \ [\mathrm{m}]$', fontsize=10)
        ax.set_ylabel(r'$Y \ [\mathrm{m}]$', fontsize=10)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_top_titles_layout()