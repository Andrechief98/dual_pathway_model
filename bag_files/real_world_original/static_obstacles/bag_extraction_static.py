#!/usr/bin/env python3

import rosbag
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
import numpy as np
import math
from collections import defaultdict
import json

# Path
script_dir = os.path.dirname(os.path.abspath(__file__))

bag_files_list = [
    "MPC_static.bag", 
    "MPC_lr_static.bag", 
    "MPC_hr_static.bag", 
    "MPC_dp_static.bag",
    "APF_static.bag"
] 

ROBOT_SEMI_AXIS_A = 0.8  # X axis
ROBOT_SEMI_AXIS_B = 0.4  # Y axis

model_mapping = {
    "mir"           : "MiR",
    "cylinder"      : "Cylinder", 
    "rover"         : "Rover",
    "person"        : "Person",
    "cardboard_box" : "Cardboard box"
}

radius_mapping = {
    "Cylinder"      : 0.3,              # Raggio cilindro
    "Rover"         : 1.0,                # Raggio rover
    "Person"        : 0.3,              # Raggio persona
    "Cardboard box" : 0.3
}

legend_mapping = {
    "MPC_static.bag" : "MPC", 
    "MPC_lr_static.bag": "MPC_{lr}", 
    "MPC_hr_static.bag": "MPC_{hr}", 
    "MPC_dp_static.bag": "MPC_{dp}",
    "APF_static.bag" : "APF"
}

def quaternion_to_yaw(q):
    """Converte un quaternione (x, y, z, w) in angolo Yaw (radianti)."""
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def extract_data_from_bag(bag_file_path, model_map):
    data = {
        'robot_path'    : [],                       # List of dictionary: {time, x, y}
        'velocity'      : [],                       # List of dictionary: {time, linear, angular}
        'obstacles'     : {},                       # Dict of {label: {x, y}}
        'fear_levels'   : defaultdict(list),        # { 'Obstacle_A': [{'time': t, 'val': v}, ...], ... }
        'mpc_stats'     : [],
        'low_road_risk' : [],
        'high_road_risk': []
        }
    
    print(f"Reading: {os.path.basename(bag_file_path)}")

    try:
        with rosbag.Bag(bag_file_path, 'r') as bag:
            start_time = None

            topics_list=[
                '/optitracker/model_states', 
                '/cmd_vel', 
                '/fearlevel', 
                '/mpc/statistics',
                '/amygdala/lowroad/risks',
                '/amygdala/highroad/risks'
                ]
            
            for topic, msg, t in bag.read_messages(topics_list):

                # For each bag topic, we consider the first time instant as start_time
                if start_time is None: 
                    start_time = t.to_sec()

                # We extract the relative time of each data with respect the first time instant
                rel_time = t.to_sec() - start_time
                
                # Extraction of robot and obstacles paths
                if topic == '/optitracker/model_states':
                    for i, name in enumerate(msg.name):
                        if name in model_map:
                            label = model_map[name]
                            pose = msg.pose[i]

                            if name == "mir":
                                # Extraction and convertion of robot orientation (from quaternion to radians)
                                yaw = quaternion_to_yaw(pose.orientation)
                                data['robot_path'].append(
                                    {
                                        'time'  :   rel_time, 
                                        'x'     :   pose.position.x, 
                                        'y'     :   pose.position.y,
                                        'yaw'   :   math.degrees(yaw) # Matplotlib usa i gradi
                                    }
                                )
                            elif label not in data['obstacles']:
                                # Extraction of obstacle position
                                data['obstacles'][label] = {
                                    'x'     :       pose.position.x, 
                                    'y'     :       pose.position.y
                                }
                
                # Extraction of robot velocities
                elif topic == '/cmd_vel':
                    data['velocity'].append(
                        {
                            'time'      :       rel_time,
                            'linear'    :       msg.linear.x,
                            'angular'   :       msg.angular.z
                        }
                    )

                # Extraction of robot fear level
                elif topic == '/fearlevel':
                    try:
                        fear_dict = json.loads(msg.data)
                        for raw_name, value in fear_dict.items():

                            # Usiamo il mapping se l'ostacolo è conosciuto, altrimenti il nome raw
                            label = model_map.get(raw_name, raw_name)
                            data['fear_levels'][label].append({'time': rel_time, 'value': value})
                    except Exception as e:
                        print(e)
                        continue
                
                # Extraction of mpc execution time and statistics
                elif topic == '/mpc/statistics':
                    data['mpc_stats'].append({
                        'time': rel_time,
                        'execution_time': msg.execution_time, 
                        'iterations': msg.iterations,
                        'status': msg.status
                    })

                elif topic == '/amygdala/lowroad/risks':
                    objects_risks = json.loads(msg.data)
                    data['low_road_risk'].append({
                        'time': rel_time,
                        'risks': objects_risks
                    })
                elif topic == '/amygdala/highroad/risks':
                    objects_risks = json.loads(msg.data)
                    data['high_road_risk'].append({
                        'time': rel_time,
                        'risks': objects_risks
                    })

    except Exception as e:
        print(f"Error: {e}")
        return None


    # Convert to DataFrames
    data['robot_path']  =   pd.DataFrame(data['robot_path'])
    data['velocity']    =   pd.DataFrame(data['velocity'])
    data['mpc_stats']   =   pd.DataFrame(data['mpc_stats'])
    for label in data['fear_levels']:
        data['fear_levels'][label] = pd.DataFrame(data['fear_levels'][label])
    return data



def plot_multi_trajectory(all_data, DT_FOOTPRINT = 2.0 ):
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = plt.get_cmap('tab10')
    
    # PLOT ROBOT TRAJECTORY
    for i, (file_name, data) in enumerate(all_data.items()):
        df_robot = data['robot_path']
        
        # Check Dataframe
        if df_robot is not None and not df_robot.empty and 'x' in df_robot.columns:
            x_vals = df_robot['x'].to_numpy()
            y_vals = df_robot['y'].to_numpy()
            yaw_vals = df_robot['yaw'].to_numpy()
            time_vals = df_robot['time'].to_numpy()
            color = cmap(i % 10)
            
            # Trajectory plot
            label_name = legend_mapping[file_name]
            ax.plot(
                    x_vals, 
                    y_vals, 
                    label       =   rf'${label_name}$',
                    color       =   color, 
                    linewidth   =   2, 
                    zorder      =   2
                )
            
            # Ellipsoid plot (footprints - each DT_FOOTPRINT seconds)
            max_t = time_vals.max()      
            checkpoint_times = np.arange(0, max_t, DT_FOOTPRINT)

            for t_target in checkpoint_times:
                idx = (np.abs(time_vals - t_target)).argmin()
                ellipse = patches.Ellipse(
                    (x_vals[idx], y_vals[idx]), 
                    width=2*ROBOT_SEMI_AXIS_A, 
                    height=2*ROBOT_SEMI_AXIS_B, 
                    angle=yaw_vals[idx], 
                    color=color, 
                    fill=False, 
                    linestyle='--', 
                    alpha=0.3
                )
                ax.add_patch(ellipse)

            # Final ellipsoid
            final_ellipse = patches.Ellipse(
                (x_vals[-1], y_vals[-1]), 
                width                       =       2*ROBOT_SEMI_AXIS_A, height=2*ROBOT_SEMI_AXIS_B, 
                angle                       =       yaw_vals[-1], 
                color                       =       color, 
                fill                        =       True, 
                alpha                       =       0.2, 
            )
            ax.add_patch(final_ellipse)
            ax.scatter(x_vals[-1], y_vals[-1], color=color, marker='x', s=50)

    # PLOT OBSTACLES POSITIONS
    first_exp_data = list(all_data.values())[0]
    obstacles_dict = first_exp_data.get('obstacles', {}) # Accedi correttamente alla chiave 'obstacles'

    for label, pos in obstacles_dict.items():
        x, y = pos['x'], pos['y']
        radius = radius_mapping.get(label, 0.0)
        
        # Center of the obstacle
        ax.scatter(x, y, color='black', marker='X', s=100, zorder=5)
        
        # Circumference of the obstacle constraint
        if radius > 0:
            circle = plt.Circle((x, y), radius, color='red', fill=True, alpha=0.1, zorder=1)
            edge = plt.Circle((x, y), radius, color='red', fill=False, linestyle='-', linewidth=1.5, zorder=3)
            ax.add_patch(circle)
            ax.add_patch(edge)
        
        ax.text(x + 0.1, y + 0.1, label, fontsize=9, fontweight='bold')

    ax.set_title(r'$\mathrm{Robot\ Trajectories\ &\ Obstacles}$', fontsize=16)
    ax.set_xlabel(r'$X \ [\mathrm{m}]$')
    ax.set_ylabel(r'$Y \ [\mathrm{m}]$')
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', bbox_to_anchor=(0.95, 1))
    plt.tight_layout()
    plt.show()



def plot_trajectory_keyframes(all_data, T_END=35, DT_FOOTPRINT=5, cols=4):
    items_to_plot = list(all_data.items())
    if not items_to_plot: return

    checkpoint_times = np.arange(0, T_END + 1e-6, DT_FOOTPRINT)
    num_frames = len(checkpoint_times)
    rows = int(np.ceil(num_frames / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 7 * rows), squeeze=False)
    fig.suptitle(r'$\mathrm{Robot\ Trajectory\ Evolution}$', fontsize=16, y=0.98)

    colors_robot = plt.cm.get_cmap('tab10').colors
    axs_flat = axs.flatten()
    
    # Mappatura colori specifici per gli ostacoli
    # Assicurati che le chiavi corrispondano esattamente ai nomi nei tuoi dati
    obs_color_map = {
        'Person': 'skyblue',
        'Rover': 'lightgray',
        'Cardboard box': 'khaki'
    }

    legend_dict = {}

    for j, t_val in enumerate(checkpoint_times):
        ax = axs_flat[j]
        
        # --- 1. Plot Robot Paths ---
        for i, (file_name, data) in enumerate(items_to_plot):
            robot_color = colors_robot[i % 10]
            df_r = data['robot_path']
            
            if df_r is not None and not df_r.empty:
                times = df_r['time'].to_numpy()
                idx_end = (np.abs(times - t_val)).argmin()
                df_curr = df_r.iloc[:idx_end + 1]
                
                if not df_curr.empty:
                    xr, yr = df_curr['x'].to_numpy(), df_curr['y'].to_numpy()
                    line, = ax.plot(xr, yr, color=robot_color, linewidth=2.5, alpha=0.8, zorder=4)
                    
                    curr_pos = df_curr.iloc[-1]
                    label_name = rf'$\mathrm{{{legend_mapping[file_name]}}}$'
                    
                    ell = patches.Ellipse(
                        (curr_pos['x'], curr_pos['y']), 
                        width=2*ROBOT_SEMI_AXIS_A, height=2*ROBOT_SEMI_AXIS_B,
                        angle=curr_pos['yaw'], 
                        color=robot_color, fill=True, alpha=0.5
                    )
                    ax.add_patch(ell)

                    if label_name not in legend_dict:
                        legend_dict[label_name] = line

        # --- 2. Plot Ostacoli Statici con colori specifici ---
        first_bag_data = items_to_plot[0][1]
        obstacles_dict = first_bag_data.get('obstacles', {})

        for obs_label, pos in obstacles_dict.items():
            x_o, y_o = pos['x'], pos['y']
            radius = radius_mapping.get(obs_label, 0.3)
            
            # Recupero colore specifico o default se non trovato
            obs_color = obs_color_map.get(obs_label, 'bisque')
            
            # Disegno ostacolo: un unico oggetto patch per gestire bene la legenda
            # Usiamo alpha=0.3 per il riempimento ma bordo solido
            circle = plt.Circle(
                (x_o, y_o), radius, 
                facecolor=obs_color, 
                edgecolor=obs_color, 
                linewidth=1.5, 
                alpha=0.8, # Alpha globale per la patch nel grafico
                zorder=2
            )
            ax.add_patch(circle)
            
            # --- Gestione Legenda per Ostacoli (Patch + Bordo) ---
            obs_key = rf'${obs_label}$'
            if obs_key not in legend_dict:
                # Creiamo un "Proxy Artist" per la legenda che mostri sia il colore che il bordo
                legend_dict[obs_key] = patches.Patch(
                    facecolor=obs_color, 
                    edgecolor=obs_color, 
                    alpha=0.6, # Leggermente più opaco per essere visibile in legenda
                    label=obs_key
                )

        # --- 3. Formattazione Subplot ---
        ax.set_xlabel(r'$X\ [\mathrm{m}]$', fontsize=12)
        ax.set_ylabel(r'$Y\ [\mathrm{m}]$', fontsize=12)
        ax.set_title(rf'$\mathbf{{t = {t_val:.1f}\ s}}$', fontsize=12, y=-0.5)
        
        ax.set_aspect('equal')
        ax.set_xlim([-1.5, 11]) 
        ax.set_ylim([-4, 4])
        ax.grid(True, linestyle='--', alpha=0.5)

    for k in range(num_frames, len(axs_flat)):
        axs_flat[k].axis('off')

    # --- 4. Legenda Globale in Basso ---
    fig.legend(legend_dict.values(), legend_dict.keys(), 
               loc='lower center', 
               ncol=len(legend_dict), 
               fontsize='large', 
               frameon=True, 
               bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.subplots_adjust(hspace=0.45, wspace=0.3) 
    
    plt.show()

def plot_velocities(all_data):
    num_bags = len(all_data)
    fig, axs = plt.subplots(num_bags, 2, figsize=(12, 4 * num_bags), squeeze=False)
    fig.suptitle('Robot Velocities', fontsize=16, family='serif')

    for i, (file_name, data) in enumerate(all_data.items()):
        df_vel = data['velocity']
        if df_vel.empty: continue
        
        t = df_vel['time'].to_numpy()
        lin = df_vel['linear'].to_numpy()
        ang = df_vel['angular'].to_numpy()

        label_name = legend_mapping[file_name]

        # Subplot Velocità Lineare
        axs[i, 0].plot(t, lin, color='blue')
        axs[i, 0].set_ylabel(rf'$\mathrm{{{label_name}}}$' + '\n' + r'$v\ [\mathrm{m/s}]$')
        axs[i, 0].grid(True, alpha=0.3)
        axs[i, 0].set_ylim([-0.1, 0.6])
        axs[i, 0].set_xlim([0, 40])

        # Subplot Velocità Angolare
        axs[i, 1].plot(t, ang, color='red')
        axs[i, 1].set_ylabel(rf'$\mathrm{{{label_name}}}$' + '\n' + r'$\omega\ [\mathrm{rad/s}]$')
        axs[i, 1].grid(True, alpha=0.3)
        axs[i, 1].set_ylim([-2, 2])
        axs[i, 1].set_xlim([0, 40])

    axs[-1, 0].set_xlabel('Time [s]')
    axs[-1, 1].set_xlabel('Time [s]')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_combined_velocities(all_data):
    # Creiamo 2 subplot sovrapposti (2 righe, 1 colonna)
    fig, (ax_lin, ax_ang) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(r'$\mathrm{Robot\ Velocities}$', fontsize=16)

    # Ciclo sui dati per aggiungere ogni linea ai due subplot
    for file_name, data in all_data.items():
        df_vel = data['velocity']
        if df_vel.empty: continue
        
        t = df_vel['time'].to_numpy()
        lin = df_vel['linear'].to_numpy()
        ang = df_vel['angular'].to_numpy()
        
        label_name = legend_mapping[file_name]

        # Plot Lineare
        ax_lin.plot(t, lin, label=rf'${label_name}$')
        # Plot Angolare
        ax_ang.plot(t, ang, label=rf'${label_name}$')

    # Formattazione Subplot Lineare
    ax_lin.set_ylabel(r'$v\ [\mathrm{m/s}]$', fontsize=12)
    ax_lin.grid(True, linestyle='--', alpha=0.6)
    ax_lin.legend(loc='upper right', fontsize='small', frameon=True)
    ax_lin.set_xlim([0,30])
    ax_lin.set_ylim([-0.1, 1])

    # Formattazione Subplot Angolare
    ax_ang.set_ylabel(r'$\omega\ [\mathrm{rad/s}]$', fontsize=12)
    ax_ang.set_xlabel(r'$t\ [\mathrm{s}]$', fontsize=12)
    ax_ang.grid(True, linestyle='--', alpha=0.6)
    ax_ang.set_xlim([0, 30])
    ax_ang.set_ylim([-2, 2])
    # Se la legenda è identica, possiamo metterla solo nel primo o in entrambi
    ax_ang.legend(loc='upper right', fontsize='small', frameon=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_distances_by_obstacle(all_data):
    sample_bag = next(iter(all_data.values()))
    obstacle_labels = list(sample_bag['obstacles'].keys())
    
    # Ridotto a 3 per coerenza con le dimensioni precedenti
    fig, axs = plt.subplots(len(obstacle_labels), 1, figsize=(10, 3 * len(obstacle_labels)), squeeze=False)
    fig.suptitle(r'$\mathrm{Obstacles\ distances}$', fontsize=18)

    summary_min_distances = {obs: {} for obs in obstacle_labels}

    for j, obs_label in enumerate(obstacle_labels):
        ax = axs[j, 0]
        
        for file_name, data in all_data.items():
            df_path = data['robot_path']
            obstacles = data['obstacles']
            
            if df_path.empty or obs_label not in obstacles:
                continue

            # Dati robot
            t = df_path['time'].to_numpy()
            rx, ry = df_path['x'].to_numpy(), df_path['y'].to_numpy()
            # Assumiamo yaw in gradi (come indicato precedentemente), lo convertiamo in radianti per le funzioni trig
            ryaw = np.radians(df_path['yaw'].to_numpy()) 

            # Dati ostacolo
            obs_pos = obstacles[obs_label]
            radius_obs = radius_mapping[obs_label]

            # 1. Vettore relativo (Centro Robot -> Centro Ostacolo)
            dx = obs_pos['x'] - rx
            dy = obs_pos['y'] - ry
            dist_centers = np.sqrt(dx**2 + dy**2)

            # 2. Angolo dell'ostacolo nel sistema globale
            phi = np.arctan2(dy, dx)

            # 3. Angolo dell'ostacolo relativo al frame del robot
            alpha = phi - ryaw

            # 4. Calcolo raggio variabile dell'ellisse (Robot) in direzione alpha
            a = ROBOT_SEMI_AXIS_A
            b = ROBOT_SEMI_AXIS_B
            r_robot_alpha = (a * b) / np.sqrt((b * np.cos(alpha))**2 + (a * np.sin(alpha))**2)

            # 5. Distanza effettiva: Centro-Centro - Raggio_Ostacolo - Raggio_Robot(alpha)
            dist_effective = dist_centers - radius_obs - r_robot_alpha
            
            summary_min_distances[obs_label][file_name] = round(np.min(dist_effective), 3)
            label_name = legend_mapping[file_name]
            
            ax.plot(t, dist_effective, label=rf'$\mathrm{{{label_name}}}$')

        # Formattazione
        clean_obs_label = obs_label.replace('_', r'\_')
        ax.set_title(rf'${clean_obs_label}$', fontsize=14)
        ax.set_ylabel(r'$d\ [\mathrm{m}]$', fontsize=12)
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 8) # Ridotto a 5m per vedere meglio il dettaglio vicino agli ostacoli
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', fontsize='small', ncol=1)

    axs[-1, 0].set_xlabel(r'$t\ [\mathrm{s}]$', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.subplots_adjust(hspace=0.4)
    plt.show()

    # Riepilogo a terminale
    print("\n--- SUMMARY: MINIMUM EFFECTIVE DISTANCES (Edge-to-Edge) ---")
    for obs, experiments in summary_min_distances.items():
        print(f"\nObstacle: {obs}")
        for bag, d_min in experiments.items():
            print(f"  - {bag}: {d_min} m")

def plot_fear_comparison(all_data):
    # Saltiamo il primo elemento come richiesto
    items_to_plot = list(all_data.items())[1:]
    num_bags = len(items_to_plot)
    
    if num_bags == 0: return

    # Usiamo il numero corretto di bags per non avere subplot vuoti
    fig, axs = plt.subplots(num_bags, 1, figsize=(12, 5 * num_bags), squeeze=False)
    fig.suptitle(r'$\mathrm{Fear\ Levels}$', fontsize=18)

    for i, (file_name, data) in enumerate(items_to_plot):
        ax = axs[i, 0]
        fear_data = data['fear_levels']
        
        if not fear_data:
            ax.text(0.5, 0.5, "No fear data found", ha='center')
            continue

        for label, df in fear_data.items():
            if not df.empty:
                ax.plot(df['time'].to_numpy(), df['value'].to_numpy(), 
                        label=rf'${label}$', linewidth=2)
                
        label_name = legend_mapping[file_name]
        ax.set_title(rf'${label_name}$', loc='center', fontsize=14)
        ax.set_ylabel(rf'$F_t$')
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, 40)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='upper right')

    axs[-1, 0].set_xlabel(r'$t\ [\mathrm{s}]$')

    # --- AGGIUNTA DELLO SPAZIO ---
    # hspace definisce lo spazio verticale tra i subplot (0.1 è il default, prova 0.3 o 0.4)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.subplots_adjust(hspace=0.4) 
    
    plt.show()

def plot_average_inference_times(all_data):
    """
    Crea un grafico a barre che confronta i tempi medi di esecuzione (inferenza)
    di ciascun algoritmo, includendo le barre di errore (deviazione standard).
    """
    labels = []
    means = []
    stds = []
    
    # Estraiamo i dati dal dizionario all_data
    for file_name, data in all_data.items():
        df_mpc = data.get('mpc_stats')
        
        if df_mpc is not None and not df_mpc.empty:
            # Convertiamo in millisecondi per una migliore leggibilità (assumendo siano in secondi)
            exec_times_ms = df_mpc['execution_time'].to_numpy() * 1000 
            
            means.append(np.mean(exec_times_ms))
            stds.append(np.std(exec_times_ms))
            labels.append(rf'$\mathrm{{{legend_mapping.get(file_name, file_name)}}}$')
        else:
            print(f"Avviso: Dati MPC non trovati per {file_name}")

    if not means:
        print("Nessun dato di statistica MPC disponibile per il plot.")
        return

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Usiamo una palette di colori coerente
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    
    bars = ax.bar(labels, means, yerr=stds, capsize=10, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Aggiungiamo i valori numerici sopra le barre per precisione
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f} ms',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 12),  # offset verticale di 12 punti
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Formattazione in stile LaTeX
    ax.set_title(r'$\mathrm{MPC\ Average\ Inference\ Time\ Comparison}$', fontsize=16)
    ax.set_ylabel(r'$\mathrm{Execution\ Time\ [ms]}$', fontsize=12)
    ax.set_xlabel(r'$\mathrm{Algorithm}$', fontsize=12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    # Ottimizzazione limiti asse Y per fare spazio alle annotazioni
    ax.set_ylim(0, 100) 
    
    plt.tight_layout()
    plt.show()

    # Riepilogo testuale a terminale
    print("\n--- INFERENCE TIME SUMMARY (ms) ---")
    for i, label in enumerate(labels):
        print(f"{label}: Mean = {means[i]:.3f} ms | Std = {stds[i]:.3f} ms")

def plot_fear_riemann_integration(all_data):
    """
    Calcola l'integrale del Fear Level usando la Somma di Riemann (Rettangoli).
    Versione compatibile con vecchie versioni di Matplotlib.
    """
    # Struttura: { ostacolo: { algoritmo: area_totale } }
    integration_results = defaultdict(dict)
    
    # 1. Calcolo dell'integrale
    for file_name, data in all_data.items():
        fear_data = data.get('fear_levels', {})
        bag_label = legend_mapping.get(file_name, file_name)
        
        for obs_label, df in fear_data.items():
            if not df.empty and len(df) > 1:
                times = df['time'].to_numpy()
                values = df['value'].to_numpy()
                
                # dt = t_{i+1} - t_i
                dts = np.diff(times)

                # Somma di Riemann (Rettangolo sinistro): sum( f[i] * dt[i] )
                riemann_sum = np.sum(values[:-1] * dts)
                integration_results[obs_label][bag_label] = riemann_sum

    if not integration_results:
        print("Dati insufficienti per il calcolo della Somma di Riemann.")
        return

    # 2. Setup del Plot
    obstacles = list(integration_results.keys())
    # Estraiamo tutti gli algoritmi unici effettivamente presenti nei risultati
    all_algs = set()
    for obs_res in integration_results.values():
        all_algs.update(obs_res.keys())
    
    # Ordiniamo gli algoritmi in base alla tua lista originale per coerenza
    algorithms = [legend_mapping[b] for b in bag_files_list if legend_mapping.get(b) in all_algs]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(obstacles))
    width = 0.15 
    cmap = plt.get_cmap('tab10')

    for i, alg in enumerate(algorithms):
        vals = [integration_results[obs].get(alg, 0.0) for obs in obstacles]
        offset = i * width - (len(algorithms) * width) / 2 + width/2
        
        # Creazione delle barre
        bars = ax.bar(x + offset, vals, width, label=rf'${alg}$', color=cmap(i), edgecolor='black', alpha=0.8)
        
        # --- FIX: Sostituzione di ax.bar_label con un ciclo manuale ---
        for bar in bars:
            height = bar.get_height()
            if height > 0: # Evitiamo di scrivere 0.00 su barre inesistenti
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    # 3. Formattazione
    ax.set_title(r'$\mathrm{Cumulative\ Fear\ Level\ (Riemann\ Sum)}$', fontsize=16)
    ax.set_ylabel(r'$\mathrm{Accumulated\ Fear}$', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([rf'$\mathrm{{{o}}}$' for o in obstacles])
    
    ax.legend(title=r'$\mathrm{Algorithms}$', loc='upper right', frameon=True)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # Aggiustiamo il limite Y per non far tagliare le scritte sopra le barre
    current_ylim = ax.get_ylim()
    ax.set_ylim(0, current_ylim[1] * 1.1)

    plt.tight_layout()
    plt.show()

    # Output testuale per verifica
    print("\n--- RIEMANN INTEGRATION SUMMARY (Per Obstacle) ---")
    total_fear_by_alg = defaultdict(float)

    for obs, algs in integration_results.items():
        print(f"Obstacle: {obs}")
        for alg, val in algs.items():
            print(f"  > {alg}: {val:.4f}")
            total_fear_by_alg[alg] += val

    print("\n--- TOTAL CUMULATIVE FEAR PER ALGORITHM ---")
    for alg in algorithms:
        if alg in total_fear_by_alg:
            print(f"  >> Total {alg}: {total_fear_by_alg[alg]:.4f}")
    



def plot_weighted_fear_risk(all_data):
    """
    Calcola l'Adaptiveness Index normalizzato:
    Index = 1 - [ sum_obj( sum_i( HR_i * LR_i ) ) / R_max ]
    dove R_max è il numero totale di istanti temporali (sommatoria di 1).
    """
    # 1. Definizione pesi High Road (HR) fissi per oggetto
    hr_weights = {
        "Person": 0.6,
        "Rover": 0.5,
        "Cardboard box": 0.2,
    }

    # Risultati finali per algoritmo
    final_indices = {}
    # Per il log testuale
    stats_summary = []

    for file_name, data in all_data.items():
        lr_data_list = data.get('low_road_risk', [])
        bag_label = legend_mapping.get(file_name, file_name)
        
        if not lr_data_list:
            continue

        total_weighted_risk = 0.0
        # R_max è la sommatoria di 1 per ogni istante di tempo registrato
        r_max = float(len(lr_data_list))
        
        for entry in lr_data_list:
            risks = entry['risks']
            for obs_name, lr_val in risks.items():
                clean_label = model_mapping.get(obs_name, obs_name)
                hr_val = hr_weights.get(clean_label, 1.0)
                
                # Sommatoria HR * LR
                total_weighted_risk += (hr_val * lr_val)

        # 2. Calcolo Formula: 1 - (Rischio_Totale / R_max)
        # Assicuriamoci che r_max > 0 per evitare divisioni per zero
        if r_max > 0:
            index_val = 1.0 - (total_weighted_risk / r_max)
            # Clipping di sicurezza nel range [0, 1]
            index_val = max(0.0, min(1.0, index_val))
        else:
            index_val = 0.0

        final_indices[bag_label] = index_val
        stats_summary.append({
            'alg': bag_label,
            'risk': total_weighted_risk,
            'r_max': r_max,
            'index': index_val
        })

    if not final_indices:
        print("Nessun dato disponibile per calcolare l'indice.")
        return

    # --- 3. Plotting ---
    algorithms = [legend_mapping[b] for b in bag_files_list if legend_mapping.get(b) in final_indices]
    vals = [final_indices[alg] for alg in algorithms]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap('tab10')
    
    bars = ax.bar(algorithms, vals, color=[cmap(i) for i in range(len(algorithms))], 
                  edgecolor='black', alpha=0.8, width=0.5)
    
    # Etichette sopra le barre
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_title(r'$\mathbf{Adaptiveness\ Index:\ } 1 - \frac{\sum_{j} \sum_{i} (HR_j \cdot LR_{j,i})}{R_{max}}$', fontsize=16)
    ax.set_ylabel(r'$\mathrm{Score\ [0-1]}$', fontsize=12)
    ax.set_ylim(0, 1.1) # Range fisso 0-1 con un po' di margine sopra
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()

    # --- 4. Riepilogo per Overleaf ---
    print("\n" + "="*60)
    print(f"{'ALGORITHM':<15} | {'SUM(HR*LR)':<12} | {'R_max (ticks)':<12} | {'INDEX':<10}")
    print("-" * 60)
    for s in stats_summary:
        print(f"{s['alg']:<15} | {s['risk']:<12.2f} | {s['r_max']:<12.0f} | {s['index']:<10.4f}")
    print("="*60)

if __name__ == "__main__":
    all_experiments_results = {}

    for file_name in bag_files_list:
        complete_path = os.path.join(script_dir, file_name)
        if os.path.exists(complete_path):
            data = extract_data_from_bag(complete_path, model_mapping)
            if data: all_experiments_results[file_name] = data

    if all_experiments_results:
        # plot_multi_trajectory(all_experiments_results)
        
        # plot_trajectory_keyframes(all_data=all_experiments_results)
        
        # plot_velocities(all_experiments_results)        
        
        # plot_combined_velocities(all_experiments_results)

        # plot_distances_by_obstacle(all_experiments_results)
        
        # plot_fear_comparison(all_experiments_results)

        # # plot_fear_phase_space(all_experiments_results)

        # plot_average_inference_times(all_experiments_results)

        plot_fear_riemann_integration(all_experiments_results)
        plot_weighted_fear_risk(all_experiments_results)