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

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))

bag_files_list = [
    "MPC_static.bag", 
    "MPC_lr_static.bag", 
    "MPC_hr_static.bag", 
    "MPC_dp_static.bag",
    "APF_static.bag"
] 

ROBOT_SEMI_AXIS_A = 0.8  # Lunghezza (asse x locale)
ROBOT_SEMI_AXIS_B = 0.4  # Larghezza (asse y locale)

DT_FOOTPRINT = 2.0       # Ogni quanti secondi disegnare la sagoma

# 1. Mapping Nomi: {Nome_nel_bag: Nome_per_il_plot}
model_mapping = {
    "mir"           : "MiR",
    "cylinder"      : "Cylinder", 
    "rover"         : "Rover",
    "person"        : "Person",
    "cardboard_box" : "Cardboard box"
}

# 2. Definizione Raggi: {Nome_per_il_plot: Raggio_in_metri}
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


def extract_data_for_plotting(bag_file_path, model_map):
    data = {
        'robot_path': [], # List of {time, x, y}
        'velocity': [],   # List of {time, linear, angular}
        'obstacles': {},   # Dict of {label: {x, y}}
        'fear_levels': defaultdict(list) # { 'Obstacle_A': [{'time': t, 'val': v}, ...], ... }
        }
    
    print(f"Reading: {os.path.basename(bag_file_path)}")
    try:
        with rosbag.Bag(bag_file_path, 'r') as bag:
            start_time = None
            
            for topic, msg, t in bag.read_messages(topics=['/gazebo/model_states', '/cmd_vel', '/fearlevel']):
                if start_time is None: start_time = t.to_sec()
                rel_time = t.to_sec() - start_time
                
                if topic == '/gazebo/model_states':
                    for i, name in enumerate(msg.name):
                        if name in model_map:
                            label = model_map[name]
                            pose = msg.pose[i]
                            if name == "mir":
                                # Estraiamo posizione e orientamento (Yaw)
                                yaw = quaternion_to_yaw(pose.orientation)
                                data['robot_path'].append({
                                    'time': rel_time, 
                                    'x': pose.position.x, 
                                    'y': pose.position.y,
                                    'yaw': math.degrees(yaw) # Matplotlib usa i gradi
                                })
                            elif label not in data['obstacles']:
                                data['obstacles'][label] = {
                                    'x': pose.position.x, 
                                    'y': pose.position.y
                                }
                
                elif topic == '/cmd_vel':
                    data['velocity'].append({
                        'time': rel_time,
                        'linear': msg.linear.x,
                        'angular': msg.angular.z
                    })

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

    except Exception as e:
        print(f"Error: {e}")
        return None


    # Convert to DataFrames
    data['robot_path'] = pd.DataFrame(data['robot_path'])
    data['velocity'] = pd.DataFrame(data['velocity'])
    for label in data['fear_levels']:
        data['fear_levels'][label] = pd.DataFrame(data['fear_levels'][label])
    return data

def plot_multi_trajectory(all_data):
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = plt.get_cmap('tab10')
    
    # --- Plot Traiettorie Robot ---
    for i, (file_name, data) in enumerate(all_data.items()):
        df_robot = data['robot_path']
        
        # Verifica se il DataFrame ha i dati necessari
        if df_robot is not None and not df_robot.empty and 'x' in df_robot.columns:
            x_vals = df_robot['x'].to_numpy()
            y_vals = df_robot['y'].to_numpy()
            yaw_vals = df_robot['yaw'].to_numpy()
            time_vals = df_robot['time'].to_numpy()
            color = cmap(i % 10)
            
            # Traiettoria
            label_name = legend_mapping[file_name]
            ax.plot(
                x_vals, 
                y_vals, 
                label=rf'${label_name}$',
                color=color, 
                linewidth=2, 
                zorder=2
                )
            
            # Plot Ellissi (Footprints) ogni DT_FOOTPRINT secondi
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

            # Ellisse finale
            final_ellipse = patches.Ellipse(
                (x_vals[-1], y_vals[-1]), 
                width=2*ROBOT_SEMI_AXIS_A, height=2*ROBOT_SEMI_AXIS_B, 
                angle=yaw_vals[-1], 
                color=color, 
                fill=True, 
                alpha=0.2, 
                # label=f'Robot Final {file_name}'
            )
            ax.add_patch(final_ellipse)
            ax.scatter(x_vals[-1], y_vals[-1], color=color, marker='x', s=50)

    # --- Plot Ostacoli (Prendiamo quelli del primo esperimento valido) ---
    first_exp_data = list(all_data.values())[0]
    obstacles_dict = first_exp_data.get('obstacles', {}) # Accedi correttamente alla chiave 'obstacles'

    for label, pos in obstacles_dict.items():
        x, y = pos['x'], pos['y']
        radius = radius_mapping.get(label, 0.0)
        
        # Centro dell'ostacolo
        ax.scatter(x, y, color='black', marker='X', s=100, zorder=5)
        
        # Disegno della circonferenza di sicurezza
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



def plot_trajectory_keyframes(all_data, T_END=30, DT_FOOTSTEP=4, cols=4):
    items_to_plot = list(all_data.items())
    if not items_to_plot: return

    checkpoint_times = np.arange(0, T_END + 1e-6, DT_FOOTSTEP)
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

        # Subplot Velocità Lineare
        axs[i, 0].plot(t, lin, color='blue')
        axs[i, 0].set_ylabel(f'{file_name}\nLin Vel [m/s]')
        axs[i, 0].grid(True, alpha=0.3)

        # Subplot Velocità Angolare
        axs[i, 1].plot(t, ang, color='red')
        axs[i, 1].set_ylabel('Ang Vel [rad/s]')
        axs[i, 1].grid(True, alpha=0.3)

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

        # Deleting the single outlier peak
        if file_name == "MPC_dp_static.bag":
            # 0.09442207 0.09442207
            lin_cleaned = lin.copy()
    
            # Cicliamo dal secondo elemento in poi
            for i in range(1, len(lin_cleaned)):
                val = lin_cleaned[i]
                
                # Check: se il valore è compreso tra 0 (escluso) e 0.5 (escluso)
                if 0 < val < 0.45:
                    # Sostituisci con il valore precedente (che a sua volta 
                    # potrebbe essere già stato "pulito" nei passi precedenti)
                    lin_cleaned[i] = lin_cleaned[i-1]
            
            lin = lin_cleaned
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
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='upper right')

    axs[-1, 0].set_xlabel(r'$t\ [\mathrm{s}]$')

    # --- AGGIUNTA DELLO SPAZIO ---
    # hspace definisce lo spazio verticale tra i subplot (0.1 è il default, prova 0.3 o 0.4)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.subplots_adjust(hspace=0.4) 
    
    plt.show()


if __name__ == "__main__":
    all_experiments_results = {}

    for file_name in bag_files_list:
        path_completo = os.path.join(script_dir, file_name)
        if os.path.exists(path_completo):
            data = extract_data_for_plotting(path_completo, model_mapping)
            if data: all_experiments_results[file_name] = data

    if all_experiments_results:
        # plot_multi_trajectory(all_experiments_results)
        
        plot_trajectory_keyframes(all_data=all_experiments_results)
        
        # plot_velocities(all_experiments_results)        
        
        plot_combined_velocities(all_experiments_results)

        plot_distances_by_obstacle(all_experiments_results)
        
        plot_fear_comparison(all_experiments_results)
        # plot_fear_phase_space(all_experiments_results)