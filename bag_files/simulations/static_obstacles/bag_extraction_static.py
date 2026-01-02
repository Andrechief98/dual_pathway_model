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
            ax.plot(x_vals, y_vals, label=f'Path: {file_name}', color=color, linewidth=2, zorder=2)
            
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
                color=color, fill=True, alpha=0.2, label=f'Robot Final {file_name}'
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

    ax.set_title('Robot Trajectories & Obstacles', fontsize=16)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()
def plot_velocities(all_data):
    num_bags = len(all_data)
    fig, axs = plt.subplots(num_bags, 2, figsize=(12, 4 * num_bags), squeeze=False)
    fig.suptitle('Robot Velocities (cmd_vel) for each Experiment', fontsize=16)

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

def plot_distances(all_data):
    num_bags = len(all_data)
    fig, axs = plt.subplots(num_bags, 1, figsize=(10, 4 * num_bags), squeeze=False)
    fig.suptitle('Distances to Obstacles over Time', fontsize=16)

    for i, (file_name, data) in enumerate(all_data.items()):
        df_path = data['robot_path']
        obstacles = data['obstacles']
        if df_path.empty or not obstacles: continue

        t = df_path['time'].to_numpy()
        rx = df_path['x'].to_numpy()
        ry = df_path['y'].to_numpy()

        for obs_label, obs_pos in obstacles.items():
            # Calcolo distanza euclidea istantanea
            dist = np.sqrt((rx - obs_pos['x'])**2 + (ry - obs_pos['y'])**2)
            
            # Soglia di sicurezza (opzionale: raggio robot + raggio ostacolo)
            # safety_limit = radius_mapping.get("MiR", 0) + radius_mapping.get(obs_label, 0)
            
            line, = axs[i, 0].plot(t, dist, label=f'Dist to {obs_label}')
            # axs[i, 0].axhline(y=safety_limit, color=line.get_color(), linestyle='--', alpha=0.3)
            axs[i, 0].set_xlim(0, 30)
            axs[i, 0].set_ylim(0, 10)

        # axs[i, 0].set_title(f'Experiment: {file_name}')
        axs[i, 0].set_ylabel(f'{file_name}\n Distance [m]')
        axs[i, 0].legend(loc='upper right')
        axs[i, 0].grid(True, alpha=0.3)

    axs[-1, 0].set_xlabel('Time [s]')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_fear_comparison(all_data):
    num_bags = len(all_data)
    if num_bags == 0: return

    # Creiamo un subplot per ogni bag file
    fig, axs = plt.subplots(num_bags, 1, figsize=(12, 5 * num_bags), squeeze=False)
    fig.suptitle('Fear Levels Analysis across Experiments', fontsize=18, fontweight='bold')

    for i, (file_name, data) in enumerate(all_data.items()):
        ax = axs[i, 0]
        fear_data = data['fear_levels']
        
        if not fear_data:
            ax.text(0.5, 0.5, "No fear data found", ha='center')
            continue

        # Plot di ogni ostacolo nel subplot del bag corrente
        for label, df in fear_data.items():
            if not df.empty:
                # FIX: .to_numpy() per evitare problemi di indexing
                ax.plot(df['time'].to_numpy(), df['value'].to_numpy(), 
                        label=f'Fear: {label}', linewidth=2)

        ax.set_title(f'Experiment: {file_name}', loc='left', fontsize=14)
        ax.set_ylabel('Fear Level [0-1]')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='upper right', title="Obstacles")

    axs[-1, 0].set_xlabel('Time [s]')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    all_experiments_results = {}

    for file_name in bag_files_list:
        path_completo = os.path.join(script_dir, file_name)
        if os.path.exists(path_completo):
            data = extract_data_for_plotting(path_completo, model_mapping)
            if data: all_experiments_results[file_name] = data

    if all_experiments_results:
        plot_multi_trajectory(all_experiments_results)
        plot_velocities(all_experiments_results)        
        plot_distances(all_experiments_results)
        plot_fear_comparison(all_experiments_results)