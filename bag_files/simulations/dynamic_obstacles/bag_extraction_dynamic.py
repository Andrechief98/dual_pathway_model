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
    "MPC_dynamic.bag",
    "MPC_lr_dynamic.bag",
    "MPC_dp_dynamic.bag",
]

# Parametri footprint / plotting (adattati dal primo script)
ROBOT_SEMI_AXIS_A = 0.8   # semiasse lungo (m) per ellisse robot
ROBOT_SEMI_AXIS_B = 0.4   # semiasse corto (m)
DT_FOOTPRINT = 5.0        # ogni quanti secondi disegnare la sagoma
VELOCITY_THRESHOLD = 0.05 # soglia per considerare in movimento (m/s)

model_mapping = {
    "mir": "MiR",
    "cylinder": "Cylinder",
    "rover": "Rover",
    "person": "Person"
}

# radius_mapping adattata dal primo script (includo MiR per sicurezza)
radius_mapping = {
    "MiR": 0.5,
    "Cylinder": 0.3,
    "Rover": 1.0,
    "Person": 0.3
}

def quaternion_to_yaw_deg(q):
    """Converte quaternion (geometry_msgs/Quaternion) in yaw in gradi."""
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))

def draw_footprints_circles(ax, df, label, color, dt_step):
    """
    Disegna cerchi lungo la traiettoria ogni dt_step secondi (per ostacoli e, se richiesto, robot).
    df deve avere colonne ['time','x','y'].
    """
    if df is None or df.empty:
        return
    radius = radius_mapping.get(label, 0.3)
    max_time = df['time'].to_numpy().max()
    checkpoint_times = np.arange(0, max_time + 1e-6, dt_step)
    for t_target in checkpoint_times:
        # indice del punto più vicino nel tempo
        idx = (df['time'].to_numpy() - t_target)
        idx = np.abs(idx).argmin()
        row_x = df['x'].to_numpy()[idx]
        row_y = df['y'].to_numpy()[idx]
        circ = plt.Circle((row_x, row_y), radius, color=color, fill=False,
                          linestyle='--', alpha=0.4, linewidth=1)
        ax.add_patch(circ)

def extract_data_for_plotting(bag_file_path, model_map):
    """
    Legge il bag e ritorna i dati sincronizzati rispetto all'inizio del moto rilevato.
    Ritorna dict con: robot_path (DataFrame con columns ['t_raw','time','x','y','yaw']),
                       obstacle_paths (dict label->DataFrame ['t_raw','time','x','y']),
                       velocity (DataFrame ['time','linear','angular']),
                       fear_levels (dict label->DataFrame ['time','value'])
    """
    raw = {'robot': [], 'obstacles': defaultdict(list), 'vel': [], 'fear': []}
    try:
        with rosbag.Bag(bag_file_path, 'r') as bag:
            bag_start = None
            for topic, msg, t in bag.read_messages(topics=['/gazebo/model_states', '/cmd_vel', '/fearlevel']):
                if bag_start is None: bag_start = t.to_sec()
                rel_t = t.to_sec() - bag_start

                if topic == '/gazebo/model_states':
                    for i, name in enumerate(msg.name):
                        if name in model_map:
                            label = model_map[name]
                            pose = msg.pose[i]
                            if name == "mir":
                                yaw = quaternion_to_yaw_deg(pose.orientation)
                                raw['robot'].append({'t_raw': rel_t, 'x': pose.position.x, 'y': pose.position.y, 'yaw': yaw})
                            else:
                                raw['obstacles'][label].append({'t_raw': rel_t, 'x': pose.position.x, 'y': pose.position.y})

                elif topic == '/cmd_vel':
                    # memorizzo sia linear che angular per i plot
                    raw['vel'].append({'t_raw': rel_t, 'linear': getattr(msg.linear, 'x', 0.0), 'angular': getattr(msg.angular, 'z', 0.0)})

                elif topic == '/fearlevel':
                    # /fearlevel con payload JSON nel campo data (come nello script 1)
                    try:
                        fear_dict = json.loads(msg.data)
                        raw['fear'].append({'t_raw': rel_t, 'data': fear_dict})
                    except Exception:
                        # ignora messaggi malformati
                        continue

    except Exception as e:
        print(f"Error reading {bag_file_path}: {e}")
        return None

    # --- Costruzione DataFrame e sincronizzazione sul primo movimento ---
    df_vel = pd.DataFrame(raw['vel'])
    if df_vel.empty or not raw['robot']:
        print(f"No velocity or robot data in {os.path.basename(bag_file_path)}")
        return None

    # trova primo istante in cui linear > soglia
    moving = df_vel[df_vel['linear'] > VELOCITY_THRESHOLD]
    if moving.empty:
        t_start = 0.0
    else:
        t_start = moving.iloc[0]['t_raw']

    # Robot: punti da t_start in poi
    df_r = pd.DataFrame(raw['robot'])
    df_r = df_r[df_r['t_raw'] >= t_start].copy()
    if df_r.empty:
        print(f"No robot points after t_start for {os.path.basename(bag_file_path)}")
        return None
    df_r['time'] = df_r['t_raw'] - t_start

    # Ostacoli: stessa logica (punti dopo t_start)
    final_obstacles = {}
    for label, pts in raw['obstacles'].items():
        df_o = pd.DataFrame(pts)
        df_o = df_o[df_o['t_raw'] >= t_start].copy()
        if df_o.empty:
            final_obstacles[label] = pd.DataFrame(columns=['t_raw','time','x','y'])
            continue
        df_o['time'] = df_o['t_raw'] - t_start
        final_obstacles[label] = df_o

    # Velocità: riallineata a t_start
    df_vel2 = df_vel[df_vel['t_raw'] >= t_start].copy()
    df_vel2['time'] = df_vel2['t_raw'] - t_start
    df_vel2 = df_vel2.reset_index(drop=True)

    # Fearlevels: aggrego per label insieme ai tempi relativi
    fear_levels = defaultdict(list)
    for item in raw['fear']:
        t_rel = item['t_raw'] - t_start
        for raw_name, val in item['data'].items():
            label = model_map.get(raw_name, raw_name)
            fear_levels[label].append({'time': t_rel, 'value': val})
    # convert to DataFrames
    fear_levels_df = {}
    for label, arr in fear_levels.items():
        df_f = pd.DataFrame(arr)
        if not df_f.empty:
            df_f = df_f[df_f['time'] >= 0].copy()
        fear_levels_df[label] = df_f

    print(f"File: {os.path.basename(bag_file_path)} | Movimento rilevato a t={t_start:.2f}s")
    return {
        'robot_path': df_r.reset_index(drop=True),
        'obstacle_paths': final_obstacles,
        'velocity': df_vel2,
        'fear_levels': fear_levels_df
    }

def plot_multi_trajectory(all_data):
    """
    Mantiene la struttura del secondo script: un subplot per bag.
    Integra i footprints del primo (cerchi per ostacoli), ellisse per robot.
    """
    num_bags = len(all_data)
    fig, axs = plt.subplots(num_bags, 1, figsize=(12, 6 * num_bags), squeeze=False)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (file_name, data) in enumerate(all_data.items()):
        ax = axs[i, 0]
        ax.set_title(f"Experiment: {file_name}", fontsize=14, fontweight='bold')

        # Robot
        df_r = data['robot_path']
        if not df_r.empty:
            xr = df_r['x'].to_numpy()
            yr = df_r['y'].to_numpy()
            yrw = df_r['yaw'].to_numpy()
            tr = df_r['time'].to_numpy()
            ax.plot(xr, yr, color=colors[0], label="Robot Path", linewidth=2, zorder=5)

            # Footprints ellittiche ogni DT_FOOTPRINT secondi (usando yaw)
            checkpoint_times = np.arange(0, tr.max() + 1e-6, DT_FOOTPRINT)
            for t_val in checkpoint_times:
                idx = (np.abs(tr - t_val)).argmin()
                curr_x = xr[idx]
                curr_y = yr[idx]
                angle_deg = yrw[idx]
                ell = patches.Ellipse((curr_x, curr_y),
                                      width=2*ROBOT_SEMI_AXIS_A,
                                      height=2*ROBOT_SEMI_AXIS_B,
                                      angle=angle_deg,
                                      color=colors[0],
                                      fill=False,
                                      linestyle='--',
                                      alpha=0.3)
                ax.add_patch(ell)
                ax.text(curr_x, curr_y + 0.25, f"t={t_val:.1f}s", color=colors[0], fontsize=9, ha='center')

            ax.scatter(xr[-1], yr[-1], color=colors[0], marker='x', s=80, zorder=6)

        # Obstacles
        for obs_idx, (label, df_o) in enumerate(data['obstacle_paths'].items()):
            obs_col = colors[(obs_idx + 1) % len(colors)]
            if df_o is None or df_o.empty:
                continue
            xo = df_o['x'].to_numpy()
            yo = df_o['y'].to_numpy()
            to = df_o['time'].to_numpy()
            radius = radius_mapping.get(label, 0.3)
            ax.plot(xo, yo, color=obs_col, linestyle=':', alpha=0.2, label=f"Obstacle: {label}")

            # Disegno cerchi sincronizzati con gli stessi checkpoint del robot
            for t_val in checkpoint_times:
                if t_val > to.max(): break
                idx = (np.abs(to - t_val)).argmin()
                curr_xo = xo[idx]
                curr_yo = yo[idx]
                circ = plt.Circle((curr_xo, curr_yo), radius, color=obs_col, fill=False, linestyle=':', alpha=0.6)
                ax.add_patch(circ)
                ax.text(curr_xo, curr_yo - (radius + 0.25), f"t={t_val:.1f}s", color=obs_col, fontsize=8, ha='center', style='italic')

        ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()

def plot_velocities(all_data):
    """
    Plotta le velocità (linear e angular) come nel primo script, mantenendo l'ordine dei bag.
    """
    num_bags = len(all_data)
    fig, axs = plt.subplots(num_bags, 2, figsize=(12, 3 * num_bags), squeeze=False)
    for i, (file_name, data) in enumerate(all_data.items()):
        df_vel = data['velocity']
        if df_vel is None or df_vel.empty:
            axs[i,0].set_visible(False)
            axs[i,1].set_visible(False)
            continue
        t = df_vel['time'].to_numpy()
        lin = df_vel['linear'].to_numpy()
        ang = df_vel['angular'].to_numpy()

        axs[i, 0].plot(t, lin, color='blue')
        axs[i, 0].set_ylabel(f'{file_name}\nLin Vel [m/s]')
        axs[i, 0].set_xlabel('Time [s]')
        axs[i, 1].plot(t, ang, color='red')
        axs[i, 1].set_ylabel('Ang Vel [rad/s]')
        axs[i, 1].set_xlabel('Time [s]')
        axs[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_distances(all_data):
    """
    Per ogni bag: distanza robot <-> ogni ostacolo (interpolando posizione ostacolo sui tempi del robot).
    """
    num_bags = len(all_data)
    fig, axs = plt.subplots(num_bags, 1, figsize=(12, 3 * num_bags), squeeze=False)
    for i, (file_name, data) in enumerate(all_data.items()):
        df_r = data['robot_path']
        if df_r is None or df_r.empty:
            axs[i,0].set_visible(False)
            continue
        t = df_r['time'].to_numpy()
        rx = df_r['x'].to_numpy()
        ry = df_r['y'].to_numpy()

        for label, df_o in data['obstacle_paths'].items():
            if df_o is None or df_o.empty:
                continue
            # interpolo le coordinate dell'ostacolo sui tempi t del robot
            ox = np.interp(t, df_o['time'].to_numpy(), df_o['x'].to_numpy())
            oy = np.interp(t, df_o['time'].to_numpy(), df_o['y'].to_numpy())
            dist = np.sqrt((rx - ox)**2 + (ry - oy)**2)
            axs[i, 0].plot(t, dist, label=f'Dist to {label}')

        axs[i, 0].set_ylabel('Distance [m]')
        axs[i, 0].set_xlabel('Time [s]')
        axs[i, 0].legend()
    plt.tight_layout()
    plt.show()

def plot_fear_comparison(all_data):
    """
    Plotta i valori di fear per ogni bag (se presenti).
    """
    num_bags = len(all_data)
    fig, axs = plt.subplots(num_bags, 1, figsize=(12, 3 * num_bags), squeeze=False)
    for i, (file_name, data) in enumerate(all_data.items()):
        fear_data = data.get('fear_levels', {})
        if not fear_data:
            axs[i,0].set_visible(False)
            continue
        for label, df in fear_data.items():
            if df is None or df.empty:
                continue
            axs[i,0].plot(df['time'].to_numpy(), df['value'].to_numpy(), label=f'Fear: {label}')
        axs[i,0].set_ylim(-0.05, 1.05)
        axs[i,0].set_xlabel('Time [s]')
        axs[i,0].legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    all_experiments_results = {}
    for f in bag_files_list:
        path = os.path.join(script_dir, f)
        if os.path.exists(path):
            d = extract_data_for_plotting(path, model_mapping)
            if d:
                all_experiments_results[f] = d
        else:
            print(f"File non trovato: {path}")

    if all_experiments_results:
        plot_multi_trajectory(all_experiments_results)
        plot_velocities(all_experiments_results)
        plot_distances(all_experiments_results)
        plot_fear_comparison(all_experiments_results)
    else:
        print("Nessun dato da plottare.")
