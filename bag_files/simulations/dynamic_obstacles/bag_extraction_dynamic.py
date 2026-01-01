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
    "MPC_hr_dynamic.bag",
    "MPC_dp_dynamic.bag",
]

# Parametri footprint / plotting (adattati dal primo script)
ROBOT_SEMI_AXIS_A   = 0.8        # semiasse lungo (m) per ellisse robot
ROBOT_SEMI_AXIS_B   = 0.4        # semiasse corto (m)
DT_FOOTPRINT        = 5.0        # ogni quanti secondi disegnare la sagoma
VELOCITY_THRESHOLD  = 0.05       # soglia per considerare in movimento (m/s)

model_mapping = {
    "mir"       : "MiR",
    "cylinder"  : "Cylinder",
    "rover"     : "Rover",
    "person"    : "Person",
    "cardboard_box" : "Cardboard box"
}

# radius_mapping adattata dal primo script (includo MiR per sicurezza)
radius_mapping = {
    "MiR"       : 0.5,
    "Cylinder"  : 0.3,
    "Rover"     : 0.6,
    "Person"    : 0.3,
    "Cardboard box" : 0.3
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
    raw = {'robot': [], 'obstacles': defaultdict(list), 'vel': [], 'fear': [], 'radial_vel': defaultdict(list)}
    try:
        with rosbag.Bag(bag_file_path, 'r') as bag:
            bag_start = None
            for topic, msg, t in bag.read_messages(topics=['/gazebo/model_states', '/cmd_vel', '/thalamus/info', '/fearlevel']):
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
                    except Exception as e:
                        print(e)
                        continue

                elif topic == '/thalamus/info':
                    try:
                        info_dict = json.loads(msg.data)
                        rel_info = info_dict.get("relative_info", {})
                        for raw_name, stats in rel_info.items():
                            label = model_map.get(raw_name, raw_name)
                            # Filtriamo solo gli ostacoli di interesse
                            if label in ["Cylinder", "Rover", "Person", "Cardboard box"]:
                                v_rad = stats.get("radial_vel", 0.0)
                                raw['radial_vel'][label].append({'t_raw': rel_t, 'value': v_rad})
                    except Exception: continue

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

    # Sincronizzazione Radial Velocity
    radial_vel_df = {}
    for label, arr in raw['radial_vel'].items():
        df_rv = pd.DataFrame(arr)
        if not df_rv.empty:
            df_rv = df_rv[df_rv['t_raw'] >= t_start].copy()
            df_rv['time'] = df_rv['t_raw'] - t_start
            radial_vel_df[label] = df_rv.reset_index(drop=True)

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
        'fear_levels': fear_levels_df,
        'radial_velocities': radial_vel_df
    }

def plot_multi_trajectory(all_data):
    fig, ax = plt.subplots(figsize=(14, 10))
    # Usiamo una palette con colori più saturi (Set1 o Dark2)
    colors = plt.cm.get_cmap('tab10').colors 
    
    obstacles_plotted = False
    max_total_time = max([df['robot_path']['time'].max() for df in all_data.values() if not df['robot_path'].empty])
    checkpoint_times = np.arange(0, max_total_time + 1e-6, DT_FOOTPRINT)

    for i, (file_name, data) in enumerate(all_data.items()):
        # Colore Robot: pieno e saturo
        robot_color = colors[i % len(colors)]
        
        # --- 1. Plot Robot ---
        df_r = data['robot_path']
        if not df_r.empty:
            xr, yr, yrw, tr = df_r['x'].to_numpy(), df_r['y'].to_numpy(), df_r['yaw'].to_numpy(), df_r['time'].to_numpy()
            
            # Linea robot più spessa (linewidth=3) e opaca (alpha=1.0)
            ax.plot(xr, yr, color=robot_color, label=f"Robot: {file_name}", zorder=10)

            for t_val in checkpoint_times:
                if t_val > tr.max(): break
                idx = (np.abs(tr - t_val)).argmin()
                # Sagoma robot: alpha aumentato a 0.6 e linea più spessa
                ell = patches.Ellipse((xr[idx], yr[idx]), width=2*ROBOT_SEMI_AXIS_A, height=2*ROBOT_SEMI_AXIS_B,
                                      angle=yrw[idx], color=robot_color, fill=False, 
                                      linestyle='--', linewidth=1, alpha=0.6)
                ax.add_patch(ell)
                ax.text(xr[idx], yr[idx] + 0.35, f"t={t_val:.1f}", color=robot_color, 
                        fontsize=9, ha='center', fontweight='bold', 
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1)) # Sfondo per leggibilità

        # --- 2. Plot Ostacoli (solo una volta) ---
        if not obstacles_plotted:
            for obs_idx, (label, df_o) in enumerate(data['obstacle_paths'].items()):
                # Colore ostacoli: prendiamo colori dalla fine della palette per distanziarli dai robot
                obs_col = colors[-(obs_idx + 1) % len(colors)] 
                
                if df_o is None or df_o.empty: continue
                xo, yo, to = df_o['x'].to_numpy(), df_o['y'].to_numpy(), df_o['time'].to_numpy()
                radius = radius_mapping.get(label, 0.3)
                
                # Traiettoria ostacolo più visibile (alpha=0.6)
                ax.plot(xo, yo, color=obs_col, linestyle=':', linewidth=2.0, alpha=0.6, label=f"Obstacle: {label}")

                for t_val in checkpoint_times:
                    if t_val > to.max(): break
                    idx = (np.abs(to - t_val)).argmin()
                    circ = plt.Circle((xo[idx], yo[idx]), radius, color=obs_col, 
                                      fill=False, linestyle='-', linewidth=2.0, alpha=0.6)
                    ax.add_patch(circ)
                    ax.text(xo[idx], yo[idx] - (radius + 0.4), f"t={t_val:.1f}", 
                            color=obs_col, fontsize=8, ha='center', style='italic', fontweight='bold')
            
            obstacles_plotted = True

    ax.set_title("Robot Trajectories vs Dynamic Obstacles", fontsize=16, fontweight='bold')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.set_xlim([-1, 12]); ax.set_ylim([-5, 5])
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6) # Grid più visibile
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True, shadow=True)

    plt.tight_layout()
    plt.show()

def plot_velocities(all_data):
    """
    Plotta le velocità (linear e angular) come nel primo script, mantenendo l'ordine dei bag.
    """
    num_bags = len(all_data)
    fig, axs = plt.subplots(num_bags, 2, figsize=(8, 8 * num_bags), squeeze=False)
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
        axs[i, 0].grid(True, alpha=0.3)
        axs[i, 1].plot(t, ang, color='red')
        axs[i, 1].set_ylabel('Ang Vel [rad/s]')
        axs[i, 1].set_xlabel('Time [s]')
        axs[i, 1].grid(True, alpha=0.3)

    plt.subplots_adjust(hspace=0.6, top=0.95, bottom=0.05)
    plt.show()


def plot_radial_velocities(all_data, skip=10):
    """
    Plotta la velocità radiale campionando i dati ogni 'skip' messaggi.
    """
    num_bags = len(all_data)
    fig, axs = plt.subplots(num_bags, 1, figsize=(10, 5 * num_bags), squeeze=False)
    
    # Palette colori per distinguere gli ostacoli
    colors = plt.cm.get_cmap('tab10').colors

    for i, (file_name, data) in enumerate(all_data.items()):
        ax = axs[i, 0]
        rv_data = data.get('radial_velocities', {})
        
        if not rv_data:
            ax.set_title(f"{file_name} - No Radial Velocity Data")
            continue
            
        for obs_idx, (label, df) in enumerate(rv_data.items()):
            if df is None or df.empty:
                continue
            
            # Applichiamo lo slicing [::skip] per prendere un messaggio ogni 10
            t = df['time'].to_numpy()[::skip]
            v = df['value'].to_numpy()[::skip]
            
            color = colors[obs_idx % len(colors)]
            
            # Plottiamo con marker per rendere visibili i punti campionati
            ax.plot(t, v, label=f'v_rad: {label}', color=color, 
                    linewidth=1.5, marker='o', markersize=3, alpha=0.8)
            
        ax.set_title(f"Radial Velocity (Sampled 1/{skip}) - {file_name}", fontweight='bold')
        ax.set_ylabel('v_radial [m/s]')
        ax.set_xlabel('Time [s]')
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()


def plot_distances(all_data):
    """
    Per ogni bag: distanza robot <-> ogni ostacolo (interpolando posizione ostacolo sui tempi del robot).
    """
    num_bags = len(all_data)
    fig, axs = plt.subplots(num_bags, 1, figsize=(8, 8 * num_bags), squeeze=False)
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
        axs[i, 0].grid(True, alpha=0.3)
    plt.subplots_adjust(hspace=0.6, top=0.95, bottom=0.05)
    plt.show()

def plot_fear_comparison(all_data):
    """
    Plotta i valori di fear per ogni bag (se presenti).
    """
    num_bags = len(all_data)
    fig, axs = plt.subplots(num_bags, 1, figsize=(8, 8 * num_bags), squeeze=False)
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
        axs[i, 0].grid(True, alpha=0.3)
    plt.subplots_adjust(hspace=0.6, top=0.95, bottom=0.05)
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
        plot_radial_velocities(all_experiments_results)
        plot_fear_comparison(all_experiments_results)
    else:
        print("Nessun dato da plottare.")
