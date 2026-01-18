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
    "APF_dynamic.bag"
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
    "Rover"     : 1.0,
    "Person"    : 0.3,
    "Cardboard box" : 0.3
}

legend_mapping = {
    "MPC_dynamic.bag" : "MPC", 
    "MPC_lr_dynamic.bag": "MPC_{lr}", 
    "MPC_hr_dynamic.bag": "MPC_{hr}", 
    "MPC_dp_dynamic.bag": "MPC_{dp}",
    "APF_dynamic.bag" : "APF"
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
            
            label_name = legend_mapping[file_name]
            # Linea robot più spessa (linewidth=3) e opaca (alpha=1.0)
            ax.plot(
                xr, 
                yr, 
                color=robot_color, 
                label=rf"${label_name}$", 
                zorder=10
                )

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

    for i, (file_name, data) in enumerate(all_data.items()):
        # --- 2. Plot Ostacoli (solo una volta) ---
        if not obstacles_plotted:
            for obs_idx, (label, df_o) in enumerate(data['obstacle_paths'].items()):
                obs_col = colors[-(obs_idx + 1) % len(colors)] 
                
                if df_o is None or df_o.empty: continue
                xo, yo, to = df_o['x'].to_numpy(), df_o['y'].to_numpy(), df_o['time'].to_numpy()
                radius = radius_mapping.get(label, 0.3)
                
                # Traiettoria ostacolo più visibile (alpha=0.6)
                ax.plot(
                    xo, 
                    yo, 
                    color=obs_col, 
                    linestyle=':', 
                    linewidth=2.0, 
                    alpha=0.6, 
                    label=rf"${label}$"
                    )

                for t_val in checkpoint_times:
                    if t_val > to.max(): break
                    idx = (np.abs(to - t_val)).argmin()
                    circ = plt.Circle((xo[idx], yo[idx]), radius, color=obs_col, 
                                      fill=False, linestyle='-', linewidth=2.0, alpha=0.6)
                    ax.add_patch(circ)
                    ax.text(xo[idx], yo[idx] - (radius + 0.4), f"t={t_val:.1f}", 
                            color=obs_col, fontsize=8, ha='center', style='italic', fontweight='bold')
            
            obstacles_plotted = True

    ax.set_title(r'$\mathrm{Robot\ Trajectories\ &\ Obstacles}$', fontsize=16)
    ax.set_xlabel(r'$X \ [\mathrm{m}]$')
    ax.set_ylabel(r'$Y \ [\mathrm{m}]$')
    ax.set_aspect('equal')
    ax.set_ylim([-4,4])
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', bbox_to_anchor=(0.95, 1))
    plt.tight_layout()
    plt.show()




def plot_trajectory_keyframes(all_data, T_END=30, DT_FOOTSTEP=4, cols=4):
    """
    Genera keyframes per ostacoli DINAMICI.
    Colori specifici per Person e Rover, con legenda unificata (patch + bordo).
    """
    items_to_plot = list(all_data.items())
    if not items_to_plot: return

    checkpoint_times = np.arange(0, T_END + 1e-6, DT_FOOTSTEP)
    num_frames = len(checkpoint_times)
    rows = int(np.ceil(num_frames / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 7 * rows), squeeze=False)
    fig.suptitle(r'$\mathrm{Robot\ Trajectory\ Evolution}$', fontsize=16, y=0.98)

    colors_robot = plt.cm.get_cmap('tab10').colors
    axs_flat = axs.flatten()
    
    # Mappatura colori specifica per ostacoli dinamici
    obs_color_map = {
        'Person': 'skyblue',
        'Rover': 'lightgray'
    }

    # Dizionario per la legenda globale
    legend_dict = {}

    for j, t_val in enumerate(checkpoint_times):
        ax = axs_flat[j]
        
        # --- 1. Plot Robot Paths (Traiettorie e Footprint) ---
        for i, (file_name, data) in enumerate(items_to_plot):
            robot_color = colors_robot[i % 10]
            df_r = data['robot_path']
            
            if df_r is not None and not df_r.empty:
                times = df_r['time'].to_numpy()
                idx_end = (np.abs(times - t_val)).argmin()
                df_curr = df_r.iloc[:idx_end + 1]

                if not df_curr.empty:
                    xr, yr = df_curr['x'].to_numpy(), df_curr['y'].to_numpy()
                    # Traiettoria robot
                    line, = ax.plot(xr, yr, color=robot_color, linewidth=2.5, alpha=0.8, zorder=5)
                    
                    # Ellisse robot (posizione attuale)
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

        # --- 2. Plot Ostacoli Dinamici ---
        # Prendiamo i percorsi degli ostacoli dal primo esperimento
        first_exp_data = items_to_plot[0][1]
        for obs_label, df_o in first_exp_data['obstacle_paths'].items():
            if df_o is None or df_o.empty: continue
            
            # Colore specifico
            obs_color = obs_color_map.get(obs_label, 'bisque')
            
            # Logica nearest neighbor per la traiettoria dell'ostacolo
            t_obs = df_o['time'].to_numpy()
            idx_o = (np.abs(t_obs - t_val)).argmin()
            df_o_curr = df_o.iloc[:idx_o + 1]
            
            if not df_o_curr.empty:
                xo, yo = df_o_curr['x'].to_numpy(), df_o_curr['y'].to_numpy()
                radius = radius_mapping.get(obs_label, 0.3)
                
                # Scia dell'ostacolo (linea punteggiata)
                ax.plot(xo, yo, color=obs_color, linestyle=':', linewidth=2, alpha=0.6, zorder=2)
                
                # Posizione attuale ostacolo (Cerchio con bordo)
                curr_o = df_o_curr.iloc[-1]
                circ = plt.Circle(
                    (curr_o['x'], curr_o['y']), radius, 
                    facecolor=obs_color, edgecolor=obs_color, 
                    linewidth=1.5, 
                    alpha=0.8, 
                    zorder=3
                )
                ax.add_patch(circ)

                # Aggiunta alla legenda (Proxy Artist per mostrare patch + bordo)
                obs_key = rf'${obs_label}$'
                if obs_key not in legend_dict:
                    legend_dict[obs_key] = patches.Patch(
                        facecolor=obs_color, edgecolor=obs_color, 
                        alpha=0.6, label=obs_key
                    )

        # --- 3. Formattazione Subplot ---
        ax.set_xlabel(r'$X\ [\mathrm{m}]$', fontsize=12)
        ax.set_ylabel(r'$Y\ [\mathrm{m}]$', fontsize=12)
        ax.set_title(rf'$\mathbf{{t = {t_val:.1f}\ s}}$', fontsize=12, y=-0.5)
        
        ax.set_aspect('equal')
        ax.set_xlim([-1.5, 11]) 
        ax.set_ylim([-4, 4])
        ax.grid(True, linestyle='--', alpha=0.5)

    # Nascondi subplot vuoti
    for k in range(num_frames, len(axs_flat)):
        axs_flat[k].axis('off')

    # --- 4. Legenda Globale ---
    fig.legend(legend_dict.values(), legend_dict.keys(), 
               loc='lower center', 
               ncol=min(len(legend_dict), 7), 
               fontsize='large', 
               frameon=True, 
               bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.subplots_adjust(hspace=0.45, wspace=0.3) 
    
    plt.show()


def plot_velocities(all_data):
    # Saltiamo il primo elemento e prendiamo i restanti
    items_to_plot = list(all_data.items())[1:]
    num_bags = len(items_to_plot)
    
    if num_bags == 0:
        print("Nessun dato disponibile per il plot delle velocità.")
        return

    # Ridotto l'altezza a 4 per rendere i plot meno "giganti" (8 era molto alto)
    fig, axs = plt.subplots(num_bags, 2, figsize=(12, 4 * num_bags), squeeze=False)
    fig.suptitle(r'$\mathrm{Robot\ Velocities}$', fontsize=16)

    for i, (file_name, data) in enumerate(items_to_plot):
        df_vel = data['velocity']
        
        if df_vel is None or df_vel.empty:
            axs[i, 0].text(0.5, 0.5, "No data", ha='center')
            axs[i, 1].text(0.5, 0.5, "No data", ha='center')
            continue
            
        t = df_vel['time'].to_numpy()
        lin = df_vel['linear'].to_numpy()
        ang = df_vel['angular'].to_numpy()

        # Pulizia nome per LaTeX
        clean_name = file_name.replace('_', r'\_')
        label_name = legend_mapping.get(file_name, clean_name)

        # --- Subplot Velocità Lineare (v) ---
        axs[i, 0].plot(t, lin, color='blue', linewidth=1.5)
        # Y-label con nome esperimento e variabile v
        axs[i, 0].set_ylabel(rf'$\mathrm{{{label_name}}}$', fontsize=11)
        axs[i, 0].grid(True, linestyle=':', alpha=0.6)

        # --- Subplot Velocità Angolare (omega) ---
        axs[i, 1].plot(t, ang, color='red', linewidth=1.5)
        axs[i, 1].set_ylabel(r'$\omega\ [\mathrm{rad/s}]$', fontsize=11)
        axs[i, 1].grid(True, linestyle=':', alpha=0.6)

    # Etichette asse X solo sull'ultima riga per pulizia
    axs[-1, 0].set_xlabel(r'$t\ [\mathrm{s}]$', fontsize=12)
    axs[-1, 1].set_xlabel(r'$t\ [\mathrm{s}]$', fontsize=12)

    # Regolazione spazi: hspace per la distanza tra righe, wspace per distanza tra colonne
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    
    plt.show()


def plot_velocities_combined(all_data):

    items_to_plot = list(all_data.items())
    
    if not items_to_plot:
        print("Nessun dato disponibile.")
        return

    # Creiamo solo 2 subplot (uno sopra l'altro)
    fig, (ax_lin, ax_ang) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(r'$\mathrm{Robot\ Velocities}$', fontsize=18)

    for file_name, data in items_to_plot:
        df_vel = data['velocity']
        if df_vel is None or df_vel.empty:
            continue
            
        t = df_vel['time'].to_numpy()
        lin = df_vel['linear'].to_numpy()
        ang = df_vel['angular'].to_numpy()

        # Recupero nome dalla legenda e pulizia per LaTeX
        label_name = legend_mapping[file_name]

        # Plot lineare nel primo subplot
        ax_lin.plot(t, lin, label=rf'$\mathrm{{{label_name}}}$', linewidth=1.5)
        
        # Plot angolare nel secondo subplot
        ax_ang.plot(t, ang, label=rf'$\mathrm{{{label_name}}}$', linewidth=1.5)

    # --- Formattazione Subplot Lineare (v) ---
    ax_lin.set_ylabel(r'$v\ [\mathrm{m/s}]$', fontsize=12)
    ax_lin.grid(True, linestyle=':', alpha=0.6)
    ax_lin.set_ylim([-0.1, 1])
    ax_lin.set_xlim([0, 35])
    # Legenda leggermente ingrandita come richiesto prima
    ax_lin.legend(loc='upper right', fontsize='small', ncol=1)

    # --- Formattazione Subplot Angolare (omega) ---
    ax_ang.set_ylabel(r'$\omega\ [\mathrm{rad/s}]$', fontsize=12)
    ax_ang.set_xlabel(r'$t\ [\mathrm{s}]$', fontsize=12)
    ax_ang.grid(True, linestyle=':', alpha=0.6)
    ax_ang.set_xlim([0, 35])
    ax_ang.set_ylim([-2.2, 2.2])
    ax_ang.legend(loc='upper right', fontsize='small', ncol=1)

    # Spazio tra i due subplot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(hspace=0.3)
    
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

    num_bags = len(all_data)
    fig, axs = plt.subplots(num_bags, 1, figsize=(8, 8 * num_bags), squeeze=False)
    
    summary_min_distances = {}
    
    for i, (file_name, data) in enumerate(all_data.items()):
        df_r = data['robot_path']
        if df_r is None or df_r.empty:
            axs[i,0].set_visible(False)
            continue
        t = df_r['time'].to_numpy()
        rx = df_r['x'].to_numpy()
        ry = df_r['y'].to_numpy()

        all_min_distances_in_bag = {}

        for label, df_o in data['obstacle_paths'].items():
            if df_o is None or df_o.empty:
                continue
            # interpolo le coordinate dell'ostacolo sui tempi t del robot
            ox = np.interp(t, df_o['time'].to_numpy(), df_o['x'].to_numpy())
            oy = np.interp(t, df_o['time'].to_numpy(), df_o['y'].to_numpy())

            
            dist = np.sqrt((rx - ox)**2 + (ry - oy)**2)

            all_min_distances_in_bag[label] = round(np.min(dist),3)

            axs[i, 0].plot(t, dist, label=f'Dist to {label}')
            axs[i, 0].set_xlim(0, 35)
            axs[i, 0].set_ylim(0, 10)
            axs[i, 0].set_ylabel(f'{file_name}\n Distance [m]')

        summary_min_distances[file_name] = all_min_distances_in_bag

        # axs[i, 0].set_ylabel('Distance [m]')
        axs[i, 0].set_xlabel('Time [s]')
        axs[i, 0].legend()
        axs[i, 0].grid(True, alpha=0.3)
    plt.subplots_adjust(hspace=0.6, top=0.95, bottom=0.05)
    plt.show()

    print("\n--- SUMMARY: MINIMUM DISTANCES PER EXPERIMENT ---")
    for bag, val in summary_min_distances.items():
        print(f"{bag}: {val}")
    print("-------------------------------------------------\n")



import numpy as np
import matplotlib.pyplot as plt

def plot_distances_by_obstacles(all_data):
    """
    Plot delle distanze edge-to-edge per ostacoli dinamici.
    Un subplot per ogni ostacolo, con tutti gli algoritmi a confronto.
    """
    # 1. Identifica tutti gli ostacoli unici presenti nei dati
    sample_bag = next(iter(all_data.values()))
    obstacle_labels = list(sample_bag['obstacle_paths'].keys())
    
    fig, axs = plt.subplots(len(obstacle_labels), 1, figsize=(10, 5 * len(obstacle_labels)), squeeze=False)
    fig.suptitle(r'$\mathrm{Obstacles\ distances}$', fontsize=18)

    summary_min_distances = {obs: {} for obs in obstacle_labels}
    colors_algo = plt.cm.get_cmap('tab10').colors

    for j, obs_label in enumerate(obstacle_labels):
        ax = axs[j, 0]
        
        for i, (file_name, data) in enumerate(all_data.items()):
            df_robot = data['robot_path']
            df_obs = data['obstacle_paths'].get(obs_label)
            
            if df_robot is None or df_robot.empty or df_obs is None or df_obs.empty:
                continue

            # Dati Robot
            t_r = df_robot['time'].to_numpy()
            rx, ry = df_robot['x'].to_numpy(), df_robot['y'].to_numpy()
            ryaw = np.radians(df_robot['yaw'].to_numpy()) # Conversione in radianti

            # --- Interpolazione Ostacolo ---
            # Poiché l'ostacolo si muove, dobbiamo sapere dove si trova esattamente ai tempi t_r del robot
            ox = np.interp(t_r, df_obs['time'].to_numpy(), df_obs['x'].to_numpy())
            oy = np.interp(t_r, df_obs['time'].to_numpy(), df_obs['y'].to_numpy())
            
            # --- Calcolo Distanza Edge-to-Edge ---
            # 1. Vettore relativo e distanza tra i centri
            dx, dy = ox - rx, oy - ry
            dist_centers = np.sqrt(dx**2 + dy**2)

            # 2. Angolo dell'ostacolo nel sistema globale
            phi = np.arctan2(dy, dx)

            # 3. Angolo dell'ostacolo relativo all'orientamento del robot (yaw)
            alpha = phi - ryaw

            # 4. Raggio variabile dell'ellisse del robot in direzione alpha
            a, b = ROBOT_SEMI_AXIS_A, ROBOT_SEMI_AXIS_B
            r_robot_alpha = (a * b) / np.sqrt((b * np.cos(alpha))**2 + (a * np.sin(alpha))**2)

            # 5. Raggio dell'ostacolo (assunto circolare)
            radius_obs = radius_mapping.get(obs_label, 0.3)

            # 6. Distanza effettiva margine-margine
            dist_eff = dist_centers - radius_obs - r_robot_alpha
            
            # Salvataggio minimo
            summary_min_distances[obs_label][file_name] = round(np.min(dist_eff), 3)

            # Plot
            label_algo = rf'$\mathrm{{{legend_mapping[file_name]}}}$'
            ax.plot(t_r, dist_eff, label=label_algo, color=colors_algo[i % 10], linewidth=2)

        # Formattazione Subplot
        ax.set_title(rf'$\mathrm{{{obs_label}}}$', fontsize=14)
        ax.set_ylabel(r'$d\ [\mathrm{m}]$', fontsize=12)
        ax.set_xlim(0, 35)
        ax.set_ylim(-1, 10) # Range ottimale per vedere gli avvicinamenti
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', fontsize='small', ncol=1)

    # Asse X finale
    axs[-1, 0].set_xlabel(r'$t\ [\mathrm{s}]$', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.subplots_adjust(hspace=0.4)
    plt.show()

    # Riepilogo Testuale
    print("\n--- SUMMARY: MINIMUM DYNAMIC DISTANCES (Edge-to-Edge) ---")
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
    for f in bag_files_list:
        path = os.path.join(script_dir, f)
        if os.path.exists(path):
            d = extract_data_for_plotting(path, model_mapping)
            if d:
                all_experiments_results[f] = d
        else:
            print(f"File non trovato: {path}")

    if all_experiments_results:
        # plot_multi_trajectory(all_experiments_results)
        plot_trajectory_keyframes(all_data=all_experiments_results)
        # plot_velocities(all_experiments_results)
        plot_velocities_combined(all_experiments_results)
        # plot_distances(all_experiments_results)
        plot_distances_by_obstacles(all_experiments_results)
        # plot_radial_velocities(all_experiments_results)
        plot_fear_comparison(all_experiments_results)
    else:
        print("Nessun dato da plottare.")
