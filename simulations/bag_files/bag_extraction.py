#!/usr/bin/env python3

import rosbag
import numpy as np
import pandas as pd
import os
from math import pi, cos, sin, atan2, sqrt

# Configurazione
BAG_FOLDER = './'
ALGO_NAMES = sorted(['APF', 'MPC', 'MPC_dp'], key=len, reverse=True)
A, B = 0.8, 0.4 # Semi-assi robot
GOAL = np.array([9.0, 9.0])
OBS_RADII = {"Cluttering": 0.7, "operator": 0.3,}

def get_yaw(q):
    return atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y**2 + q.z**2))

def extract():
    raw_results = []
    files = [f for f in os.listdir(BAG_FOLDER) if f.endswith('.bag')]
    
    for filename in files:
        algo = next((name for name in ALGO_NAMES if filename.startswith(name)), "Unknown")
        if algo == "Unknown": continue
        
        print(f"Reading: {filename}...")
        bag = rosbag.Bag(os.path.join(BAG_FOLDER, filename))
        run_id = filename.replace(".bag", "")

        last_recorded_t = 0
        sampling_interval = 0.1  # Estrae un dato ogni 0.1 secondi (10Hz)
        
        for topic, msg, t in bag.read_messages(topics=['/gazebo/model_states', '/odom']):
            curr_t = t.to_sec()
            
            if topic == '/gazebo/model_states':
                # DOWN_SAMPLING: Salta il messaggio se non è passato abbastanza tempo
                if curr_t - last_recorded_t < sampling_interval:
                    continue

                try:
                    idx = msg.name.index("mir")
                    pos = msg.pose[idx].position
                    yaw = get_yaw(msg.pose[idx].orientation)
                    
                    # Calcolo distanza minima analitica da tutti gli ostacoli
                    min_d = float('inf')
                    for i, name in enumerate(msg.name):
                        if name in ["walls", "ground_plane", "mir"]: continue
                        
                        dx, dy = msg.pose[i].position.x - pos.x, msg.pose[i].position.y - pos.y
                        d_centers = sqrt(dx**2 + dy**2)
                        alpha = atan2(dy, dx) - yaw
                        r_robot = (A*B) / sqrt((B*cos(alpha))**2 + (A*sin(alpha))**2)
                        r_obs = next((r for k, r in OBS_RADII.items() if k in name), 0.3)
                        
                        d_edge = d_centers - r_obs - r_robot
                        if d_edge < min_d: min_d = d_edge
                    
                    raw_results.append([algo, run_id, curr_t, pos.x, pos.y, yaw, min_d])
                    last_recorded_t = curr_t

                except ValueError: continue
        bag.close()

    df = pd.DataFrame(raw_results, columns=['algo', 'run', 't', 'x', 'y', 'yaw', 'min_dist'])
    df.to_csv('raw_navigation_data.csv', index=False)
    print("Estrazione completata: raw_navigation_data.csv creato.")

if __name__ == "__main__":
    extract()