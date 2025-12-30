#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import threading

class LiveRiskPlotter:
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.start_time = rospy.get_time()
        
        # Struttura dati: { 'nome_oggetto': {'time': [], 'high': [], 'low': []} }
        self.data_store = {}
        self.lock = threading.Lock()
        
        # Subscriber
        rospy.Subscriber("/amygdala/highroad/risks", String, self.highroad_callback)
        rospy.Subscriber("/amygdala/lowroad/risks", String, self.lowroad_callback)
        
        self.fig = None
        self.axs = {}

    def parse_json(self, msg_data):
        try:
            # Rimuove eventuali escape e converte in dict
            return json.loads(msg_data)
        except json.JSONDecodeError:
            rospy.logwarn("Errore nel parsing JSON")
            return {}

    def highroad_callback(self, msg):
        risks = self.parse_json(msg.data)
        current_time = rospy.get_time() - self.start_time
        with self.lock:
            for obj, value in risks.items():
                self._ensure_obj(obj)
                self.data_store[obj]['high'].append(value)
                self.data_store[obj]['time_h'].append(current_time)
                # Mantieni finestra temporale
                if len(self.data_store[obj]['high']) > self.window_size:
                    self.data_store[obj]['high'].pop(0)
                    self.data_store[obj]['time_h'].pop(0)

    def lowroad_callback(self, msg):
        risks = self.parse_json(msg.data)
        current_time = rospy.get_time() - self.start_time
        with self.lock:
            for obj, value in risks.items():
                self._ensure_obj(obj)
                self.data_store[obj]['low'].append(value)
                self.data_store[obj]['time_l'].append(current_time)
                if len(self.data_store[obj]['low']) > self.window_size:
                    self.data_store[obj]['low'].pop(0)
                    self.data_store[obj]['time_l'].pop(0)

    def _ensure_obj(self, obj):
        if obj not in self.data_store:
            self.data_store[obj] = {
                'time_h': [], 'high': [],
                'time_l': [], 'low': []
            }

    def init_plot(self):
        # Aspetta che arrivino i primi dati per sapere quanti subplot creare
        rospy.loginfo("In attesa di dati per inizializzare i plot...")
        while not self.data_store and not rospy.is_shutdown():
            rospy.sleep(0.1)
        
        with self.lock:
            obj_names = sorted(self.data_store.keys())
            num_objs = len(obj_names)
            
            self.fig, axes = plt.subplots(num_objs, 1, figsize=(8, 4 * num_objs), sharex=True)
            if num_objs == 1: axes = [axes] # Gestione caso singolo oggetto
            
            for name, ax in zip(obj_names, axes):
                self.axs[name] = ax
                ax.set_title(f"Risk Analysis: {name}")
                ax.set_ylim(-0.1, 1.1)
                ax.grid(True, alpha=0.3)
                ax.set_ylabel("Risk Level")
            
            axes[-1].set_xlabel("Time [s]")
            plt.tight_layout()

    def update_plot(self, frame):
        with self.lock:
            for obj, ax in self.axs.items():
                ax.cla() # Pulisce il subplot per il nuovo frame
                
                # Plot Highroad
                if self.data_store[obj]['time_h']:
                    ax.plot(self.data_store[obj]['time_h'], self.data_store[obj]['high'], 
                            label='Highroad', color='blue', linewidth=2)
                
                # Plot Lowroad
                if self.data_store[obj]['time_l']:
                    ax.plot(self.data_store[obj]['time_l'], self.data_store[obj]['low'], 
                            label='Lowroad', color='red', linestyle='--', linewidth=2)
                
                ax.set_title(f"Risk: {obj}")
                ax.set_ylim(-0.05, 1.05)
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.2)

if __name__ == '__main__':
    rospy.init_node('amygdala_live_plotter', anonymous=True)
    
    plotter = LiveRiskPlotter(window_size=100)
    plotter.init_plot()
    
    # Animazione Matplotlib
    ani = FuncAnimation(plotter.fig, plotter.update_plot, interval=100) # 10Hz update
    
    plt.show()
    rospy.spin()