#!/usr/bin/env python3
import rospy
import yaml
import math
import numpy as np
from mpc_planner.msg import mpcParameters
from tkinter import Tk, Scale, VERTICAL, Label, Button
import threading

class FearTrigger:
    def __init__(self, yaml_path):
        # --- Caricamento parametri iniziali ---
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        self.msg = mpcParameters()
        self.msg.Q = data["mpc_planner"]['Q_weights']
        self.msg.R = data["mpc_planner"]['R_weights']
        self.msg.P = data["mpc_planner"]['P_weights']
        self.msg.alfa = data["mpc_planner"]["alfa"]
        self.msg.beta = data["mpc_planner"]["beta"]

        # --- Fuzzy parameters iniziali ---
        self.fear = 0.0  # valore di paura ∈ [0,1]

        # valori di alfa/beta per ciascun livello fuzzy
        self.fuzzy_params = {
            "low":    {"alfa": 1, "beta": 0.1},
            "medium": {"alfa": 15.0, "beta": 0.05},
            "high":   {"alfa": 30.0, "beta": 0.01},
        }

        # --- Setup ROS ---
        rospy.init_node('mpc_param_publisher')
        self.pub = rospy.Publisher('/mpc/params', mpcParameters, queue_size=1)
        self.rate = rospy.Rate(10)  # Hz

        # --- Avvia UI in thread separato ---
        ui_thread = threading.Thread(target=self.create_ui)
        ui_thread.daemon = True
        ui_thread.start()

        self.run()

    def membership_low(self, x):
        """Low: 1 fino a 0.2, poi gaussiana decrescente fino a 0.5"""
        if x <= 0.2:
            return 1.0
        elif x < 0.5:
            sigma = 0.1
            return math.exp(-0.5 * ((x - 0.2) / sigma) ** 2)
        else:
            return 0.0

    def membership_medium(self, x):
        """Medium: gaussiana centrata in 0.5"""
        sigma = 0.1
        return math.exp(-0.5 * ((x - 0.5) / sigma) ** 2)

    def membership_high(self, x):
        """High: cresce da 0.5 a 0.8 (gaussiana), poi resta 1"""
        if x <= 0.5:
            return 0.0
        elif x < 0.8:
            sigma = 0.1
            return math.exp(-0.5 * ((x - 0.8) / sigma) ** 2)
        else:
            return 1.0

    def normalize(self, values):
        """Normalizza valori"""
        # exp_vals = np.exp(values)
        # print(np.round(exp_vals / np.sum(exp_vals), 2))
        # return np.round(exp_vals / np.sum(exp_vals), 2)
        print(np.round(values/np.sum(values),2))
        return np.round(values/np.sum(values),2)

    def compute_fuzzy_params(self):
        """Calcola alfa e beta fuzzy-weighted"""
        μ_low = self.membership_low(self.fear)
        μ_med = self.membership_medium(self.fear)
        μ_high = self.membership_high(self.fear)

        μ_vec = np.array([μ_low, μ_med, μ_high])
        w = self.normalize(μ_vec)  # normalizzazione pesi

        alfa_values = np.array([
            self.fuzzy_params["low"]["alfa"],
            self.fuzzy_params["medium"]["alfa"],
            self.fuzzy_params["high"]["alfa"]
        ])
        beta_values = np.array([
            self.fuzzy_params["low"]["beta"],
            self.fuzzy_params["medium"]["beta"],
            self.fuzzy_params["high"]["beta"]
        ])

        self.msg.alfa = float(np.dot(w, alfa_values))
        self.msg.beta = float(np.dot(w, beta_values))

    def create_ui(self):
        """Crea interfaccia con slider per paura e parametri fuzzy"""
        root = Tk()
        root.title("NMPC Fuzzy Parameter Controller")

        # Slider per la paura
        Label(root, text="Fear Level [0–1]").pack()
        fear_slider = Scale(root, from_=0.0, to=1.0, orient=VERTICAL, resolution=0.1,
                            command=lambda v: self.set_fear(float(v)))
        fear_slider.set(self.fear)
        fear_slider.pack(fill='x', pady=5)

        # # Slider per alfa e beta fuzzy (per ciascun livello)
        # for level in ["low", "medium", "high"]:
        #     Label(root, text=f"α ({level})").pack()
        #     Scale(root, from_=0.0, to=200.0, orient=VERTICAL, resolution=1,
        #           command=lambda v, l=level: self.set_fuzzy_param(l, "alfa", float(v))
        #           ).pack(fill='x')
        #     Label(root, text=f"β ({level})").pack()
        #     Scale(root, from_=0.0, to=100.0, orient=VERTICAL, resolution=1,
        #           command=lambda v, l=level: self.set_fuzzy_param(l, "beta", float(v))
        #           ).pack(fill='x')

        root.mainloop()

    def set_fear(self, value):
        self.fear = value

    def set_fuzzy_param(self, level, param, value):
        self.fuzzy_params[level][param] = value

    def run(self):
        while not rospy.is_shutdown():
            # Calcolo fuzzy in tempo reale
            self.compute_fuzzy_params()

            # Pubblicazione parametri NMPC
            self.pub.publish(self.msg)
            self.rate.sleep()


if __name__ == '__main__':
    try:
        yaml_path = "/home/andrea/ros_packages_aggiuntivi/src/dual_pathway_model/mpc_planner/config/mpc_params.yaml"
        FearTrigger(yaml_path)
    except rospy.ROSInterruptException:
        pass
