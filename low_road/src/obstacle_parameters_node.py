#!/usr/bin/env python3
import rospy
import yaml
import math
import numpy as np
import os
import json
from mpc_planner.msg import mpcParameters, objectCostParameters
from std_msgs.msg import String, Float32
from tkinter import Tk, Scale, VERTICAL, Label, Button
import threading

script_dir = os.path.dirname(os.path.abspath(__file__))
mpc_params_file_path = "../../mpc_planner/config/mpc_params.yaml"
full_path = os.path.join(script_dir, mpc_params_file_path)

class Node:
    def __init__(self, yaml_path):
        # --- Caricamento parametri iniziali ---
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        self.msg = mpcParameters()
        self.msg.Q = data["mpc_planner"]['Q_weights']
        self.msg.R = data["mpc_planner"]['R_weights']
        self.msg.P = data["mpc_planner"]['P_weights']

        object_msg = objectCostParameters()
        object_msg.objectName = "none"
        object_msg.alfa = data["mpc_planner"]["alfa"][0]
        object_msg.beta = data["mpc_planner"]["beta"][0]

        self.msg.objectsList.append(object_msg)

        self.fuzzy_params = {
            "low":    {"alfa": 1, "beta": 0.1},
            "medium": {"alfa": 50.0, "beta": 0.05},
            "high":   {"alfa": 100.0, "beta": 0.01},
        }

        # --- Setup ROS ---
        rospy.init_node('mpc_param_publisher')
        self.pub = rospy.Publisher('/mpc/params', mpcParameters, queue_size=1)
        self.sub = rospy.Subscriber('/fearlevel', String, self.fearCallback)
        self.rate = rospy.Rate(20)  # Hz

        self.fear = {}

    def fearCallback(self, msg):
        self.fear = json.loads(msg.data)
        return

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
        # print(np.round(values/np.sum(values),2))
        return np.round(values/np.sum(values),2)

    def compute_fuzzy_params(self):
        """Calcola alfa e beta fuzzy-weighted per ogni oggeto (se almeno un oggetto è presente)"""

        if self.fear != {}:

            self.msg.objectsList = []
            for object_name, fear in self.fear.items():
                object_msg = objectCostParameters()
                
                μ_low = self.membership_low(fear)
                μ_med = self.membership_medium(fear)
                μ_high = self.membership_high(fear)

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

                object_msg.objectName = object_name
                object_msg.alfa = (np.float32(np.dot(w, alfa_values)))
                object_msg.beta = (np.float32(np.dot(w, beta_values)))

                self.msg.objectsList.append(object_msg)
        return 



    def run(self):
        while not rospy.is_shutdown():
            # Calcolo fuzzy in tempo reale
            self.compute_fuzzy_params()

            # Pubblicazione parametri NMPC
            self.pub.publish(self.msg)
            self.rate.sleep()


if __name__ == '__main__':
    try:
        mpc_parameters_node = Node(full_path)
        mpc_parameters_node.run()

    except Exception as e:
        rospy.logerr(f"[OBSTACLE PARAMETERS NODE]: {e}")
        pass
