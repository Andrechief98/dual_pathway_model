#!/usr/bin/env python3
import rospy
import yaml
from mpc_planner.msg import mpcParameters
from tkinter import Tk, Scale, VERTICAL, Label, Button
import threading
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
mpc_params_file_path = "../../mpc_planner/config/mpc_params.yaml"
full_path = os.path.join(script_dir, mpc_params_file_path)

class MPCParamPublisher:
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

        # --- Setup ROS ---
        rospy.init_node('mpc_param_publisher')
        self.pub = rospy.Publisher('/mpc/params', mpcParameters, queue_size=10)
        self.rate = rospy.Rate(10)  # Hz

        # --- Avvia UI in thread separato ---
        ui_thread = threading.Thread(target=self.create_ui)
        ui_thread.daemon = True
        ui_thread.start()

        self.run()

    def create_ui(self):
        """Crea la finestra di configurazione con slider dinamici"""
        root = Tk()
        root.title("MPC Parameter Tuner")

        # Slider per i pesi Q (entrambi uguali)
        Label(root, text="Peso Q[0] = Q[1]").pack()
        q_slider = Scale(root, from_=0.0, to=10.0, orient=VERTICAL, resolution=1,
                         command=lambda v: self.set_Q(1, float(v)))
        q_slider.set(self.msg.Q[0])
        q_slider.pack(fill='x')

         # Slider per i pesi Q )
        Label(root, text="Peso Q[2]").pack()
        q_slider = Scale(root, from_=0.0, to=10.0, orient=VERTICAL, resolution=1,
                         command=lambda v: self.set_Q(2, float(v)))
        q_slider.set(self.msg.Q[2])
        q_slider.pack(fill='x')

        # Slider per i pesi R (indipendenti)
        Label(root, text="Peso R[0]").pack()
        r0_slider = Scale(root, from_=0.0, to=10.0, orient=VERTICAL, resolution=1,
                          command=lambda v: self.set_R(0, float(v)))
        r0_slider.set(self.msg.R[0])
        r0_slider.pack(fill='x')

        Label(root, text="Peso R[1]").pack()
        r1_slider = Scale(root, from_=0.0, to=10.0, orient=VERTICAL, resolution=1,
                          command=lambda v: self.set_R(1, float(v)))
        r1_slider.set(self.msg.R[1])
        r1_slider.pack(fill='x')

        # Slider per i pesi P (entrambi uguali)
        Label(root, text="Peso P[0] = P[1]").pack()
        p_slider = Scale(root, from_=0.0, to=100.0, orient=VERTICAL, resolution=1,
                         command=lambda v: self.set_P(float(v)))
        p_slider.set(self.msg.P[0])
        p_slider.pack(fill='x')

        Label(root, text="Peso alfa").pack()
        p_slider = Scale(root, from_=0.0, to=300.0, orient=VERTICAL, resolution=1,
                         command=lambda v: self.set_alfa(float(v)))
        p_slider.set(self.msg.alfa)
        p_slider.pack(fill='x')

        Label(root, text="Peso beta").pack()
        p_slider = Scale(root, from_=0.0001, to=0.1, orient=VERTICAL, resolution=0.0001,
                         command=lambda v: self.set_beta(float(v)))
        p_slider.set(self.msg.beta)
        p_slider.pack(fill='x')



        # # Slider per l’orizzonte predittivo
        # Label(root, text="Prediction Horizon").pack()
        # horizon_slider = Scale(root, from_=1, to=50, orient=HORIZONTAL, resolution=1,
        #                        command=lambda v: self.set_param('horizon', float(v)))
        # horizon_slider.set(self.msg.horizon)
        # horizon_slider.pack(fill='x')

        # Bottone per attivare/disattivare pubblicazione
        #Button(root, text="Toggle Enable", command=self.toggle_enable).pack(pady=10)

        root.mainloop()

    # --- Callback slider ---
    def set_Q(self, index, value):
        """Aggiorna Q[0] e Q[1] uguali e Q[2]"""
        if index==2:
            self.msg.Q[index] = value
        else:
            self.msg.Q[0] = value
            self.msg.Q[1] = value
    
    def set_P(self, value):
        """Aggiorna P[0] e P[1] uguali"""
        self.msg.P[0] = value
        self.msg.P[1] = value

    def set_R(self, index, value):
        """Aggiorna R[i]"""
        self.msg.R[index] = value

    def set_alfa(self,value):
        """Aggiorna alfa"""
        self.msg.alfa = value

    def set_beta(self,value):
        """Aggiorna beta"""
        self.msg.beta = value


    # def set_param(self, field, value):
    #     """Aggiorna campo generico (fear_level, horizon...)"""
    #     setattr(self.msg, field, value)


    def run(self):
        """Loop principale ROS"""
        while not rospy.is_shutdown():
            self.pub.publish(self.msg)
            self.rate.sleep()

if __name__ == '__main__':
    MPCParamPublisher(full_path)
