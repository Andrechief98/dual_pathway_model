#!/usr/bin/env python3
import rospy
import numpy as np
import math
import threading
from mpc_planner.msg import mpcParameters, objectCostParameters
from tkinter import Tk, Scale, HORIZONTAL, Label, Frame, Canvas, Scrollbar, VERTICAL

class FuzzyMPCNode:
    def __init__(self, obstacle_names):
        # --- Configurazione ROS ---
        rospy.init_node('fuzzy_mpc_gui_publisher')
        self.pub = rospy.Publisher('/mpc/params', mpcParameters, queue_size=1)
        self.rate = rospy.Rate(20) 

        self.obstacle_names = obstacle_names
        self.fuzzy_params = {
            "low":    {"alfa": 1.0,   "beta": 0.1},
            "medium": {"alfa": 10.0,  "beta": 0.05},
            "high":   {"alfa": 20.0, "beta": 0.01},
        }

        # --- Setup Interfaccia Grafica ---
        self.root = Tk()
        self.root.title("Fuzzy MPC Controller - Full Tuning")
        self.root.geometry("800x700")

        # --- SEZIONE MATRICI MPC (Q, R, P) ---
        mpc_frame = Frame(self.root, bd=2, relief="ridge", padx=10, pady=10)
        mpc_frame.pack(side="top", fill="x")
        Label(mpc_frame, text="PARAMETRI MPC (Q, R, P)", font=('Helvetica', 12, 'bold')).pack()

        self.q_sliders = self.create_weight_sliders(mpc_frame, "Matrice Q (Stati)", 3, [10.0, 10.0, 5.0], 20.0)
        self.r_sliders = self.create_weight_sliders(mpc_frame, "Matrice R (Ingressi)", 2, [1.0, 5.0], 10.0)
        self.p_sliders = self.create_weight_sliders(mpc_frame, "Matrice P (Terminale)", 3, [30.0, 30.0, 5.0], 100.0)

        # --- SEZIONE OSTACOLI (SCROLLABLE) ---
        Label(self.root, text="OSTACOLI FUZZY", font=('Helvetica', 12, 'bold'), pady=10).pack()
        
        container = Frame(self.root)
        container.pack(fill="both", expand=True)

        self.canvas = Canvas(container)
        scrollbar = Scrollbar(container, orient=VERTICAL, command=self.canvas.yview)
        self.scrollable_frame = Frame(self.canvas)

        # Configurazione per larghezza dinamica
        self.scrollable_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind('<Configure>', self._on_canvas_configure)

        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.obs_sliders = {} # {nome: (slider1, slider2)}

        for name in self.obstacle_names:
            frame = Frame(self.scrollable_frame, bd=2, relief="groove", pady=10)
            frame.pack(fill="x", padx=15, pady=5, expand=True)
            
            Label(frame, text=f"OSTACLOLO: {name}", font=('Helvetica', 10, 'bold'), fg="darkblue").pack()
            
            sub_frame = Frame(frame)
            sub_frame.pack(fill="x", expand=True)

            s1 = Scale(sub_frame, from_=0, to=1, resolution=0.01, orient=HORIZONTAL, label="Sensore A (Fear)")
            s1.set(0.1)
            s1.pack(side="left", fill="x", expand=True, padx=10)
            
            s2 = Scale(sub_frame, from_=0, to=1, resolution=0.01, orient=HORIZONTAL, label="Sensore B (Fear)")
            s2.set(0.1)
            s2.pack(side="right", fill="x", expand=True, padx=10)
            
            self.obs_sliders[name] = (s1, s2)

        self.pub_thread = threading.Thread(target=self.publish_loop, daemon=True)
        self.pub_thread.start()

    def create_weight_sliders(self, parent, title, num, defaults, max_val):
        frame = Frame(parent)
        frame.pack(fill="x", pady=2)
        Label(frame, text=title, width=20, anchor="w").pack(side="left")
        sliders = []
        for i in range(num):
            s = Scale(frame, from_=0, to=max_val, resolution=1, orient=HORIZONTAL)
            s.set(defaults[i])
            s.pack(side="left", fill="x", expand=True)
            sliders.append(s)
        return sliders

    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        # Forza il frame interno a essere largo quanto il canvas
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    # --- Logica Fuzzy ---
    def membership_low(self, x):
        if x <= 0.2: return 1.0
        elif x < 0.5: return math.exp(-0.5 * ((x - 0.2) / 0.1) ** 2)
        return 0.0

    def membership_medium(self, x):
        return math.exp(-0.5 * ((x - 0.5) / 0.1) ** 2)

    def membership_high(self, x):
        if x <= 0.5: return 0.0
        elif x < 0.8: return math.exp(-0.5 * ((x - 0.8) / 0.1) ** 2)
        return 1.0

    def compute_fuzzy(self, fear_val):
        mu = np.array([self.membership_low(fear_val), self.membership_medium(fear_val), self.membership_high(fear_val)])
        total = np.sum(mu)
        w = mu / total if total > 0 else mu
        alfas = np.array([self.fuzzy_params["low"]["alfa"], self.fuzzy_params["medium"]["alfa"], self.fuzzy_params["high"]["alfa"]])
        betas = np.array([self.fuzzy_params["low"]["beta"], self.fuzzy_params["medium"]["beta"], self.fuzzy_params["high"]["beta"]])
        return float(np.dot(w, alfas)), float(np.dot(w, betas))

    def publish_loop(self):
        while not rospy.is_shutdown():
            msg = mpcParameters()
            # Lettura Real-time matrici Q, R, P
            msg.Q = [s.get() for s in self.q_sliders]
            msg.R = [s.get() for s in self.r_sliders]
            msg.P = [s.get() for s in self.p_sliders]
            
            for name, (s1, s2) in self.obs_sliders.items():
                avg_fear = (0.3*s1.get() + 0.7*s2.get())
                alfa, beta = self.compute_fuzzy(avg_fear)
                
                obj = objectCostParameters()
                obj.objectName = name
                obj.alfa = alfa
                obj.beta = beta
                msg.objectsList.append(obj)
            
            self.pub.publish(msg)
            self.rate.sleep()

    def run(self):
        self.root.mainloop()

if __name__ == '__main__':
    try:
        miei_ostacoli = ["person", "rover", "cylinder"]
        node = FuzzyMPCNode(obstacle_names=miei_ostacoli)
        node.run()
    except rospy.ROSInterruptException:
        pass