#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import json
import numpy as np

class Realtime3DPlotter:
    def __init__(self, window_size=200, refresh_hz=20):

        self.window_size = window_size
        self.refresh_hz = refresh_hz

        self.fear_buffer = deque(maxlen=self.window_size)
        self.dist_buffer = deque(maxlen=self.window_size)
        self.rad_buffer  = deque(maxlen=self.window_size)

        



        self._running = True

        rospy.init_node("realtime_3d_plotter", anonymous=True)

        rospy.Subscriber("/fearlevel", String, self.callback_fear)
        rospy.Subscriber("/thalamus/info", String, self.callback_thalamus)

        # --- Setup grafico ---
        plt.ion()
        self.fig = plt.figure(figsize=(8, 7))
        self.ax = self.fig.add_subplot(111, projection="3d")

        # scatter iniziale vuoto
        self.scatter = self.ax.scatter([], [], [], c=[], cmap="viridis", s=30)

        self.colorbar = self.fig.colorbar(self.scatter, ax=self.ax, shrink=0.6, label="Fear Level")

        self.ax.set_xlabel("Relative Distance")
        self.ax.set_ylabel("Radial Velocity")
        self.ax.set_zlabel("Fear Level")
        self.ax.set_title("Realtime 3D Fear Plot")

        # importante: NON ridisegnare la colorbar ogni volta
        self.fig.canvas.mpl_connect('close_event', self.handle_close)

        # definisci una tua funzione (personalizzabile)
        def gaussian(x, mu=0, sigma=2):
            return np.exp(-((x - mu)**2) / (2 * sigma**2))

        # -----------------------------
        #  Sigmoid (radial velocity)
        # -----------------------------
        def sigmoid(y, k=0.5, y0=-1):
            return 1.0 / (1.0 + np.exp(-k * (y - y0)))

        # -----------------------------
        #  Composed Fear Function
        # -----------------------------
        def fear_function(x, y):
            return gaussian(x) * sigmoid(y)

        self.fear_function = fear_function

        # disegna una superficie della funzione (una volta sola)
        X = np.linspace(0, 5, 50)
        Y = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(X, Y)
        Z = self.fear_function(X, Y)

        self.ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.5)

        # scatter dinamico
        self.scatter = self.ax.scatter([], [], [], c='red', s=30)

        # liste cumulative di punti
        self.all_x = []
        self.all_y = []
        self.all_z = []

        

    # ---------------------------------------------------------

    def handle_close(self, event):
        self._running = False
        rospy.signal_shutdown("Finestra chiusa")

    # ---------------------------------------------------------

    def callback_fear(self, msg):
        try:
            self.fear_buffer.append(float(msg.data))
        except:
            rospy.logwarn("Valore non numerico in /fearlevel")

    # ---------------------------------------------------------

    def callback_thalamus(self, msg):
        try:
            data = json.loads(msg.data)
            object_list = list(data["relative_info"].keys())
            object_name = object_list[0]
            data = data["relative_info"][object_name]
            self.dist_buffer.append(data.get("relative_dist"))
            self.rad_buffer.append(data.get("radial_vel"))
        except:
            rospy.logwarn("Errore parsing JSON in /thalamus/info")

    # ---------------------------------------------------------

    def update_plot(self):
        # sincronizzazione dei topic
        min_len = min(len(self.fear_buffer), len(self.dist_buffer), len(self.rad_buffer))
        if min_len < 1:
            return

        # ultimo punto ricevuto
        x = self.dist_buffer[-1]
        y = self.rad_buffer[-1]

        # generi lo z dalla funzione (invece che dal topic)
        z = self.fear_function(x, y)

        # aggiungi nuovo punto
        self.all_x.append(x)
        self.all_y.append(y)
        self.all_z.append(z)

        # rimuovi punti se superano window_size
        if len(self.all_x) > self.window_size:
            self.all_x.pop(0)
            self.all_y.pop(0)
            self.all_z.pop(0)

        # aggiorna scatter (i punti NON si muovono più)
        self.scatter._offsets3d = (self.all_x, self.all_y, self.all_z)

        plt.draw()
        plt.pause(0.001)



    # ---------------------------------------------------------

    def spin(self):
        rate = rospy.Rate(self.refresh_hz)
        while not rospy.is_shutdown() and self._running:
            self.update_plot()
            rate.sleep()

        self.cleanup()

    # ---------------------------------------------------------

    def cleanup(self):
        plt.ioff()
        plt.close(self.fig)
        rospy.loginfo("Plot chiuso correttamente")

# ---------------------------------------------------------

if __name__ == "__main__":
    plotter = Realtime3DPlotter(window_size=200, refresh_hz=20)
    plotter.spin()
