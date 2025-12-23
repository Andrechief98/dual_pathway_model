#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import matplotlib.pyplot as plt
from collections import deque
import json
import numpy as np

class Realtime2DColorMapPlotter:
    def __init__(self, window_size=100, refresh_hz=30):
        self.window_size = window_size
        self.refresh_hz = refresh_hz

        # Buffer dati
        self.dist_buffer = deque(maxlen=self.window_size)
        self.rad_buffer  = deque(maxlen=self.window_size)

        self._running = True

        rospy.init_node("realtime_2d_fear_plotter", anonymous=True)
        rospy.Subscriber("/fearlevel", String, self.callback_fear_dummy)
        rospy.Subscriber("/thalamus/info", String, self.callback_thalamus)
        self.fear_level = 0

        # Setup plot 2D
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 6))

        self.ax.set_xlabel("Rel Dist (m)")
        self.ax.set_ylabel("Rad Vel (m/s)")
        # self.ax.set_title("Realtime 2D Fear Map")

        # Funzioni matematiche (parametri riportati a quelli originali)
        def gaussian(x, mu=0, sigma=2):
            return np.exp(-((x - mu)**2) / (2 * sigma**2))

        def sigmoid(y, k=0.5, y0=-1):
            return 1.0 / (1.0 + np.exp(-k * (y - y0)))

        def fear_function(x, y):
            return gaussian(x) * sigmoid(y)

        self.fear_function = fear_function

        # Griglia per la mappa (2D colormap che rappresenta la "z")
        self.X = np.linspace(0, 5, 100)
        self.Y = np.linspace(-1, 1, 100)
        self.Xm, self.Ym = np.meshgrid(self.X, self.Y)
        self.Zm = self.fear_function(self.Xm, self.Ym)
        self.Zm = self.Zm/self.Zm.max()

        # Disegna la mappa statica (pcolormesh) e colorbar
        self.mesh = self.ax.pcolormesh(self.Xm, self.Ym, self.Zm, cmap="viridis", shading="auto", alpha=0.9)
        self.cbar = self.fig.colorbar(self.mesh, ax=self.ax)
        self.cbar.set_label('Fear')

        # Punto dinamico e scia (in 2D, colore determinato dal valore di fear)
        # Inizializziamo come scatter vuoti
        self.dynamic_point = self.ax.scatter([], [], c='red', s=40, edgecolors='k')
        self.trail = self.ax.plot([], [], [], color="red", linewidth=2)[0]

        # Deques per la scia
        self.all_x = deque(maxlen=self.window_size)
        self.all_y = deque(maxlen=self.window_size)
        self.all_z = deque(maxlen=self.window_size)

        self.fig.canvas.mpl_connect('close_event', self.handle_close)

    def callback_fear_dummy(self, msg):

        self.fear_level = list(json.loads(msg.data).values())[0] # first object

    def callback_thalamus(self, msg):
        try:
            data = json.loads(msg.data)
            if "relative_info" in data and len(data["relative_info"]) > 0:
                object_name = list(data["relative_info"].keys())[0] # first object
                obj = data["relative_info"][object_name]
                self.dist_buffer.append(obj.get("relative_dist", 0.0))
                self.rad_buffer.append(obj.get("radial_vel", 0.0))
        except Exception as e:
            rospy.logwarn(f"JSON Parse error: {e}")

    def handle_close(self, event):
        self._running = False
        rospy.signal_shutdown("Window Closed")

    def update_plot(self):
        if len(self.dist_buffer) < 1:
            self.fig.canvas.flush_events()
            return

        # Ultimo punto
        x = self.dist_buffer[-1]
        y = self.rad_buffer[-1]
        # z = self.fear_function(x, y)
        z = float(self.fear_level)

        # Aggiorna scia
        self.all_x.append(x)
        self.all_y.append(y)
        self.all_z.append(z)

        # Aggiorna punto dinamico
        self.dynamic_point.set_offsets(np.c_[[x], [y]])  # point stays red

        # Aggiorna scia (tutti i punti recenti)
        if len(self.all_x) > 0:
            offsets = np.c_[list(self.all_x), list(self.all_y)]
            colors  = np.array(list(self.all_z))

            self.trail.set_data(self.all_x, self.all_y)

        plt.draw()
        self.fig.canvas.start_event_loop(0.001)

    def spin(self):
        rate = rospy.Rate(self.refresh_hz)
        while not rospy.is_shutdown() and self._running:
            self.update_plot()
            rate.sleep()
        self.cleanup()

    def cleanup(self):
        plt.ioff()
        plt.close(self.fig)
        rospy.loginfo("Plot closed")

if __name__ == "__main__":
    plotter = Realtime2DColorMapPlotter(window_size=200, refresh_hz=30)
    plotter.spin()
