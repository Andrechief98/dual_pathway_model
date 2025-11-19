#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import json
import numpy as np

class Realtime3DPlotter:
    def __init__(self, window_size=200, refresh_hz=30):
        self.window_size = window_size
        self.refresh_hz = refresh_hz

        # Buffer dati
        self.dist_buffer = deque(maxlen=self.window_size)
        self.rad_buffer  = deque(maxlen=self.window_size)

        self._running = True

        rospy.init_node("realtime_3d_plotter", anonymous=True)
        rospy.Subscriber("/fearlevel", String, self.callback_fear_dummy)
        rospy.Subscriber("/thalamus/info", String, self.callback_thalamus)

        # Setup plot
        plt.ion()
        self.fig = plt.figure(figsize=(8, 7))
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.ax.set_xlabel("Relative Distance")
        self.ax.set_ylabel("Radial Velocity")
        self.ax.set_zlabel("Fear Value (Surface)")
        self.ax.set_title("Realtime 3D Fear Plot (Color‑Encoded Height)")

        # Funzioni matematiche
        def gaussian(x, mu=0, sigma=2):
            return np.exp(-((x - mu)**2) / (2 * sigma**2))

        def sigmoid(y, k=0.5, y0=-1):
            return 1.0 / (1.0 + np.exp(-k * (y - y0)))

        def fear_function(x, y):
            return gaussian(x) * sigmoid(y)

        self.fear_function = fear_function

        # Griglia superficie
        X = np.linspace(0, 5, 40)
        Y = np.linspace(-1, 1, 40)
        X, Y = np.meshgrid(X, Y)
        Z = self.fear_function(X, Y)

        # Superficie color-mapped
        self.ax.plot_surface(
            X, Y, Z, cmap="viridis", alpha=0.4,
            rstride=1, cstride=1,
            linewidth=0, antialiased=False
        )

        # Punto dinamico + scia
        self.dynamic_point = self.ax.scatter([], [], [], c='red', s=40, depthshade=False)
        self.trail = self.ax.plot([], [], [], color="red", linewidth=2)[0]

        # Deques per la scia
        self.all_x = deque(maxlen=self.window_size)
        self.all_y = deque(maxlen=self.window_size)
        self.all_z = deque(maxlen=self.window_size)

        self.fig.canvas.mpl_connect('close_event', self.handle_close)

    def callback_fear_dummy(self, msg):
        pass

    def callback_thalamus(self, msg):
        try:
            data = json.loads(msg.data)
            if "relative_info" in data and len(data["relative_info"]) > 0:
                object_name = list(data["relative_info"].keys())[0]
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
        z = self.fear_function(x, y)

        # Aggiorna scia
        self.all_x.append(x)
        self.all_y.append(y)
        self.all_z.append(z)

        # Aggiorna punto singolo
        self.dynamic_point._offsets3d = ([x], [y], [z])

        # Aggiorna linea della scia
        self.trail.set_data(self.all_x, self.all_y)
        self.trail.set_3d_properties(self.all_z)

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
    plotter = Realtime3DPlotter(window_size=200, refresh_hz=30)
    plotter.spin()
