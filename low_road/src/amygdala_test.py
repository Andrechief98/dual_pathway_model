#!/usr/bin/env python3
import rospy
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class AmygdalaNode:
    def __init__(self):
        # ROS node initialization
        rospy.init_node("amygdala_node", anonymous=True)

        self.u_amyg = 0
        self.u_cortex = 0
        self.u_eff = 0
        self.previous_u_eff = 0

        self.fear_level = 0
        self.dot_fear_level = 0

        self.wn = 10        # Natural frequency
        self.zeta = 0.9     # Damping coefficient
        self.alpha = 0.5    # Exponential decay coefficient

        self.previous_time_instant = rospy.get_time()

        # Data for plotting
        self.time_data = []
        self.fear_data = []
        self.start_time = rospy.get_time()

        # --- Setup Matplotlib Figure ---
        plt.ion()
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)
        self.line, = self.ax.plot([], [], label="Fear level", color="red")
        self.ax.set_xlabel("Tempo [s]")
        self.ax.set_ylabel("Fear level")
        self.ax.set_ylim(0, 1.2)
        self.ax.set_xlim(0, 10)
        self.ax.legend()
        self.ax.grid(True)

        # Slider per controllare u_eff (0 - 1)
        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.slider = Slider(ax_slider, 'u_eff', 0.0, 1.0, valinit=self.u_eff)
        self.slider.on_changed(self.update_reference)

    def update_reference(self, val):
        """Aggiornamento interattivo del valore di riferimento della paura"""
        self.u_eff = self.slider.val

    def fear_dynamics(self):
        x1 = self.fear_level
        x2 = self.dot_fear_level

        current_time_instant = rospy.get_time()
        dt = current_time_instant - self.previous_time_instant
        if dt <= 0:
            return

        # Crescita del riferimento
        if self.u_eff >= x1:
            dx1 = x2
            dx2 = -2*self.zeta*self.wn*x2 - (self.wn**2)*x1 + (self.wn**2)*self.u_eff
            x1 = x1 + dx1 * dt
            x2 = x2 + dx2 * dt
        else:
            dx1 = -self.alpha * (x1 - self.u_eff)
            x1 = x1 + dx1 * dt
            x2 = dx1

        self.fear_level = max(0, min(1.2, x1))  # clamp tra 0 e 1.2
        self.dot_fear_level = x2
        self.previous_time_instant = current_time_instant

    def update_plot(self):
        """Graphical update in real time"""
        t = rospy.get_time() - self.start_time
        self.time_data.append(t)
        self.fear_data.append(self.fear_level)

        # Mantiene la finestra temporale di 10s
        if t > 10:
            self.ax.set_xlim(t-10, t)

        self.line.set_xdata(self.time_data)
        self.line.set_ydata(self.fear_data)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.pause(0.01)

    def spin(self):
        rate = rospy.Rate(50)  # 50 Hz
        while not rospy.is_shutdown():
            self.fear_dynamics()
            self.update_plot()
            rate.sleep()

        rospy.loginfo("ROS loop ended.")


if __name__ == "__main__":
    node = AmygdalaNode()
    node.spin()
