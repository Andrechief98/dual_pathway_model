#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import matplotlib.pyplot as plt
from collections import deque

class RealtimePlotter:
    def __init__(self, topic_name="/fearlevel", window_size=200, refresh_hz=20):
        self.topic_name = topic_name
        self.window_size = window_size
        self.refresh_hz = refresh_hz
        self.data_buffer = deque(maxlen=self.window_size)
        self._running = True  # 👈 Flag per sapere se tenere aperto il loop

        rospy.init_node("realtime_plotter", anonymous=True)
        rospy.Subscriber(self.topic_name, String, self.callback)

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], "b-", linewidth=2)
        self.ax.set_xlim(0, self.window_size)
        self.ax.set_ylim(-0.1, 1.5)
        self.ax.grid(True)
        self.ax.set_title(f"Realtime Topic Plot: {self.topic_name}")

        # 👇 Collega evento di chiusura finestra
        self.fig.canvas.mpl_connect('close_event', self.handle_close)

    def handle_close(self, event):
        """Callback quando la finestra viene chiusa"""
        rospy.loginfo("🛑 Finestra chiusa manualmente — arresto del nodo.")
        self._running = False
        rospy.signal_shutdown("Finestra chiusa")

    def callback(self, msg):
        try:
            value = float(msg.data)
            self.data_buffer.append(value)
        except ValueError:
            rospy.logwarn(f"Messaggio non numerico ricevuto: {msg.data}")

    def update_plot(self):
        if self.data_buffer:
            data = list(self.data_buffer)
            self.line.set_xdata(range(len(data)))
            self.line.set_ydata(data)
            plt.draw()
            plt.pause(0.001)

    def spin(self):
        rate = rospy.Rate(self.refresh_hz)
        try:
            while not rospy.is_shutdown() and self._running:
                self.update_plot()
                rate.sleep()
        except rospy.ROSInterruptException:
            pass
        finally:
            self.cleanup()

    def cleanup(self):
        plt.ioff()
        plt.close(self.fig)
        rospy.loginfo("✅ Plot chiuso correttamente")

if __name__ == "__main__":
    plotter = RealtimePlotter("/fearlevel", 200, 20)
    plotter.spin()
