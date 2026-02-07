#!/usr/bin/env python3

import rospy
import tf2_ros
import tf2_geometry_msgs  # Fondamentale per far funzionare tf_buffer.transform con PoseStamped
import tf.transformations as tft
from geometry_msgs.msg import Pose, Twist, Quaternion, PoseStamped
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
import math

class FakeOptiTrackerNode:
    def __init__(self, model_names, reference_name="mir", target_frame="odom", rate_hz=100):
        rospy.init_node('optitracker_node', anonymous=True)
        
        self.model_names = model_names
        self.reference_name = reference_name
        self.target_frame = target_frame
        self.dt = 1.0 / rate_hz
        self.rate = rospy.Rate(rate_hz)

        # Stato: [x, y, theta, v, omega, initialized]
        self.states = {name: [0.0, 0.0, 0.0, 0.0, 0.0, False] for name in self.model_names}
        self.odom_pubs = {name: rospy.Publisher(f'/{name}/odom', Odometry, queue_size=10) for name in self.model_names}
        
        # Buffer TF per la trasformazione world -> odom
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Subscriber
        for name in self.model_names:
            # 1. Pose reali da VRPN per inizializzazione
            rospy.Subscriber(f'/vrpn_client_node/{name}/pose', PoseStamped, self._vrpn_callback, name)
            
            # 2. Comandi di velocità (tranne per il riferimento se gestito da odom reale)
            if name != self.reference_name:
                rospy.Subscriber(f'/{name}/cmd_vel', Twist, self._cmd_vel_callback, name)

        self.model_state_pub = rospy.Publisher('/optitracker/model_states', ModelStates, queue_size=1)
        rospy.loginfo(f"Fake OptiTracker: Inizializzazione in corso verso frame '{target_frame}'...")

    def _vrpn_callback(self, msg, name):
        """Riceve la posa, la trasforma in odom e inizializza lo stato se non già fatto."""
        # Se lo stato è già inizializzato, ignoriamo i messaggi VRPN successivi 
        # (lasciamo che sia il cmd_vel a comandare)
        if self.states[name][5]:
            return

        try:
            # Trasformazione automatica nel frame target (es. 'odom')
            # Timeout breve per non bloccare la callback
            transformed_msg = self.tf_buffer.transform(msg, self.target_frame, timeout=rospy.Duration(0.1))
            
            p = transformed_msg.pose
            q = [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
            theta = tft.euler_from_quaternion(q)[2]

            self.states[name][0:3] = [p.position.x, p.position.y, theta]
            self.states[name][5] = True
            rospy.loginfo(f"Modello '{name}' inizializzato in {self.target_frame}: x={p.position.x:.2f}, y={p.position.y:.2f}")

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            # Fallisce silenziosamente finché la TF world->odom non è disponibile
            pass

    def _cmd_vel_callback(self, msg, name):
        if self.states[name][5]:
            self.states[name][3] = msg.linear.x
            self.states[name][4] = msg.angular.z

    def _update_physics(self):
        """Integrazione numerica."""
        for name in self.states:
            if not self.states[name][5]: continue
            
            if name == self.reference_name:
                # Per il robot principale, proviamo a leggere la posizione reale da TF odom->base_link
                try:
                    trans = self.tf_buffer.lookup_transform(self.target_frame, 'base_link', rospy.Time(0))
                    self.states[name][0] = trans.transform.translation.x
                    self.states[name][1] = trans.transform.translation.y
                    q = [trans.transform.rotation.x, trans.transform.rotation.y, 
                         trans.transform.rotation.z, trans.transform.rotation.w]
                    self.states[name][2] = tft.euler_from_quaternion(q)[2]
                except:
                    pass # Se fallisce, mantiene l'ultima posizione nota
            else:
                # Modello Uniciclo per gli altri oggetti
                x, y, theta, v, omega, _ = self.states[name]
                new_theta = theta + (omega * self.dt)
                new_x = x + (v * math.cos(new_theta) * self.dt)
                new_y = y + (v * math.sin(new_theta) * self.dt)
                self.states[name][0:3] = [new_x, new_y, new_theta]

    def _publish_all(self):
        msg = ModelStates()
        now = rospy.Time.now()
        
        for name, data in self.states.items():
            if not data[5]: continue
            
            x, y, theta, v, omega, _ = data
            q_list = tft.quaternion_from_euler(0, 0, theta)
            quat = Quaternion(*q_list)

            # Pose e Twist per ModelStates
            p = Pose(); p.position.x = x; p.position.y = y; p.orientation = quat
            tw = Twist(); tw.linear.x = v; tw.angular.z = omega
            
            msg.name.append(name)
            msg.pose.append(p)
            msg.twist.append(tw)

            # Pubblicazione Odom Individuale
            od = Odometry()
            od.header.stamp = now
            od.header.frame_id = self.target_frame
            od.child_frame_id = f"{name}_link"
            od.pose.pose = p
            od.twist.twist = tw
            self.odom_pubs[name].publish(od)

        if msg.name:
            self.model_state_pub.publish(msg)

    def run(self):
        while not rospy.is_shutdown():
            self._update_physics()
            self._publish_all()
            self.rate.sleep()

if __name__ == '__main__':
    models = ['mir', 'rover', 'person']
    try:
        node = FakeOptiTrackerNode(models, reference_name='mir')
        node.run()
    except rospy.ROSInterruptException:
        pass