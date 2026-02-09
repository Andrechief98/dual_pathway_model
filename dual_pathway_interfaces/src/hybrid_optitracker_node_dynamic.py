#!/usr/bin/env python3

import rospy
import tf2_ros
import tf2_geometry_msgs 
import numpy as np
import tf.transformations as tft
import math
from geometry_msgs.msg import PoseStamped, Pose, Twist, Quaternion
from nav_msgs.msg import Odometry  # Importato per il topic /rover/odom
from gazebo_msgs.msg import ModelStates
from functools import partial

class HybridOptiTrackerNode:
    def __init__(self, object_names, target_frame="odom", rate_hz=50):
        rospy.init_node('hybrid_optitracker_node', anonymous=True)
        
        self.object_names = object_names
        self.target_frame = target_frame
        self.rate = rospy.Rate(rate_hz)
        self.dt = 1.0 / rate_hz
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.latest_poses = {name: Pose() for name in self.object_names}
        self.initialized = {name: False for name in self.object_names}
        
        # Stato Rover: [x, y, theta, v, omega]
        self.rover_state = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Publishers
        self.model_state_pub = rospy.Publisher('/optitracker/model_states', ModelStates, queue_size=1)
        self.rover_odom_pub = rospy.Publisher('/rover/odom', Odometry, queue_size=10)
        
        # Subscribers OptiTrack
        for name in self.object_names:
            topic_name = f"/vrpn_client_node/{name}/pose"
            rospy.Subscriber(topic_name, PoseStamped, partial(self.pose_callback, obj_name=name))

        # Subscriber cmd_vel per Rover
        rospy.Subscriber('/rover/cmd_vel', Twist, self.rover_cmd_vel_callback)

    def pose_callback(self, msg, obj_name):
        if obj_name == "rover" and self.initialized["rover"]:
            return

        try:
            transformed_msg = self.tf_buffer.transform(msg, self.target_frame, timeout=rospy.Duration(0.1))
            curr_pose = transformed_msg.pose
            
            if obj_name == "rover":
                q = [curr_pose.orientation.x, curr_pose.orientation.y, 
                     curr_pose.orientation.z, curr_pose.orientation.w]
                theta = tft.euler_from_quaternion(q)[2]
                self.rover_state[0:3] = [curr_pose.position.x, curr_pose.position.y, theta]
                self.initialized["rover"] = True
                rospy.loginfo(f"Rover inizializzato a: x={curr_pose.position.x:.2f}, y={curr_pose.position.y:.2f}")

            self.latest_poses[obj_name] = curr_pose
            self.initialized[obj_name] = True

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            pass

    def rover_cmd_vel_callback(self, msg):
        if self.initialized["rover"]:
            self.rover_state[3] = msg.linear.x
            self.rover_state[4] = msg.angular.z

    def update_rover_physics(self):
        """Integrazione numerica per il Rover."""
        if not self.initialized["rover"]:
            return

        x, y, theta, v, omega = self.rover_state

        new_theta = theta + (omega * self.dt)
        new_x = x + (v * math.cos(new_theta) * self.dt)
        new_y = y + (v * math.sin(new_theta) * self.dt)

        self.rover_state[0:3] = [new_x, new_y, new_theta]
        
        q_list = tft.quaternion_from_euler(0, 0, new_theta)
        self.latest_poses["rover"].position.x = new_x
        self.latest_poses["rover"].position.y = new_y
        self.latest_poses["rover"].orientation = Quaternion(*q_list)

    def publish_rover_odom(self):
        """Pubblica l'odometria specifica per il rover."""
        if not self.initialized["rover"]:
            return

        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = self.target_frame
        odom.child_frame_id = "rover_base_link" # Puoi cambiare il nome del link se necessario

        # Posa
        odom.pose.pose = self.latest_poses["rover"]

        # Velocità
        odom.twist.twist.linear.x = self.rover_state[3]
        odom.twist.twist.angular.z = self.rover_state[4]

        self.rover_odom_pub.publish(odom)

    def run(self):
        rospy.loginfo("Nodo Hybrid OptiTracker con Odom avviato...")
        while not rospy.is_shutdown():
            # 1. Aggiorna fisica
            self.update_rover_physics()
            
            # 2. Pubblica ModelStates (Person + Rover + altri)
            ms_msg = ModelStates()
            for name in self.object_names:
                if self.initialized[name]:
                    ms_msg.name.append(name)
                    ms_msg.pose.append(self.latest_poses[name])
                    tw = Twist()
                    if name == "rover":
                        tw.linear.x = self.rover_state[3]
                        tw.angular.z = self.rover_state[4]
                    ms_msg.twist.append(tw)
            
            if ms_msg.name:
                self.model_state_pub.publish(ms_msg)
            
            # 3. Pubblica Odometria specifica del rover
            self.publish_rover_odom()
                
            self.rate.sleep()

if __name__ == '__main__':
    objects = ['mir', 'rover', 'person']
    try:
        node = HybridOptiTrackerNode(objects)
        node.run()
    except rospy.ROSInterruptException:
        pass