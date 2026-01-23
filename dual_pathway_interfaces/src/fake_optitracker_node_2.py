#!/usr/bin/env python3

import rospy
import tf2_ros
import numpy as np
import tf.transformations as tft
from geometry_msgs.msg import Pose, Twist
from gazebo_msgs.msg import ModelStates
import math

class FakeOptiTrackerNode:
    def __init__(self, objects_config, reference_name="mir", rate_hz=100):
        rospy.init_node('optitracker_node', anonymous=True)
        
        self.objects_config = objects_config
        self.reference_name = reference_name
        self.rate = rospy.Rate(rate_hz)
        
        # Buffer per ascoltare l'odometria reale del robot
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.model_state_pub = rospy.Publisher(
            '/optitracker/model_states', 
            ModelStates, 
            queue_size=1
        )

        rospy.loginfo("Fake OptiTracker: sincronizzazione con frame 'odom' avviata.")

    def get_robot_odom_pose(self):
        """Recupera la posizione attuale del robot nel frame odom."""
        try:
            # Cerchiamo la trasformata da odom a base_link (la posizione del robot)
            trans = self.tf_buffer.lookup_transform('odom', 'base_link', rospy.Time(0), rospy.Duration(1.0))
            return trans.transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            # Se TF non è ancora pronto, restituiamo una trasformata identità
            return None

    def _matrix_to_pose(self, matrix):
        pose = Pose()
        trans = tft.translation_from_matrix(matrix)
        quat = tft.quaternion_from_matrix(matrix)
        pose.position.x, pose.position.y, pose.position.z = trans
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat
        return pose

    def update_synthetic_data(self):
        t = rospy.get_time()
        msg = ModelStates()
        
        # Otteniamo la posizione reale del robot per pubblicarla correttamente nel messaggio
        robot_transform = self.get_robot_odom_pose()

        for name, config in self.objects_config.items():
            if name == self.reference_name:
                # Pubblichiamo la posizione REALE del robot letta da odom
                if robot_transform:
                    p = Pose()
                    p.position.x = robot_transform.translation.x
                    p.position.y = robot_transform.translation.y
                    p.position.z = robot_transform.translation.z
                    p.orientation = robot_transform.rotation
                    msg.pose.append(p)
                else:
                    msg.pose.append(Pose()) # Fallback se TF fallisce
            else:
                # Per gli ostacoli, usiamo il centro specificato in odom
                cx, cy = config.get("center", [0.0, 0.0])
                r = config.get("radius", 0.0)
                
                # Calcolo del movimento circolare (opzionale, se r > 0)
                freq = 0.0
                x = cx + r * math.cos(freq * t)
                y = cy + r * math.sin(freq * t)
                
                raw_matrix = tft.translation_matrix([x, y, 0.0])
                msg.pose.append(self._matrix_to_pose(raw_matrix))
            
            msg.name.append(name)
            msg.twist.append(Twist())
            
        return msg

    def run(self):
        while not rospy.is_shutdown():
            msg = self.update_synthetic_data()
            if msg.pose: # Pubblica solo se abbiamo dati
                self.model_state_pub.publish(msg)
            self.rate.sleep()

if __name__ == '__main__':
    # Coordinate assolute rispetto all'origine del frame /odom
    objects = {
        'mir': {
            'center': [0.0, 0.0] # Questo verrà sovrascritto dalla posizione reale
        }, 
        # 'rover': {
        #     'center': [5.0, 2.0] # Ostacolo fisso (o orbitante) nel punto 5,2 di odom
        # }, 
        # 'person': {
        #     'center': [1.0, 1.0] # Ostacolo statico a 1 metro dall'origine odom
        # },
        'cardboard_box': {
            'center': [2.0, 2.5] # Ostacolo statico a 1 metro dall'origine odom
        }
    }
    
    try:
        node = FakeOptiTrackerNode(objects, reference_name='mir', rate_hz=100)
        node.run()
    except rospy.ROSInterruptException:
        pass