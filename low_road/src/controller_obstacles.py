#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

class ProportionalController:
    def __init__(self, name, cmd_topic, odom_topic, goal_x, goal_y, max_vel, kp_l=1.4, kp_a=4.0):
        self.name = name
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.max_vel = max_vel
        
        self.kp_l = kp_l
        self.kp_a = kp_a
        
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        
        self.pub = rospy.Publisher(cmd_topic, Twist, queue_size=10)
        self.sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback)

    def normalize_angle(self, angle):
        """Mantiene l'angolo nell'intervallo [-pi, pi]"""
        return math.atan2(math.sin(angle), math.cos(angle))

    def odom_callback(self, msg):
        # Posizione
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        # Orientamento originale
        orientation_q = msg.pose.pose.orientation
        quaternion = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, raw_yaw) = euler_from_quaternion(quaternion)

        self.current_yaw = self.normalize_angle(raw_yaw)

    def update_control(self):
        delta_x = self.goal_x - self.current_x
        delta_y = self.goal_y - self.current_y
        
        distance = math.sqrt(delta_x**2 + delta_y**2)
        angle_to_goal = math.atan2(delta_y, delta_x)
        
        # Calcolo errore angolare basato sul yaw corretto
        angle_error = self.normalize_angle(angle_to_goal - self.current_yaw)
        
        msg = Twist()

        if distance < 0.15: # Tolleranza arrivo
            msg.linear.x = 0.0
            msg.angular.z = 0.0
        else:
            # Controllo Angolare
            msg.angular.z = self.kp_a * angle_error
            
            # Controllo Lineare (rallenta se non è allineato)
            v_x = self.kp_l * distance * math.cos(angle_error)
            
            # Saturazione velocità massima
            if abs(v_x) > self.max_vel:
                v_x = math.copysign(self.max_vel, v_x)
                
            msg.linear.x = max(0.0, v_x) 

        self.pub.publish(msg)

def main():
    rospy.init_node('multi_object_controller', anonymous=True)

    config_list = [
        {
            "name": "person",
            "cmd": "/person_walking/cmd_vel",
            "odom": "/person_walking/odom",
            "goal": (0.0, 2.0),
            "max_v": 0.5,
        },
        {
            "name": "rover",
            "cmd": "/rover/cmd_vel",
            "odom": "/rover/odom",
            "goal": (0.0, -2.0),
            "max_v": 0.5,
        }
    ]

    controllers = []
    for c in config_list:
        ctrl = ProportionalController(
            name=c["name"],
            cmd_topic=c["cmd"],
            odom_topic=c["odom"],
            goal_x=c["goal"][0],
            goal_y=c["goal"][1],
            max_vel=c["max_v"],
        )
        controllers.append(ctrl)

    rate = rospy.Rate(20) # Frequenza leggermente più alta per maggiore reattività
    while not rospy.is_shutdown():
        for ctrl in controllers:
            ctrl.update_control()
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass