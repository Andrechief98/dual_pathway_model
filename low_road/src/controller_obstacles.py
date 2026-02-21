#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

class ProportionalController:
    def __init__(self, name, cmd_topic, odom_topic, waypoints, max_vel, kp_l=1.4, kp_a=4.0):
        self.name = name
        # Gestiamo sia un singolo goal che una lista di waypoints
        if isinstance(waypoints, list):
            self.waypoints = waypoints
        else:
            self.waypoints = [waypoints]
            
        self.current_waypoint_idx = 0
        self.max_vel = max_vel
        
        self.kp_l = kp_l
        self.kp_a = kp_a
        
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        
        # Flag per sapere se il percorso è terminato
        self.all_goals_reached = False
        
        self.pub = rospy.Publisher(cmd_topic, Twist, queue_size=10)
        self.sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback)

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        orientation_q = msg.pose.pose.orientation
        quaternion = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, raw_yaw) = euler_from_quaternion(quaternion)
        self.current_yaw = self.normalize_angle(raw_yaw)

    def update_control(self):
        # Se abbiamo finito tutti i punti, fermiamo il robot
        if self.current_waypoint_idx >= len(self.waypoints):
            if not self.all_goals_reached:
                rospy.loginfo(f"[{self.name}] Tutti i waypoints raggiunti. Stop.")
                self.all_goals_reached = True
            self.stop_robot()
            return

        # Prendi il target attuale
        target_x, target_y = self.waypoints[self.current_waypoint_idx]
        
        delta_x = target_x - self.current_x
        delta_y = target_y - self.current_y
        
        distance = math.sqrt(delta_x**2 + delta_y**2)
        angle_to_goal = math.atan2(delta_y, delta_x)
        angle_error = self.normalize_angle(angle_to_goal - self.current_yaw)
        
        msg = Twist()

        # Soglia di arrivo al waypoint attuale
        if distance < 0.20: 
            rospy.loginfo(f"[{self.name}] Waypoint {self.current_waypoint_idx + 1} raggiunto.")
            self.current_waypoint_idx += 1
            # Fermiamo brevemente per transizione pulita
            self.stop_robot()
        else:
            # Controllo Angolare
            msg.angular.z = self.kp_a * angle_error
            
            # Controllo Lineare
            v_x = self.kp_l * distance * math.cos(angle_error)
            
            if abs(v_x) > self.max_vel:
                v_x = math.copysign(self.max_vel, v_x)
                
            msg.linear.x = max(0.0, v_x) 
            self.pub.publish(msg)

    def stop_robot(self):
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.pub.publish(msg)

def main():
    rospy.init_node('multi_object_controller', anonymous=True)

    # Definizione delle configurazioni con supporto a Waypoints multipli
    config_list = [
        {
            "name": "person",
            "cmd": "/person_walking/cmd_vel",
            "odom": "/person_walking/odom",
            "waypoints": [(3, -1.0), (7, 2.5)],
            "max_v": 0.8,
        },
        {
            "name": "rover",
            "cmd": "/rover/cmd_vel",
            "odom": "/rover/odom",
            "waypoints": [(0, -2)],
            "max_v": 0.5,
        }
    ]

    controllers = []
    for c in config_list:
        ctrl = ProportionalController(
            name=c["name"],
            cmd_topic=c["cmd"],
            odom_topic=c["odom"],
            waypoints=c["waypoints"],
            max_vel=c["max_v"],
        )
        controllers.append(ctrl)

    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        for ctrl in controllers:
            ctrl.update_control()
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass