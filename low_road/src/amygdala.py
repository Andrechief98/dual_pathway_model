#!/usr/bin/env python3
import rospy
import json
import math
from std_msgs.msg import String
from nav_msgs.msg import Odometry



class AmygdalaNode:
    def __init__(self):
        # ROS node initialization
        rospy.init_node("amygdala_node", anonymous=True)

        # Subscribers
        self.state_sub = rospy.Subscriber("/current_state", String, self.fear_reference_evaluation)        # From Thalamus
        # self.cortex_input = rospy.Subscriber("", self.update_cortex_input)                               # From Cortex
        
        self.u_amyg = 0
        self.u_cortex = 0
        self.u_eff = 0
        self.previous_u_eff = 0

        self.current_state = None

        self.fear_level = 0
        self.dot_fear_level = 0

        self.wn = 10                                                                                       # Natural frequency
        self.zeta = 0.9                                                                                    # Damp coefficient
        self.alpha = 0.5                                                                                   # Decreasing exponential coefficient

        self.previous_time_instant = rospy.get_rostime().secs 


    def fear_reference_evaluation(self, msg):
        self.current_state = json.loads(msg.data)

        # Extraction of raw information
        # - Number of obstacles
        # - Distances
        # - Velocities
        # - Accelerations


        # Evaluation of Amygdala input
        x = 0
        self.amyg = 1/(1 + math.exp(-x))                                                                    # Or some other type of risk assessment

        # Computation of the actual fear input as sum of both Amygdala and Cortex contributes
        self.u_eff = self.u_amyg + self.u_cortex
        return
    
    def update_cortex_input(self, msg):
        self.u_cortex = msg.u_cortex
        return

    def fear_dynamics(self):
        x1 = self.fear_level
        x2 = self.dot_fear_level

        current_time_instant = rospy.get_rostime().secs 
        dt = current_time_instant - self.previous_time_instant

        # Increasing in the fear reference
        if self.u_eff >= x1:
            # Second order differential equation
            dx1 = x2
            dx2 = -2*self.zeta*self.wn*x2 - (self.wn**2)*self.x1 + (self.wn**2)*self.u_eff

            # Euler integration
            x1 = x1 + dx1*dt
            x2 = x2 + dx2*dt

            self.fear_level = x1
            self.dot_fear_level = x2

            self.previous_time_instant = current_time_instant 

        # Decreasing in the fear reference
        else:
            # First order differential equation
            dx1 = -self.alpha*(x1-self.u_eff)
            
            # Euler integration
            x1 = x1 + dx1*dt
            x2 = dx1

            self.fear_level = x1
            self.dot_fear_level = x2

        

    def spin(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            self.fear_dynamics()
            rate.sleep()

        rospy.loginfo("ROS loop ended.")


if __name__ == "__main__":
    node = AmygdalaNode()
    node.spin()
