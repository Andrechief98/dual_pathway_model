#!/usr/bin/env python3
import rospy
import json
import math
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from mpc_planner.msg import mpcParameters
from ollama import chat
from ollama import ChatResponse
import yaml



class AmygdalaNode:
    def __init__(self):
        # ROS node initialization
        rospy.init_node("amygdala_node", anonymous=True)

        # Subscribers
        self.object_state_sub = rospy.Subscriber("/thalamus/info", String, self.update_low_road_risk)         # From Thalamus (low road information)
        self.cortex_input = rospy.Subscriber("/vlm/image/description", String, self.update_high_road_risk)                                # From Cortex (high road information)
        
        # Publishers
        self.low_road_risks_pub = rospy.Publisher("/amygdala/lowroad/risks", String, queue_size=1)
        self.fear_level_pub = rospy.Publisher("/fearlevel", String, queue_size=1)
        self.max_risk_pub = rospy.Publisher("/maxrisk", String, queue_size=1)

        self.flag = True

        self.u_low_road = 0
        self.u_high_road = 0

        self.u_eff = 0
        self.previous_u_eff = 0


        self.fear_level = 0
        self.dot_fear_level = 0

        # Fear dynamics parameters
        self.wn = 10                                                                                       # Natural frequency
        self.zeta = 0.9                                                                                    # Damp coefficient
        self.alpha = 0.5                                                                                   # Decreasing exponential coefficient

        self.previous_time_instant = rospy.get_time()


    def update_low_road_risk(self, msg):
        
        thalamus_info = json.loads(msg.data)

        # Extraction of relative information from the incoming message
        # - Distances
        # - Velocities
        # - Accelerations

        rel_thalamus_info = thalamus_info["relative_info"]
        # print(rel_thalamus_info)
        
        
        object_risk_levels = {}

        if rel_thalamus_info!={}:
            for object in rel_thalamus_info.keys():
                # print(object)

                rel_dist = rel_thalamus_info[object]["relative_dist"]
                rel_orient = rel_thalamus_info[object]["relative_orient"]
                rel_rad_vel = rel_thalamus_info[object]["radial_vel"]

                # print(rel_dist)
                # print(f"Obstacle radial vel: {rel_rad_vel}")

                # Gaussian 
                mu_d = 0
                sigma_d = 2
                rel_dist_risk = math.exp(-0.5 * ((rel_dist - mu_d)/sigma_d)**2)
                
                # Logistic
                k2 = 0.5;         # Slope 
                v0 = -1;          # Neutral value (0.5 in rel_lin_vel = -1)
                # rel_vel_risk = 1 / (1 + math.exp(-k2 * (-rel_rad_vel - v0))); 

                object_risk_levels[object] = round(rel_dist_risk,3)


            # Verify that it works for a single object
            object_name_list = list(rel_thalamus_info.keys())
            first_name = object_name_list[0]
            self.u_low_road = object_risk_levels[first_name] 
            # self.u_low_road = max(list(object_risk_levels.values()))
        else:
            # We don't have any object
            self.u_low_road = 0
        
        # For plotting 
        self.low_road_risks_pub.publish(json.dumps(object_risk_levels))
        return
    
    def update_high_road_risk(self, msg):
        image_description = msg.data

        # TO DO: add the conversation management
        response: ChatResponse = chat(
            model='gemma3:270m', 
            messages=[
            {
                'role': 'user',
                'content': image_description,
            },
            ]
        )

        # Response of the Small Language Model
        message_content = response.message.content
        # print(message_content)

        # Example of small Language Model output:
        if self.flag :
            message_content_ex = {
                "risk": 1
            }

            self.flag  = False
        else:
            message_content_ex = {
                "risk": 0
            }

            self.flag  = True
        
        self.u_high_road = message_content_ex["risk"]
        return
    

    def fear_dynamics(self):
        # Computation of the actual risk input as the mean of both low-road and high-road contributes
        self.u_eff = (self.u_low_road + self.u_high_road)/2
        # print(self.u_eff)
        # self.u_eff = self.u_low_road

        x1 = self.fear_level
        x2 = self.dot_fear_level

        current_time_instant = rospy.get_time()
        dt = current_time_instant - self.previous_time_instant

        if dt <= 0:
            return

        # Increasing in the fear reference
        if self.u_eff >= x1:
            # Second order differential equation
            dx1 = x2
            dx2 = -2*self.zeta*self.wn*x2 - (self.wn**2)*x1 + (self.wn**2)*self.u_eff

            # Euler integration
            x1 = x1 + dx1*dt
            x2 = x2 + dx2*dt

        # Decreasing in the fear reference
        else:
            # First order differential equation
            dx1 = -self.alpha*(x1-self.u_eff)
            
            # Euler integration
            x1 = x1 + dx1*dt
            x2 = dx1

        self.fear_level = max(0, min(1.2, round(x1,3)))
        self.dot_fear_level = round(x2,3)
        self.previous_time_instant = current_time_instant 

        # print(max(self.fear_level,0))

        self.fear_level_pub.publish(str(self.fear_level))
        return

        

    def spin(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.fear_dynamics()
            rate.sleep()

        rospy.loginfo("ROS loop ended.")


if __name__ == "__main__":
    node = AmygdalaNode()
    node.spin()
