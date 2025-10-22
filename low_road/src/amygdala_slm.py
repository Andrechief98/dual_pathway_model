#!/usr/bin/env python3
import rospy
import json
from std_msgs.msg import String
from ollama import chat
from ollama import ChatResponse
from nav_msgs.msg import Odometry
from low_road.srv import StringExchange, StringExchangeResponse
from low_road.msg import promptProcessingAction, promptProcessingFeedback, promptProcessingResult, promptProcessingGoal

class AmygdalaNode:
    def __init__(self):
        # ROS node initialization
        rospy.init_node("amygdala_node", anonymous=True)

        # Subscribers
        self.state_sub = rospy.Subscriber("/current_state", String, self.update_status_callback)

        self.current_state = None


    def update_status_callback(self, msg):
        self.current_state = json.loads(msg.data)
        rospy.loginfo("update fatto")
        return
    
    def fear_evaluation(self):
        # response: ChatResponse = chat(
        #     model='gemma3:270m', 
        #     messages=[
        #     {
        #         'role': 'user',
        #         'content': 'Why is the sky blue?',
        #     },
        #     ]
        # )
        rospy.loginfo("fear evaluated")
        return
        


    def spin(self):
        rate = rospy.Rate(2)
        while not rospy.is_shutdown():
            self.fear_evaluation()
            rate.sleep()

        rospy.loginfo("ROS loop ended.")


if __name__ == "__main__":
    node = AmygdalaNode()
    node.spin()
