#!/usr/bin/env python3
import rospy
import json
from std_msgs.msg import String
from ollama import chat
from ollama import ChatResponse


class AmygdalaNode:
    def __init__(self):
        # ROS node initialization
        rospy.init_node("amygdala_node", anonymous=True)

        # Subscribers

    
    def fear_evaluation(self):
        response: ChatResponse = chat(
            model='gemma3:270m', 
            messages=[
            {
                'role': 'user',
                'content': 'Why is the sky blue?',
            },
            ]
        )
        message_content = response.message.content
        print(message_content)
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
