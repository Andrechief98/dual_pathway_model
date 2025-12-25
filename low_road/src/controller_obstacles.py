#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist

def controller_obstacles():

    rospy.init_node('controller_obstacles', anonymous=True)

    topics_list = [
        {
            "topic": "/person_walking/cmd_vel", 
            "vx": 0.3, 
            "vy": 0.0
        },
        {
            "topic": "/rover/cmd_vel", 
            "vx": 0.3, 
            "vy": 0.0
        }
    ]

    controllers = []
    
    for item in topics_list:
        # Publisher for each topic in the list
        pub = rospy.Publisher(item["topic"], Twist, queue_size=10)
        

        msg = Twist()
        msg.linear.x = item["vy"]
        msg.linear.y = - item["vx"]
        msg.linear.z = 0.0
        msg.angular.x = 0.0 
        msg.angular.y = 0.0 
        msg.angular.z = 0.0 
        
        controllers.append((pub, msg))

    rate = rospy.Rate(10) 

    while not rospy.is_shutdown():
        for pub, msg in controllers:
            pub.publish(msg)
        
        rate.sleep()

if __name__ == '__main__':
    try:
        controller_obstacles()
    except rospy.ROSInterruptException:
        pass