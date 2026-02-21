#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Point
import random

def noise_publisher():
    rospy.init_node('gaussian_noise_injection', anonymous=True)
    
    rob_pub = rospy.Publisher('/robot/gaussian_noise', Point, queue_size=10)
    obs_pub = rospy.Publisher('/obstacles/gaussian_noise', Point, queue_size=10)

    robot_mean = rospy.get_param('/robot_gaussian_mean_noise', 0.0)
    robot_stddev = rospy.get_param('/robot_gaussian_std_dev_noise', 0.02)
    obs_mean = rospy.get_param('/obstacles_gaussian_mean_noise', 0.0)
    obs_stddev = rospy.get_param('/obstacles_gaussian_std_dev_noise', 0.02)

    rate = rospy.Rate(10)

    rospy.loginfo(f"Gaussian noise injector run")

    while not rospy.is_shutdown():
        robot_noise_msg = Point()
        
        robot_noise_msg.x = random.gauss(robot_mean, robot_stddev)
        robot_noise_msg.y = random.gauss(robot_mean, robot_stddev)
        robot_noise_msg.z = random.gauss(robot_mean, robot_stddev)  # used as noise for robot theta
        
        rob_pub.publish(robot_noise_msg)
        
        obs_noise_msg = Point()
        
        obs_noise_msg.x = random.gauss(obs_mean, obs_stddev)
        obs_noise_msg.y = random.gauss(obs_mean, obs_stddev)
        obs_noise_msg.z = 0.0
        
        obs_pub.publish(obs_noise_msg)
        
        rate.sleep()

if __name__ == '__main__':
    try:
        noise_publisher()
    except rospy.ROSInterruptException:
        pass