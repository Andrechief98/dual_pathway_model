#!/usr/bin/env python3

import rospy
import tf2_ros
import tf2_geometry_msgs 
import numpy as np
import tf.transformations as tft
from geometry_msgs.msg import PoseStamped, Pose, Twist, TransformStamped
from gazebo_msgs.msg import ModelStates
from functools import partial

class OptiTrackerNode:
    def __init__(self, object_names, target_frame="odom", rate_hz=100):
        """
        Initialize the OptitrackerNode that collects objects pose published from the OptiTrack.
        
        :param object_names: List of string with object names on the OptiTrack
        :param rate_hz: Publishing frequency of the ModelStates message
        """
        rospy.init_node('optitracker_node', anonymous=True)
        
        self.object_names = object_names
        self.target_frame = target_frame
        self.rate = rospy.Rate(rate_hz)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.latest_poses = {name: Pose() for name in self.object_names}
        self.model_state_pub = rospy.Publisher('/optitracker/model_states', ModelStates, queue_size=1)
        
        self.subscribers = []
        for name in self.object_names:
            topic_name = f"/vrpn_client_node/{name}/pose"
            rospy.loginfo(f"Subscribed to topic: {topic_name}")
            
            # The optitracker doesn't publish the name of the object. It means that, to create 
            # the Gazebo message, we must use the name from the subscribers list. This avoid the 
            # creation of different callbacks for each subscriber. To do it, it is possible to use 
            # function "partial". This allow to create a function starting from a predefined one but 
            # fixing specific parameters.
            try:
                sub = rospy.Subscriber(
                    topic_name, 
                    PoseStamped, 
                    partial(self.pose_callback, obj_name=name),
                )
                print(f"Subscription to {name} done")
            except Exception as e:
                print(e)
            self.subscribers.append(sub)


    def pose_callback(self, msg, obj_name):
        """It updates the last pose for a specific object."""
        try:
            transformed_pose = self.tf_buffer.transform(msg, "odom", timeout=rospy.Duration(0.1))
            self.latest_poses[obj_name] = transformed_pose.pose

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            pass

    def run(self):
        """Main loop"""
        rospy.loginfo("Optitracker running ...")
        
        while not rospy.is_shutdown():
            msg = ModelStates()
            try:
                for name in self.object_names:
                    msg.name.append(name)
                    msg.pose.append(self.latest_poses[name])
                    msg.twist.append(Twist()) 
                
                # Publishing aggregate message
                self.model_state_pub.publish(msg)
                self.rate.sleep()
            except Exception as e:
                print(e)
if __name__ == '__main__':
    # 'mir' must be in the list to initialize the origin
    objects = ['mir', 'cardboard_box']
    
    try:
        bridge = OptiTrackerNode(objects)
        bridge.run()
    except rospy.ROSInterruptException:
        pass