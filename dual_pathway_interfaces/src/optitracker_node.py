#!/usr/bin/env python3

import rospy
import tf2_ros
import tf2_geometry_msgs 
import numpy as np
import tf.transformations as tft
from geometry_msgs.msg import PoseStamped, Pose, Twist, TransformStamped
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
from functools import partial

class OptiTrackerNode:
    def __init__(self, object_names, target_frame="odom", rate_hz=20):
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
        self.mir_prev_pose = None
        self.mir_prev_time = None
        self.mir_twist = Twist()

        self.model_state_pub = rospy.Publisher('/optitracker/model_states', ModelStates, queue_size=1)
        # self.robot_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        self.subscribers = []
        for name in self.object_names:
            # if name == "mir":
            #     continue
            # else:
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


    def odom_callback(self, msg):
        self.latest_poses["mir"] = msg.pose.pose

    def pose_callback(self, msg, obj_name):
        """It updates the last pose for a specific object."""

        try:
            transformed_pose = self.tf_buffer.transform(msg, "odom", timeout=rospy.Duration(0.1))
            curr_pose = transformed_pose.pose
            curr_time = msg.header.stamp

            if obj_name == "mir":
                if self.mir_prev_pose is not None and self.mir_prev_time is not None:
                    dt = (curr_time - self.mir_prev_time).to_sec()
                    if dt > 0:
                        # 1. Linear Velocity: (curr_pos - prev_pos) / dt
                        self.mir_twist.linear.x = (curr_pose.position.x - self.mir_prev_pose.position.x) / dt
                        self.mir_twist.linear.y = (curr_pose.position.y - self.mir_prev_pose.position.y) / dt
                        self.mir_twist.linear.z = (curr_pose.position.z - self.mir_prev_pose.position.z) / dt

                        # # 2. Angular Velocity: Use tf.transformations for rotation differentiation
                        q_prev = [self.mir_prev_pose.orientation.x, self.mir_prev_pose.orientation.y, 
                                  self.mir_prev_pose.orientation.z, self.mir_prev_pose.orientation.w]
                        q_curr = [curr_pose.orientation.x, curr_pose.orientation.y, 
                                  curr_pose.orientation.z, curr_pose.orientation.w]
                        
                        # Find relative rotation: q_rel = q_curr * inverse(q_prev)
                        q_rel = tft.quaternion_multiply(q_curr, tft.quaternion_inverse(q_prev))
                        # Convert to axis-angle (approximate for small dt)
                        angle, axis = self.quaternion_to_axis_angle(q_rel)
                        self.mir_twist.angular.x = (axis[0] * angle) / dt
                        self.mir_twist.angular.y = (axis[1] * angle) / dt
                        self.mir_twist.angular.z = (axis[2] * angle) / dt

                self.mir_prev_pose = curr_pose
                self.mir_prev_time = curr_time

            self.latest_poses[obj_name] = transformed_pose.pose

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            pass

    def quaternion_to_axis_angle(self, q):
        """Robustly extract angle and axis from a quaternion [x, y, z, w]."""
        # Clamp w to avoid sqrt of negative or acos of > 1 due to precision
        w = np.clip(q[3], -1.0, 1.0)
        
        angle = 2 * np.arccos(w)
        
        # Use a small epsilon to avoid division by zero
        sin_half_angle = np.sqrt(max(0, 1.0 - w**2))
        
        if sin_half_angle < 1e-6:
            # If angle is near zero, axis is arbitrary
            axis = np.array([0.0, 0.0, 0.0])
        else:
            axis = np.array([q[0], q[1], q[2]]) / sin_half_angle
            
        return angle, axis
            
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
    # objects = ['mir', 'rover', 'cardboard_box', 'person']
    
    objects = ['mir', 'rover','person', 'cardboard_box']
    try:
        bridge = OptiTrackerNode(objects)
        bridge.run()
    except rospy.ROSInterruptException:
        pass