#!/usr/bin/env python3

import rospy
import tf2_ros
import numpy as np
import tf.transformations as tft
from geometry_msgs.msg import Pose, Twist, TransformStamped
from gazebo_msgs.msg import ModelStates
import math

class FakeOptiTrackerNode:
    def __init__(self, object_names, reference_name="mir", rate_hz=100):
        """
        Initialize the FakeOptiTrackerNode that generates synthetic poses.
        
        :param object_names: List of string with object names
        :param rate_hz: Publishing frequency of the ModelStates message
        """
        rospy.init_node('optitracker_node', anonymous=True)
        
        self.object_names = object_names
        self.reference_name = reference_name
        self.rate = rospy.Rate(rate_hz)
        
        # Origin management
        self.origin_matrix_inv = None
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster()
        
        # Dictionary for last pose memorization
        self.latest_poses = {name: Pose() for name in self.object_names}
        
        # Publisher for the final topic
        self.model_state_pub = rospy.Publisher(
            '/optitracker/model_states', 
            ModelStates, 
            queue_size=1
        )

        # In Simulation, we define the fake "OptiTrack World" origin immediately
        self._set_fake_origin()

    def _set_fake_origin(self):
        """Simulates the first capture of the reference robot to lock the origin."""
        # Assume the robot 'mir' is at (1.0, 1.0, 0.0) in the OptiTrack volume at startup
        self.fake_start_x = 1.0
        self.fake_start_y = 1.0
        
        initial_matrix = tft.translation_matrix([self.fake_start_x, self.fake_start_y, 0.0])
        self.origin_matrix_inv = tft.inverse_matrix(initial_matrix)
        
        # Broadcast static TF (map is now at the robot's startup position)
        static_tf = TransformStamped()
        static_tf.header.stamp = rospy.Time.now()
        static_tf.header.frame_id = "optitrack_world"
        static_tf.child_frame_id = "map"
        static_tf.transform.translation.x = self.fake_start_x
        static_tf.transform.translation.y = self.fake_start_y
        static_tf.transform.translation.z = 0.0
        static_tf.transform.rotation.w = 1.0
        self.static_broadcaster.sendTransform(static_tf)
        
        rospy.loginfo(f"Fake origin set. '{self.reference_name}' will be at (0,0,0) in 'map' frame.")

    def _matrix_to_pose(self, matrix):
        """Helper to convert 4x4 matrix to geometry_msgs/Pose"""
        pose = Pose()
        trans = tft.translation_from_matrix(matrix)
        quat = tft.quaternion_from_matrix(matrix)
        pose.position.x, pose.position.y, pose.position.z = trans
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat
        return pose

    def update_synthetic_data(self):
        """Updates poses: reference is static, other objects are moving."""
        t = rospy.get_time()
        
        for name in self.object_names:
            if name == self.reference_name:
                # MIR stays exactly where it started (Result: relative pose will be 0,0,0)
                raw_matrix = tft.translation_matrix([self.fake_start_x, self.fake_start_y, 0.0])
            else:
                # Obstacles move in a circle relative to the OptiTrack volume
                # Center (2.0, 2.0), Radius 0.8m
                cx, cy = 2.0, 2.0
                r = 0.8
                freq = 0.5 # rad/s
                
                x = cx + r * math.cos(freq * t)
                y = cy + r * math.sin(freq * t)
                raw_matrix = tft.translation_matrix([x, y, 0.0])
            
            # Apply relative transformation
            relative_matrix = np.dot(self.origin_matrix_inv, raw_matrix)
            self.latest_poses[name] = self._matrix_to_pose(relative_matrix)

    def run(self):
        """Main loop"""
        rospy.loginfo("Fake Optitracker running... [MIR: STATIC | OTHERS: MOVING]")
        
        while not rospy.is_shutdown():
            # Generate fake coordinates
            self.update_synthetic_data()

            # Prepare ModelStates message
            msg = ModelStates()
            for name in self.object_names:
                msg.name.append(name)
                msg.pose.append(self.latest_poses[name])
                msg.twist.append(Twist()) 
            
            # Publishing aggregate message
            self.model_state_pub.publish(msg)
            self.rate.sleep()

if __name__ == '__main__':
    # Objects to simulate
    objects = ['mir', 'rover', 'person', 'cardboard_box']
    
    try:
        bridge = FakeOptiTrackerNode(objects, reference_name='mir', rate_hz=100)
        bridge.run()
    except rospy.ROSInterruptException:
        pass