#!/usr/bin/env python3

import rospy
import tf2_ros
import numpy as np
import tf.transformations as tft
from geometry_msgs.msg import PoseStamped, Pose, Twist, TransformStamped
from gazebo_msgs.msg import ModelStates
from functools import partial

class OptiTrackerNode:
    def __init__(self, object_names, reference_name="mir", rate_hz=100):
        """
        Initialize the OptitrackerNode that collects objects pose published from the OptiTrack.
        
        :param object_names: List of string with object names on the OptiTrack
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
            '/optitracker_node/model_states', 
            ModelStates, 
            queue_size=1
        )
        
        # Dynamic subscribers creation
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

    def _pose_to_matrix(self, pose):
        """Helper to convert geometry_msgs/Pose to 4x4 matrix"""
        translation = [pose.position.x, pose.position.y, pose.position.z]
        rotation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        return tft.concatenate_matrices(tft.translation_matrix(translation), tft.quaternion_matrix(rotation))

    def _matrix_to_pose(self, matrix):
        """Helper to convert 4x4 matrix to geometry_msgs/Pose"""
        pose = Pose()
        trans = tft.translation_from_matrix(matrix)
        quat = tft.quaternion_from_matrix(matrix)
        pose.position.x, pose.position.y, pose.position.z = trans
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat
        return pose

    def pose_callback(self, msg, obj_name):
        """It updates the last pose for a specific object."""
        
        # If the reference object ("mir") is received for the first time, set the origin
        if obj_name == self.reference_name and self.origin_matrix_inv is None:
            initial_matrix = self._pose_to_matrix(msg.pose)
            self.origin_matrix_inv = tft.inverse_matrix(initial_matrix)
            
            # Broadcast the static transform to TF for visualization
            static_tf = TransformStamped()
            static_tf.header.stamp = rospy.Time.now()
            static_tf.header.frame_id = "optitrack_world"
            static_tf.child_frame_id = "mir_initial_pose"
            static_tf.transform.translation.x = msg.pose.position.x
            static_tf.transform.translation.y = msg.pose.position.y
            static_tf.transform.translation.z = msg.pose.position.z
            static_tf.transform.rotation = msg.pose.orientation
            self.static_broadcaster.sendTransform(static_tf)
            
            rospy.loginfo(f"Origin initialized using {obj_name} first pose.")

        # Apply transformation if origin is set
        if self.origin_matrix_inv is not None:
            current_matrix = self._pose_to_matrix(msg.pose)
            # Relative pose: T_rel = T_origin_inv * T_current
            relative_matrix = np.dot(self.origin_matrix_inv, current_matrix)
            self.latest_poses[obj_name] = self._matrix_to_pose(relative_matrix)
        else:
            # Until "mir" is found, we store the raw pose
            self.latest_poses[obj_name] = msg.pose

    def run(self):
        """Main loop"""
        rospy.loginfo("Optitracker running ...")
        
        while not rospy.is_shutdown():
            # We wait until the origin is set by the reference robot
            if self.origin_matrix_inv is None:
                self.rate.sleep()
                continue

            # We use the Gazebo msg: ModelStates
            msg = ModelStates()
            
            for name in self.object_names:
                msg.name.append(name)
                msg.pose.append(self.latest_poses[name])
                
                # Twist info set to 0
                msg.twist.append(Twist()) 
            
            # Publishing aggregate message
            self.model_state_pub.publish(msg)
            
            self.rate.sleep()

if __name__ == '__main__':
    # 'mir' must be in the list to initialize the origin
    objects = ['mir', 'test']
    
    try:
        bridge = OptiTrackerNode(objects, reference_name='mir', rate_hz=100)
        bridge.run()
    except rospy.ROSInterruptException:
        pass