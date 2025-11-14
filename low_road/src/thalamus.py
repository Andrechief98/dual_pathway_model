#!/usr/bin/env python3
import rospy
import threading
import actionlib
import json
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from dual_pathway_interfaces.srv import highRoadInfo, highRoadInfoResponse
from dual_pathway_interfaces.msg import promptProcessingAction, promptProcessingFeedback, promptProcessingResult, promptProcessingGoal
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import math


class ThalamusNode:
    def __init__(self):
        # ROS node initialization
        rospy.init_node("thalamus_node", anonymous=True)

        self.current_state = {
            "pose": {
                "position":[],
                "orientation":[]
            },
            "twist":{
                "linear":[],
                "angular":[]
            }
        }


        self.object_state = {}
        self.relative_info = {}

        # Publishers 
        self.current_state_pub = rospy.Publisher("/current_state", String, queue_size=1)
        self.thalamus_info_pub = rospy.Publisher("/thalamus/info", String, queue_size=1)

        # Subscribers
        self.odom_sub = rospy.Subscriber("/odometry/filtered", Odometry, self.odom_feedback_callback)
        self.gazebo_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.thalamus_info_callback)

        # Server
        self.camera_image_server = rospy.Service('/get_camera_image/', highRoadInfo, self.getCameraImage)

        # Action Clients
        self.cerebralCortexClient =  actionlib.SimpleActionClient("/cerebral_cortex_call", promptProcessingAction)

        # --- Variabile di stato (thread-safe) ---
        self.running = True
        self.lock = threading.Lock()

        # --- Thread separato per input utente ---
        self.input_thread = threading.Thread(target=self.user_input_loop)
        self.input_thread.daemon = True  # termina quando il nodo chiude
        self.input_thread.start()

        self.relevant_object_list = [] # To update the list of object within robot's fields of view


    def odom_feedback_callback(self, msg):
        
        odometry_filtered = msg

        self.current_state["pose"]["position"] = [
            odometry_filtered.pose.pose.position.x,
            odometry_filtered.pose.pose.position.y,
            odometry_filtered.pose.pose.position.z
            ]
        
        self.current_state["pose"]["orientation"] = [
            odometry_filtered.pose.pose.orientation.x,
            odometry_filtered.pose.pose.orientation.y,
            odometry_filtered.pose.pose.orientation.z,
            odometry_filtered.pose.pose.orientation.w,
            ]
        
        self.current_state["twist"]["linear"] = [
            odometry_filtered.twist.twist.linear.x,
            odometry_filtered.twist.twist.linear.y,
            odometry_filtered.twist.twist.linear.z
            ]
        
        self.current_state["twist"]["angular"] = [
            odometry_filtered.twist.twist.angular.x,
            odometry_filtered.twist.twist.angular.y,
            odometry_filtered.twist.twist.angular.z
            ]

        self.current_state_pub.publish(json.dumps(self.current_state))
        return
    
    def getCameraImage(self, req):
        response = highRoadInfoResponse()
        response.frame = rospy.wait_for_message("/camera/color/image_raw", Image)

        relevant_info_dict = {}

        for object in self.relevant_object_list:
            relevant_info_dict[object] = self.relative_info[object]

        response.relevant_info = json.dumps(relevant_info_dict)
        return response
        

    def thalamus_info_callback(self, msg):
        # Structure of the Gazebo message ModelStates
        #   string[] name
        #   geometry_msgs/Pose[] pose
        #   geometry_msgs/Twist[] twist 
        
        gazebo_msg = msg

        name_list = gazebo_msg.name
        pose_list = gazebo_msg.pose

        neglecting_name_list = ["ground_plane", "walls"]

        

        # Information extraction
        for name, pose in zip(name_list, pose_list):
            
            if name in neglecting_name_list:
                continue
            else:
                if name not in self.object_state.keys():
                    # We have a new object. We add all the initial information and set the initial velocity to zero
                    current_time = rospy.Time.now().to_sec()
                    current_pos = [pose.position.x, pose.position.y, pose.position.z] 
                    quaternion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
                    roll_x, pitch_y, yaw_z = euler_from_quaternion(quaternion)

                    self.object_state[name] = {
                        "current_time": current_time, 
                        "current_pos": current_pos,
                        "current_orient": yaw_z,
                        "current_lin_vel": [0.0, 0.0],
                        "current_ang_vel": 0.0,
                        "current_lin_acc": [0.0, 0.0]
                    }


                else:
                    # print(f"Name: {name}")
                    # We extract the previous information to compute the velocity and acceleration
                    previous_time = self.object_state[name]["current_time"]
                    previous_pos = self.object_state[name]["current_pos"]
                    previous_orient = self.object_state[name]["current_orient"]
                    previous_lin_vel = self.object_state[name]["current_lin_vel"]

                    # We compute the dt
                    current_time = rospy.Time.now().to_sec()
                    dt = current_time - previous_time

                    if dt <= 0:
                        dt = 0.001

                    # We extract current information from Gazebo msg
                    current_pos = [pose.position.x, pose.position.y, pose.position.z] 
                    quaternion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
                    _, _, current_orient = euler_from_quaternion(quaternion) # roll_x, pitch_y, yaw_z 

                    # We compute the current linear andangular velocity
                    dx = current_pos[0] - previous_pos[0]
                    dy = current_pos[1] - previous_pos[1]
                    current_lin_vel = [round(dx/dt,3), round(dy/dt,3)]
                    # print(f"Ostacle vel: {current_lin_vel}")

                    current_ang_vel = (current_orient - previous_orient)/dt

                    # We compute the current linear acceleration
                    dvx = current_lin_vel[0] - previous_lin_vel[0]
                    dvy = current_lin_vel[1] - previous_lin_vel[1]
                    current_lin_acc = [dvx/dt, dvy/dt]

                    self.object_state[name] = {
                        "current_time": current_time, 
                        "current_pos": current_pos,
                        "current_orient": current_orient,
                        "current_lin_vel": current_lin_vel,
                        "current_ang_vel": current_ang_vel,
                        "current_lin_acc": current_lin_acc
                    }


        self.relative_info = {}
        self.relevant_object_list = []

        for name in self.object_state.keys():
            if name == "mir":
                continue
            
            else:

                robot_pos = self.object_state["mir"]["current_pos"]
                obj_pos = self.object_state[name]["current_pos"]

                robot_orient = self.object_state["mir"]["current_orient"]
                obj_orient = self.object_state[name]["current_orient"]

                robot_lin_vel = self.object_state["mir"]["current_lin_vel"]
                obj_lin_vel = self.object_state[name]["current_lin_vel"]

                rel_x = obj_pos[0] - robot_pos[0]
                rel_y = obj_pos[1] - robot_pos[1]

                angle_to_object = math.atan2(rel_y, rel_x)

                angle_diff = angle_to_object - robot_orient

                angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))

                relative_dist = np.array([
                    obj_pos[0] - robot_pos[0],
                    obj_pos[1] - robot_pos[1]
                ])

                dist_norm = np.linalg.norm(relative_dist)

                relative_orient = obj_orient - robot_orient

                relative_lin_vel = np.array([
                    obj_lin_vel[0] - robot_lin_vel[0],
                    obj_lin_vel[1] - robot_lin_vel[1]
                ])
                                    

                if dist_norm < 1e-3: 
                    dist_norm = 1e-3

                # Velocità radiale = proiezione scalare
                v_rad = round(np.dot(obj_lin_vel, relative_dist)/dist_norm,3)
                # print(f"Obstacle radial velocity: {v_rad}")
                

                # We save the object
                self.relative_info[name] = {
                    "relative_dist" : dist_norm,
                    "relative_orient" : relative_orient,
                    "radial_vel" : v_rad,
                }

                if abs(angle_diff) <= math.radians(90):   # 180° Field Of View
                    # The object is within robot's field of view
                    self.relevant_object_list.append(name)
                




        # Collecting both global and relative information
        self.thalamus_info = {
            "object_state": self.object_state,
            "relative_info": self.relative_info
        }
        self.thalamus_info_pub.publish(json.dumps(self.thalamus_info))
        return
    



    def action_feedback_cb(self, feedback):
        try:
            token = feedback.token
            print(token, end="", flush=True)
        except Exception as e:
            rospy.logwarn(f"Error in the received feedback: {e}")


    def user_input_loop(self):
        """
            Separated thread to read user input asynchronously from ROS data managing. The user input is sent to the cerebral cortex as Action
            Comandi utente:
            - /bye     -> esci e chiudi nodo
        """
        while not rospy.is_shutdown() and self.running:
            try:
                # success = self.cerebralCortexClient.wait_for_server()

                user_input = input("User >> ")  # lettura bloccante, ma in thread separato
                
                if user_input.lower() == "/bye":
                    rospy.loginfo("Shutting down by user command.")
                    with self.lock:
                        self.running = False
                    rospy.signal_shutdown("User requested exit")
                    break
                else:

                    goal = promptProcessingGoal()

                    goal.input_text = user_input
                    print("AI >> ", end="", flush=True)
                    self.cerebralCortexClient.send_goal(goal, feedback_cb = self.action_feedback_cb)
                    self.cerebralCortexClient.wait_for_result()
                    result = self.cerebralCortexClient.get_result()
                    print()

            except EOFError:
                break
            except Exception as e:
                rospy.logerr(f"Error in input loop: {e}")


    def spin(self):
        rospy.spin()
        rospy.loginfo("ROS loop ended.")


if __name__ == "__main__":
    node = ThalamusNode()
    node.spin()
