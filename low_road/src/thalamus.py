#!/usr/bin/env python3
import rospy
import threading
import actionlib
import json
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from low_road.srv import StringExchange, StringExchangeResponse
from low_road.msg import promptProcessingAction, promptProcessingFeedback, promptProcessingResult, promptProcessingGoal

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

        # Publishers 
        self.current_state_pub = rospy.Publisher("/current_state", String, queue_size=10)

        # Subscribers
        self.odom_sub = rospy.Subscriber("/odometry/filtered", Odometry, self.odom_feedback_callback)

        # Action Clients
        self.cerebralCortexClient =  actionlib.SimpleActionClient("/cerebral_cortex_call", promptProcessingAction)

        # --- Variabile di stato (thread-safe) ---
        self.running = True
        self.lock = threading.Lock()

        # --- Thread separato per input utente ---
        self.input_thread = threading.Thread(target=self.user_input_loop)
        self.input_thread.daemon = True  # termina quando il nodo chiude
        self.input_thread.start()


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
