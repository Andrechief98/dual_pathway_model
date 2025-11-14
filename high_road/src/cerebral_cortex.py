#!/usr/bin/env python3
import rospy
import json
import base64
import os
import asyncio
import logging
import threading
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Empty
import actionlib
from dual_pathway_interfaces.msg import promptProcessingAction, promptProcessingFeedback, promptProcessingResult
from agents import Agent, Runner, SQLiteSession, function_tool
from openai.types.responses import ResponseTextDeltaEvent

script_dir = os.path.dirname(__file__)

class CortexNode:
    def __init__(self):
        rospy.init_node("cortex_node", anonymous=True)

        # Publishers
        self.pub = rospy.Publisher("/cortex_output", String, queue_size=10)
        self.goal_publisher = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)

        # Service clients (create proxies; chiamata vera fatta nel tool)
        self.clear_costmap_client = rospy.ServiceProxy("/move_base/clear_costmaps", Empty)
        self.image_saver = rospy.ServiceProxy('/image_saver/save', Empty)

        # Action server
        self.cerebral_cortex_server = actionlib.SimpleActionServer(
            '/cerebral_cortex_call',
            promptProcessingAction,
            execute_cb=self.cerebral_cortex_callback,
            auto_start=False
        )
        self.cerebral_cortex_server.start()

        # Logging
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger("Cortex")

        @function_tool
        def navigate_to(destination_to_reach: str) -> str:
            """
                It allows to move the robot to a given destination, according to the user request.
                It returns either the success or the failure of the function call.

                Args: 
                    destination_to_reach: name of the destination indicated by the user.
            """
            try:
                cfg_path = os.path.join(script_dir, "..", "config", "info.json")
                with open(cfg_path, "r") as f:
                    info = json.load(f)
            except Exception as e:
                rospy.logerr(f"Failed to load config: {e}")
                return json.dumps({"robot_deployment_success": False, "additional_info": f"Config load error: {e}"})

            areas_dict = info.get("areas", {})
            print(destination_to_reach)
            if destination_to_reach in areas_dict:
                x, y = areas_dict[destination_to_reach]["coordinates"]
                goal_msg = PoseStamped()
                goal_msg.pose.position.x = float(x)
                goal_msg.pose.position.y = float(y)
                goal_msg.pose.position.z = 0.0
                goal_msg.pose.orientation.w = 1.0
                goal_msg.header.stamp = rospy.Time.now()
                goal_msg.header.frame_id = "map"

                try:
                    self.goal_publisher.publish(goal_msg)
                    return json.dumps({"robot_deployment_success": True, "additional_info": None})
                except Exception as e:
                    rospy.logerr(e)
                    return json.dumps({"robot_deployment_success": False, "additional_info": str(e)})
            else:
                return json.dumps({"robot_deployment_success": False, "additional_info": "Destination not present in the file."})

        @function_tool
        def read_file() -> str:
            """
                It allows to read the entire file to obtain the information about areas in the environment.
            """
            try:
                cfg_path = os.path.join(script_dir, "..", "config", "info.json")
                with open(cfg_path, "r") as f:
                    info = json.load(f)
                rospy.loginfo("File loaded successfully")
                return json.dumps(info)
            except Exception as e:
                rospy.logerr(f"read_file error: {e}")
                return json.dumps({"error": str(e)})

        @function_tool
        def take_image() -> str:
            """
                
            """
            # questo è un tool sincrono che fa operazioni bloccanti: meglio che il caller lo richiami in executor
            try:
                rospy.wait_for_service('/image_saver/save', timeout=5.0)
            except Exception as e:
                rospy.logerr(f"wait_for_service failed: {e}")
                return f"Image saver service not available: {e}"

            try:
                self.image_saver()
                rospy.loginfo("Image saved successfully!")
            except rospy.ServiceException as e:
                rospy.logerr(f"Image saver service failed: {e}")
                return f"Image saver service failed: {e}"

            try:
                path = os.path.join(script_dir, "images", "current_situation.jpg")
                with open(path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode('utf-8')
                return img_b64
            except Exception as e:
                rospy.logerr(f"Failed to send image: {e}")
                return f"Failed to send image: {e}"



        # Agent & session
        self.agent = Agent(
            name="Corteccia",
            instructions=("You are the cerebral cortex of a mobile robot. "
                          "You must use your tools to obtain information about the environment, move the robot and see the environment through images."),
            tools=[
                navigate_to, 
                read_file, 
                take_image
                ]
        )
        self.session = SQLiteSession("prova")

        # asyncio loop in separate thread
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.loop_thread.start()

        # ensure clean shutdown
        rospy.on_shutdown(self._on_shutdown)



    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def cerebral_cortex_callback(self, goal):
        rospy.loginfo("Received cerebral cortex goal")

        # lancia il coroutine sulla loop thread e ottieni il future
        future = asyncio.run_coroutine_threadsafe(self._process_goal(goal), self.loop)

        try:
            # attendi che il coroutine finisca.
            # se vuoi evitare hang infiniti, puoi usare un timeout: future.result(timeout=60)
            future.result()
        except Exception as e:
            rospy.logerr(f"Error while processing goal (in future.result): {e}")
            # prova ad abortire il goal se non è già stato terminato
            try:
                res = promptProcessingResult()
                res.output_text = f"Exception: {e}"
                self.cerebral_cortex_server.set_aborted(res)
            except Exception as e2:
                rospy.logerr(f"Also failed to abort goal cleanly: {e2}")


    async def _process_goal(self, goal):
        feedback = promptProcessingFeedback()
        result = promptProcessingResult()
        input_text = goal.input_text

        try:
            # run_streamed può essere coroutine oppure sync; gestiamo entrambi i casi
            response = Runner.run_streamed(self.agent, input_text, session=self.session)
            if asyncio.iscoroutine(response):
                response = await response

            rospy.loginfo("> AI streaming started")

            # se il response espone stream_events() come async generator
            async for event in response.stream_events():
                # controllo preemption (se il client ha cancellato)
                if self.cerebral_cortex_server.is_preempt_requested():
                    rospy.loginfo("Preempt requested: aborting")
                    result.output_text = "Preempted"
                    self.cerebral_cortex_server.set_preempted(result)
                    return

                # gestione tipi di evento (adatta ai tipi del tuo Runner)
                if getattr(event, "type", None) == "raw_response_event" and isinstance(getattr(event, "data", None), ResponseTextDeltaEvent):
                    delta = event.data.delta
                    # invia feedback al client actionlib
                    feedback.token = delta
                    self.cerebral_cortex_server.publish_feedback(feedback)


                elif getattr(event, "type", None) == "tool_call_event":
                    rospy.loginfo(f"[TOOL CALL] {event.data}")
                elif getattr(event, "type", None) == "tool_result_event":
                    rospy.loginfo(f"[TOOL RESULT] {event.data}")
                else:
                    # fallback log
                    rospy.logdebug(f"Event: {event}")

            rospy.loginfo("AI > streaming finished")
            result.output_text = "Finished"
            self.cerebral_cortex_server.set_succeeded(result)
        except Exception as e:
            rospy.logerr(f"Error in goal processing: {e}")
            result.output_text = f"Error: {e}"
            # se non è già stato terminato
            try:
                self.cerebral_cortex_server.set_aborted(result)
            except Exception:
                pass

    def spin(self):
        rospy.loginfo("CortexNode spinning...")
        rospy.spin()
        rospy.loginfo("ROS loop ended. Stopping asyncio loop...")
        self._on_shutdown()

    def _on_shutdown(self):
        # stop event loop e join del thread
        try:
            if self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)
            if self.loop_thread.is_alive():
                self.loop_thread.join(timeout=2.0)
        except Exception as e:
            rospy.logwarn(f"Exception on shutdown: {e}")

if __name__ == "__main__":
    node = CortexNode()
    try:
        node.spin()
    except rospy.ROSInterruptException:
        pass
