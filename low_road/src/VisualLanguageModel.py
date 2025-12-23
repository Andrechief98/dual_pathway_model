#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PIL import Image as PILImage
import cv2
from dual_pathway_interfaces.srv import highRoadInfo, highRoadInfoRequest
from ollama import chat
import time
from typing import List
from pydantic import BaseModel, Field
import base64
from io import BytesIO
import json

SYSTEM_PROMPT = """
        [SYSTEM]
        You are a vision model expert in safety and risk analysis. 
        Your goal is to evaluate the dangerousness associated to each object in the environment using a float value ranging from 0.00 to 1.0.

        [INPUT DATA]
        1) Current image (time instant t): represents what you see. 
        2) Spatial information (time instant t): Relative Distance (m), relative Angle (rad), and Radial Velocity (m/s) for each object.
        3) Memory (time instant t-1): Previous danger scores and spatial data for each object.

                
        [RISK EVALUATION RULES]
        - INTRINSIC BASELINE RISKS: Every object starts with a base risk based on its nature.
        - TEMPORAL EVOLUTION: If Distance(t) < Distance(t-1) or radial velocity is positive, the object is 'Closing In'. Risk must increase accordingly.
        - POTENTIAL CONSEQUENCES: you must consider which could be the possible consequences related to each object during the risk evaluation.
        - UNCERTAINTY: If you are uncertain about the possible consequences, it is better to assign an greater value in order to be conservative and ensure more safety.
        - SMOOTHING: YOU MUST AVOID drastically changing in the risk value (e.g., from 0.1 to 0.9 or from 0.7 to 0.85) unless a critical threat happens.
        
        [RISK SCALE]
        - 0.0 - 0.29: Non vulnerable, small or inhert objects.
        - 0.3 - 0.59: Objects with predictale movement or relatively close.
        - 0.6 - 0.79: Humans, animals, robots, rovers or possible unpredictable objects that can move unexpectedly or can be fragile or dangerous.
        - 0.8 - 1.0: Imminent collision or high unpredictable movements.

        Return a very coinces response considering indicated JSON format.

        [EXAMPLE OF ANALYSIS]
            {
                'objects': [
                    {
                    "name": "human_worker",
                    "visual_features": "Person wearing a high-visibility reflective vest, holding a tablet.",
                    "temporal_analysis": "Distance decreased by 0.5m; the subject is actively closing the gap with the robot.",
                    "potential_consequence": "Physical injury to the operator and immediate legal/operational shutdown.",
                    "dangerousness": 0.85
                    }
                ]
            }
        
        """

# class Object(BaseModel):
#     object_id: str = Field(description = "The object ID received in the message, e.g. object_1, object_2, etc")
#     name: str = Field(description = "The corresponding name of the identified object, e.g. person, robot, rover, cylinder, etc")
#     # relative_distance: float
#     # relative_angle: float
#     spatial_context: str = Field(description="Describe the proximity and orientation (e.g., 'critically close and frontal')")
#     action: str = Field(description= "The action that the object is doing according to visual, spatial and physical information.")
#     reasoning: str = Field(description="Brief analysis of the object's behavior and environment.")
#     dangerousness: float = Field(
#         description= "Value ranging from 0 to 1 that express both intrinsic dangerousness of the object related to the type of object and what the object is doing.",
#         ge=0.0,
#         le=1.0
#         )

class Object(BaseModel):
    name: str 
    visual_features: str = Field(description="Physical characteristics (eg. fragile, heavy, mobile)")
    temporal_analysis: str = Field(description="Comparison between time instant 't' and 't-1'. Is the object close? Has it changed trajectory? the previous dangerousness is coherent?")
    potential_consequence: str = Field(description="What could happen in case of collision? What could the object do saddenly and unpredictably?")
    previous_dangerousness: str
    reasoning: str = Field(description="Reasong to change the dangerousness value from previous timestep")
    dangerousness: float = Field(
        description="Final risk evalution ranging from 0 to 1. ",
        ge=0.0, 
        le=1.0
    )

class vlmResponse(BaseModel):
    # overall_safety_context: str = Field(description="Summary of the most critical threats in the scene.")
    objects: List[Object]

class VLMnode:
    def __init__(self):

        # Node initialization
        rospy.init_node('vlm_node', anonymous=True)
        self.bridge = CvBridge()

        # Publisher
        self.image_description_pub = rospy.Publisher("/vlm/image/description", String, queue_size=1)

        # Service client
        self.get_image_service = rospy.ServiceProxy("/get_camera_image", highRoadInfo)

        self.streaming = True

        self.previous_info = {}

    def build_prompt(self, current_info):
        current_str = json.dumps(current_info)
        
        if self.previous_info == {}:
            memory_str = current_str
        else:
            memory_str = json.dumps(self.previous_info)

        return f"""
        [CURRENT DATA (t)]
        {current_str}

        [PREVIOUS DATA (t-1)]
        {memory_str}
        """


    def get_frame_from_camera(self):
        try:
            rospy.wait_for_service('/get_camera_image')
            response = self.get_image_service()

            relevant_info = response.relevant_info
            frame = response.frame

            cv_image = self.bridge.imgmsg_to_cv2(frame, desired_encoding='bgr8')
            return cv_image, relevant_info
        
        except rospy.ServiceException as e:
            rospy.logwarn(f"[VLMImageRequester] Errore chiamando il servizio: {e}")
            return None


    def show_image(self, cv_image):
        if cv_image is not None:
            cv2.imshow("Frame ricevuto", cv_image)
            cv2.waitKey(3000)
            cv2.destroyWindow("Frame ricevuto")


    def convert_to_pil(self, cv_image):
        """
        Convert an OpenCV image (BGR) in PIL image (RGB)
        """
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_image)
        return pil_image

    def convert_pil_to_base64(self, pil_image, image_format="JPEG"):
        """
        Converte un'immagine PIL in una stringa Base64.
        
        Args:
            pil_image: Oggetto PIL.Image da convertire.
            image_format: Formato in cui salvare l'immagine nel buffer (es. "JPEG", "PNG").
                        JPEG è spesso preferito per un equilibrio tra dimensione e qualità.

        Returns:
            Stringa Base64 (UTF-8) dell'immagine.
        """
        
        # 1. Creare un buffer di memoria per "salvare" l'immagine senza usare il disco
        buffer = BytesIO()
        
        # 2. Salvare l'immagine PIL nel buffer. È NECESSARIO specificare il formato.
        #    'quality=85' è opzionale ma consigliato per i JPEG per ridurre le dimensioni del payload
        pil_image.save(buffer, format=image_format, quality=85)
        
        # 3. Ottenere il valore binario dal buffer
        image_bytes = buffer.getvalue()
        
        # 4. Codificare i byte in Base64
        base64_bytes = base64.b64encode(image_bytes)
        
        # 5. Decodificare i byte Base64 in una stringa UTF-8 per la trasmissione
        base64_string = base64_bytes.decode('utf-8')
        
        return base64_string


    def vlm_inference(self, base64_image, relevant_info_msg):
        
        mask_object_name = False

        print("Previous info:")
        print(self.previous_info)

        if mask_object_name:
            relevant_info = json.loads(relevant_info_msg)
            print("Current info:")
            print(relevant_info)

            # Substituting Explicit keys with "object id"
            current_info = {}
            id = 1
            
            for k,v in relevant_info.items():
                object_id = "object_" + str(id)
                current_info[object_id] = relevant_info[k]
                id += 1
            print("Masked current info:")
            print(current_info)
        else:
            current_info = json.loads(relevant_info_msg)
            print("Current info:")
            print(current_info)


        prompt = self.build_prompt(current_info)


        vlm_inference_response = chat(
            'qwen3-vl:8b-instruct',
            messages=[
                    {
                        'role': 'system',
                        'content': SYSTEM_PROMPT
                    },
                    {
                        'role': 'user', 
                        'content': prompt,
                        'images': [base64_image]
                    }
                ],
            format=vlmResponse.model_json_schema(),
            options={
                'temperature': 0,
                # 'top_p': 0.8,
                'top_k':1,              # Exploit only the most probable token
                'seed': 42
                },
            stream = self.streaming,
        )


        self.previous_info = current_info

        if self.streaming:
            full_response_content = ""
            for chunk in vlm_inference_response:
                print(chunk['message']['content'], end='', flush=True) 
                full_response_content += chunk['message']['content']

            # Add the response to the messages to maintain the history
            # messages += [
            #     {'role': 'user', 'content': user_input},
            #     {'role': 'assistant', 'content': full_response_content},
            # ]
            self.image_description_pub.publish(full_response_content)

            response_dict = json.loads(full_response_content)


            for object in response_dict["objects"]:
                self.previous_info[object['name']] = {
                "dangerousness": object['dangerousness'],
                "info": current_info.get(object['name'], "N/A")
            }
            


        else:
            try:
                image_description = vlmResponse.model_validate_json(vlm_inference_response.message.content)
            except Exception as e:
                rospy.logerr("[VLM_node: Response format not respected]")

            print(image_description)

            # Add the response to the messages to maintain the history
            # messages += [
            #     {'role': 'user', 'content': user_input},
            #     {'role': 'assistant', 'content': response.message.content},
            # ]
        
            self.image_description_pub.publish(vlm_inference_response.message.content)

        
        return


    def run(self):
        while not rospy.is_shutdown():

            cv_image, relevant_info = self.get_frame_from_camera()
            if cv_image is None:
                continue

            # self.show_image(cv_image)
            pil_image = self.convert_to_pil(cv_image)
            base64_image = self.convert_pil_to_base64(pil_image)
            print("#####")
            start_time = time.perf_counter()
            self.vlm_inference(base64_image, relevant_info)
            end_time = time.perf_counter()
            print(f"\nInference time: {round(end_time-start_time,4)}")
            print("#####")
            # rospy.sleep(5)


if __name__ == '__main__':
    try:
        VLM_node = VLMnode()
        VLM_node.run()
    except rospy.ROSInterruptException as e:
        rospy.logerr(f"[VLM_node] Errore chiamando il servizio: {e}")
        cv2.destroyAllWindows()
        pass
