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
from pydantic import BaseModel
import base64
from io import BytesIO
import json

class Object(BaseModel):
    name: str
    hazardous_properties: str
    dangerousness: float


class vlmResponse(BaseModel):
    summary: str
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

        self.prompt = """aaaaaaaa"""
        self.streaming = True


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


    def vlm_inference(self, base64_image, relevant_info):
        
        # TO ADD: cosine similarity between consecutive images

        prompt = f"""
        You are a robot in an indoor environment equipped with a camera. 
        I will provide you an image with some spatial and physical information associated to each object in the environment. The image represents what you see.
        Spatial information includes:
         - relative distance (in meters) from you
         - relative orientation (in radians)
         - radial velocity (in m/s) which tells you if the object is moving towards you
        Your goal is to understand and match visual information with both physical and spatial information. Based on such understanding, you must provide a value of dangerousness for each object in the environment.
        Such value will be used to modify your navigation behavior.

        Return a response considering indicated JSON format.

        """

        relevant_info = {
            "image_spatial_physical_info" : relevant_info
        }

        print(relevant_info)
        
        prompt += json.dumps(relevant_info)


        vlm_inference_response = chat(
            'qwen3-vl:8b-instruct',
            messages=[
                {
                    'role': 'user', 
                    'content': prompt,
                    'images': [base64_image]
                    }
                ],
            format=vlmResponse.model_json_schema(),
            options={'temperature': 0},
            stream = self.streaming
        )



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
