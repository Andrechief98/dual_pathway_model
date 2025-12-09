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


    def get_frame_from_camera(self):
        try:
            rospy.wait_for_service('/get_camera_image', timeout=5.0)
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


    def vlm_inference(self, pil_image, relevant_info):
        
        # TO ADD: cosine similarity between consecutive images

        # print(relevant_info)

        prompt = f"""
        You are a robot in an indoor environment. Your goal is to understand what you see in the image and provide a value of dangerousness for each object in the environment.
        Such value will be used to modify your navigation behavior. Not consider walls as objects.

        Return a response considering indicated JSON format.
        """

        vlm_inference_response = chat(
            'qwen3-vl:8b',
            messages=[
                {
                    'role': 'user', 
                    'content': prompt,
                    'image': [pil_image]
                    }
                ],
            format=vlmResponse.model_json_schema(),
            options={'temperature': 0},
        )
        
        self.image_description_pub.publish(vlm_inference_response.message.content)
        return


    def run(self):
        while not rospy.is_shutdown():
            cv_image, relevant_info = self.get_frame_from_camera()
            if cv_image is None:
                continue

            # self.show_image(cv_image)
            pil_image = self.convert_to_pil(cv_image)
            start_time = time.perf_counter()
            self.vlm_inference(pil_image, relevant_info)
            end_time = time.perf_counter()
            print(f"Inference time: {round(end_time-start_time,4)}")
            # rospy.sleep(5)


if __name__ == '__main__':
    try:
        VLM_node = VLMnode()
        VLM_node.run()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
        pass
