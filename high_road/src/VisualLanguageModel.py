#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PIL import Image as PILImage
import cv2
from dual_pathway_interfaces.srv import highRoadInfo, highRoadInfoRequest

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
        ###### TO DO: VLM LOGIC
        
        print(relevant_info)

        vlm_inference_response = "Why the sky is blue?"
        
        self.image_description_pub.publish(vlm_inference_response)
        return


    def run(self):
        while not rospy.is_shutdown():
            cv_image, relevant_info = self.get_frame_from_camera()
            if cv_image is None:
                continue

            # self.show_image(cv_image)
            pil_image = self.convert_to_pil(cv_image)
            self.vlm_inference(pil_image, relevant_info)
            rospy.sleep(5)


if __name__ == '__main__':
    try:
        VLM_node = VLMnode()
        VLM_node.run()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
        pass
