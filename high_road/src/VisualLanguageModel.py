#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PIL import Image as PILImage
import cv2
from srv import highRoadInfo  # Sostituisci con il nome del tuo servizio

class VLMImageRequester:
    def __init__(self):
        # Inizializza il nodo
        rospy.init_node('vlm_image_requester', anonymous=True)
        self.bridge = CvBridge()

        # Parametri
        self.service_name = rospy.get_param("~service_name", "/get_camera_image")
        self.display_time_ms = rospy.get_param("~display_time_ms", 3000)  # durata visualizzazione in ms

        rospy.loginfo(f"[VLMImageRequester] Nodo pronto. Servizio: {self.service_name}")

        # Attende che il servizio sia disponibile
        rospy.wait_for_service(self.service_name)
        self.get_image_service = rospy.ServiceProxy(self.service_name, highRoadInfo)

    def get_single_frame(self, timeout=5.0):
        """
        Richiede un singolo frame dal servizio ROS
        """
        try:
            # Chiamata al servizio
            response = self.get_image_service()
            cv_image = self.bridge.imgmsg_to_cv2(response.image, desired_encoding='bgr8')
            return cv_image
        except rospy.ServiceException as e:
            rospy.logwarn(f"[VLMImageRequester] Errore chiamando il servizio: {e}")
            return None

    def show_image(self, cv_image):
        """
        Mostra l'immagine in una finestra OpenCV per display_time_ms millisecondi
        """
        if cv_image is not None:
            cv2.imshow("Frame ricevuto", cv_image)
            cv2.waitKey(self.display_time_ms)
            cv2.destroyWindow("Frame ricevuto")

    def convert_to_pil(self, cv_image):
        """
        Converte un'immagine OpenCV (BGR) in PIL.Image (RGB)
        """
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_image)
        return pil_image

    def process_with_vlm(self, pil_image):
        """
        Qui va la logica del VLM.
        Attualmente simulata con un ritardo
        """
        rospy.loginfo(f"[VLMImageRequester] Elaborazione immagine (size={pil_image.size})...")
        rospy.sleep(2.0)  # Simula il processing
        rospy.loginfo("[VLMImageRequester] Elaborazione completata ✅")

    def run(self):
        """
        Loop principale: richiede un frame alla volta, lo visualizza e lo processa
        """
        while not rospy.is_shutdown():
            cv_image = self.get_single_frame()
            if cv_image is None:
                continue  # riprova se servizio non disponibile

            self.show_image(cv_image)
            pil_image = self.convert_to_pil(cv_image)
            self.process_with_vlm(pil_image)


if __name__ == '__main__':
    try:
        node = VLMImageRequester()
        node.run()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
        pass
