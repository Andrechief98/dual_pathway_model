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
import os
from typing import List
from pydantic import BaseModel, Field, create_model
from typing import List, Optional, Dict, Any, Type
import base64
from io import BytesIO
import json
import yaml


from typing import List, Optional, Dict, Any
from pydantic import create_model, Field

class factoryResponseModel:
    @staticmethod
    def get_python_type(type_name: str):
        type_map = {"str": str, "float": float, "int": int, "bool": bool}
        base_type = type_map.get(type_name, str)
        return base_type

    @classmethod
    def build_pydantic_models(cls, schema_config: Dict[str, Any]):
        """Dynamically creates Object and vlmResponse classes with constraints."""
        
        fields = {}
        for field_name, attr in schema_config.items():
            # Determiniamo il tipo
            py_type = cls.get_python_type(attr['type'])
            
            # Prepariamo gli argomenti per pydantic.Field
            field_kwargs = {
                "description": attr.get('description', "")
            }
            
            # Aggiungiamo i vincoli numerici se presenti (ge, le)
            if "ge" in attr:
                field_kwargs["ge"] = float(attr["ge"])
            if "le" in attr:
                field_kwargs["le"] = float(attr["le"])

            fields[field_name] = (py_type, Field(..., **field_kwargs))

        # 1. Creazione dinamica della classe del singolo oggetto
        DynamicObject = create_model("DynamicObject", **fields)
        
        # 2. Creazione della classe wrapper per la risposta VLM
        DynamicResponse = create_model(
            "vlmResponse", 
            objects=(List[DynamicObject], Field(..., description="List of evaluated objects"))
        )
        
        return DynamicResponse

script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = "../config/vlm_config.yaml"
full_path = os.path.join(script_dir, relative_path)

# print("Full path")
# print(full_path)

# Load YAML
with open(full_path, 'r') as f:
    config = yaml.safe_load(f)

# Load experiment VLM configuration
experiment_id = "test_1"
exp_data = next(e for e in config['experiments'] if e['id'] == experiment_id)

SYSTEM_PROMPT = exp_data['system_prompt']

vlmResponse = factoryResponseModel.build_pydantic_models(exp_data['schema'])


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

        self.experiment = rospy.get_param("/experiment", "")


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

            # artifically remove the "person"
             
            current_info = json.loads(relevant_info_msg)

            if self.experiment == "dynamic":
                name_to_remove = "person"
                if name_to_remove in list(current_info.keys()):
                    del current_info[name_to_remove]



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
