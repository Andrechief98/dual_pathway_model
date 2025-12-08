from ollama import chat
import time
import json
import os
from typing import List
from pydantic import BaseModel

class MostDangerousObstacle(BaseModel):
    object_id: float
    name:str
    dangerousness: float
    reason: str

class FarthestObject(BaseModel):
    object_id: float
    name: str
    relative_distance: float

class Object(BaseModel):
    object_id: float
    name: str

class ObjectFeatures(BaseModel):
    object_id: float
    name: str
    attributes: str
    relative_distance: float
    relative_angle: float
    action: str
    dangerousness: float

class ImageDescription(BaseModel):
    summary: str
    farthest_object: FarthestObject
    objects: List[Object]
    most_dangerous_object: MostDangerousObstacle
    objects_features: List[ObjectFeatures]



script_dir = os.path.dirname(os.path.abspath(__file__))

image_rel_path = "ollama_test_images/static_obstacles.png"
image_full_path = os.path.join(script_dir, image_rel_path)

responses_rel_path = "ollama_responses/test2.json"
responses_full_path = os.path.join(script_dir, responses_rel_path)

# TEST_2: OBJECT FEATURES DESCRIPTION
# The idea is to provide an image and corresponding spatial information to the model in order to:
# 1) understand if it is able to extract important object related features
# 2) answer to specific spatial question
# 3) measure the inference time required by the model

if __name__ == "__main__":
    model_to_test = "gemma3:4b" 
    n_trials = 3
    streaming = False

    with open(responses_full_path) as f:
        responses = json.load(f)

    if model_to_test not in list(responses.keys()):
        responses[model_to_test] = {} 

    for trial in range(0, n_trials):
        
        messages=[
            {
            'role': 'user',
            'content': '''I will provide you an image with some spatial and physical information associated to each object in the environment. The image represents what you see. Spatial information includes relative angles (in radians) and relative distance (in meters) from you.
            Convention for angles (right-hand for positive rotation):
                - 0 radians means in front of you;
                - the angle becomes positive towards left 
                - the angle becomes negative towards right
            Your goal is to understand and match visual information with both physical and spatial information. Based on such understanding, you must reply to the following questions:
            1) What is the farthest object from you?
            2) Provide the corresponding name that identifies the object_1, object_2, object_3
            3) Describe which could be the most dangerous object using a float value between 0 and 1.
            4) Extract and describe revelant features of each object
            
            Return the response following the provided JSON format.''',
            }
        ]


        user_input = {
            "image_spatial_physical_info":{
                "object_1":{
                    "relative_angle": 1.57,
                    "relative_distance": 2
                },
                "object_2":{
                    "relative_angle": -1.57,
                    "relative_distance": 3
                },
                "object_3":{
                    "relative_angle": 0,
                    "relative_distance": 0.5
                }
            }
        }
        user_input = json.dumps(user_input)
        
        start_time = time.perf_counter()
        response = chat(
            model_to_test,
            messages=[
                *messages, 
                {
                    'role': 'user', 
                    'content': user_input,
                    'images':[image_full_path]
                    }
                ],
            format=ImageDescription.model_json_schema(),
            options={'temperature': 0.1},
            stream=streaming
        )
        
        if streaming:
            full_response_content = ""
            for chunk in response:
                print(chunk['message']['content'], end='', flush=True) 
                full_response_content += chunk['message']['content']

            end_time = time.perf_counter()

            inference_time = round(end_time-start_time,4)
            print(f"Inference time: {inference_time}")

            # Add the response to the messages to maintain the history
            messages += [
                {'role': 'user', 'content': user_input},
                {'role': 'assistant', 'content': full_response_content},
            ]
            
            trial_str = "trial_" + str(trial+1)
            responses[model_to_test][trial_str]={
                "response" : json.loads(full_response_content),
                "inference_time" : inference_time
            }

        else:
            end_time = time.perf_counter()

            inference_time = round(end_time-start_time,4)
            print(f"Inference time: {inference_time}")

            image_description = ImageDescription.model_validate_json(response.message.content)
            print(image_description)
            print()

            # Add the response to the messages to maintain the history
            messages += [
                {'role': 'user', 'content': user_input},
                {'role': 'assistant', 'content': response.message.content},
            ]
            
            trial_str = "trial_" + str(trial+1)
            responses[model_to_test][trial_str]={
                "response" : json.loads(response.message.content),
                "inference_time" : inference_time
            }
    
    with open(responses_full_path, "w") as f:
        json.dump(responses, f, indent=4)