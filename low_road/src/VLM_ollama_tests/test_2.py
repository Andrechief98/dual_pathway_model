from ollama import chat
import time
import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
image_rel_path = "/ollama_test_images/static_obstacles.jpeg"
full_path = os.path.join(script_dir, image_rel_path)

# TEST_2: OBJECT FEATURES DESCRIPTION
# The idea is to provide an image and corresponding spatial information to the model in order to:
# 1) understand if it is able to extract important object related features
# 2) answer to specific spatial question
# 3) measure the inference time required by the model

if __name__ == "__main__":


    messages=[
        {
        'role': 'user',
        'content': '''I will provide you an image with some spatial and physical information associated to each object in the environment. Such information includes relative angles (in radians) and relative distance (in meters) from you.
        Convention for angles (right-hand for positive rotation):
            - 0 radians means in front of you;
            - the angle becomes positive towards left 
            - the angle becomes negative towards right
        Your goal is to understand and couple visual information with both physical and spatial information. Based on such understanding, you must reply to the following questions:
        1) What is the farest object from you?
        2) Provide the corresponding name that identifies the object_1, object_2, object_3
        3) Describe which could be the most dangerous object.
        4) Extract and describe revelant features of each object''',
        }
    ]


    user_input = {
        "image_spatial_physical_info":{
            "object_1":{
                "relative_angle": -1.57,
                "relative_distance": 2
            },
            "object_2":{
                "relative_angle": 1.57,
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
        'qwen3vl-2b:custom',
        messages=[
            *messages, 
            {
                'role': 'user', 
                'content': user_input,
                'images':[full_path]
                }
            ],
    )
    end_time = time.perf_counter()

    print(f"Inference time: {round(end_time-start_time,4)}")

    # Add the response to the messages to maintain the history
    messages += [
        {'role': 'user', 'content': user_input},
        {'role': 'assistant', 'content': response.message.content},
    ]
    print(response.message.content + '\n')
    
    # print("--------------")
    # print(messages)