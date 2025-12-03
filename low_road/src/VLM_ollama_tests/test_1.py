from ollama import chat
import time
import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
image_rel_path = "/ollama_test_images/runner.jpeg"
full_path = os.path.join(script_dir, image_rel_path)

# TEST_1: PHYSICAL INFORMATION UNDERSTANDING AND CONTEXTUALIZED DESCRIPTION
# The idea is to provide an image along with physical information to the model in order to:
# 1) understand if it is able to understand it
# 2) contextualize the description of what is seeing
# 3) measure the inference time required by the model


if __name__ == "__main__":

    vel_list = [25, 20, 15, 10, 5, 0]

    messages=[
        {
        'role': 'user',
        'content': 'Describe what the subject of the image is doing considering provided information. Be concise and coherent with information provided.',
        'images': [full_path],
        }
    ]

    for vel in vel_list:
        user_input = {
            "image_info_in_time":{
                "runner":{
                    "current_velocity": vel
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