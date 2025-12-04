from ollama import chat
import time
import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

image_rel_path = "ollama_test_images/runner.jpeg"
image_full_path = os.path.join(script_dir, image_rel_path)

responses_rel_path = "ollama_responses/test1.json"
responses_full_path = os.path.join(script_dir, responses_rel_path)

# TEST_1: PHYSICAL INFORMATION UNDERSTANDING AND CONTEXTUALIZED DESCRIPTION
# The idea is to provide an image along with physical information to the model in order to:
# 1) understand if it is able to understand it
# 2) contextualize the description of what is seeing
# 3) measure the inference time required by the model


if __name__ == "__main__":

    model_to_test = "qwen2.5vl:3b" 
    n_trials = 2
    with open(responses_full_path) as f:
        responses = json.load(f)

    if model_to_test not in list(responses.keys()):
        responses[model_to_test] = {} 

    for trial in range(0, n_trials):
        vel_list = [25, 10, 0]

        messages=[
            {
            'role': 'user',
            'content': 'Describe what the subject of the image is doing considering provided information. Be concise and coherent with information provided.',
            'images': [image_full_path],
            }
        ]

        trial_str = "trial_" + str(trial+1)
        responses[model_to_test][trial_str] = []

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
                model_to_test,
                messages=[
                    *messages, 
                    {
                        'role': 'user', 
                        'content': user_input,
                        'images':[image_full_path]
                        }
                    ],
            )
            end_time = time.perf_counter()

            inference_time = round(end_time-start_time,4)
            print(f"Inference time: {inference_time}")

            # Add the response to the messages to maintain the history
            messages += [
                {'role': 'user', 'content': user_input},
                {'role': 'assistant', 'content': response.message.content},
            ]
            print(response.message.content + '\n')
        
            
            responses[model_to_test][trial_str].append(
                {   
                    "velocity" : vel,
                    "response" : response.message.content,
                    "inference_time" : inference_time
                }
            )

    with open(responses_full_path, "w") as f:
        json.dump(responses, f, indent=4)