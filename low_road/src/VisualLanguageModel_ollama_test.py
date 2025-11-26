from ollama import chat
import time
import json
# from pathlib import Path

# Pass in the path to the image
path = "/home/andrea/Scaricati/runner.jpeg"

# You can also pass in base64 encoded image data
# img = base64.b64encode(Path(path).read_bytes()).decode()
# or the raw bytes
# img = Path(path).read_bytes()


if __name__ == "__main__":

    vel_list = [25, 20, 15, 10, 5, 0]

    messages=[
        {
        'role': 'user',
        'content': f'Describe what the subject of the image is doing considering provided information. Be concise and coherent with information provided.',
        'images': [path],
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
                    'content': "/nothink" + user_input,
                    'images':[path]
                    }
                ],
            think=False,
            options={
                "num_ctx": 2048
            }
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