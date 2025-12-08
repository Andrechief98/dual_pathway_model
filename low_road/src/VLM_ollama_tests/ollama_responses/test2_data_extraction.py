import json
import os
import re
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))

input_json_file_name = "test2.json"
full_input_json_path = os.path.join(script_dir, input_json_file_name)

output_json_file_name = "test2_results.json"
full_output_json_path = os.path.join(script_dir, output_json_file_name)

# GROUND TRUTH
ground_truth = {
    "farthest_object": {
                    "object_id": 2,
                    "name": "person",
                    "relative_distance": 3
                },
    "objects": [
            {
                "object_id": 1,
                "name": "wheeled robot"
            },
            {
                "object_id": 2,
                "name": "person"
            },
            {
                "object_id": 3,
                "name": "blue cylinder"
            }
        ],
"most_dangerous_object": {
            "object_id": 1,
            "name": "wheeled robot",
            "dangerousness": 0.8,
            "reason": "The wheeled robot is a mechanical, motorized object with significant movement potential and mass. It poses the greatest risk of collision or mechanical injury in an unfamiliar environment."
        },
"objects_features": [
            {
                "object_id": 1,
                "name": "wheeled robot",
                "attributes": "Large, black, heavy-duty vehicle with multiple wheels, likely a mobile platform or transport unit. It is stationary but could move if activated.",
                "relative_distance": 2,
                "relative_angle": 1.57,
                "action": "stationary",
                "dangerousness": 0.8
            },
            {
                "object_id": 2,
                "name": "person",
                "attributes": "Human figure standing upright, wearing a white shirt and blue jeans. Appears to be stationary, observing the scene, but could move at any time.",
                "relative_distance": 3,
                "relative_angle": -1.57,
                "action": "stationary",
                "dangerousness": 0.3
            },
            {
                "object_id": 3,
                "name": "blue cylinder",
                "attributes": "Solid blue, compact cylindrical object with a flat top. Appears to be stationary and non-functional, serving as a static obstacle.",
                "relative_distance": 0.5,
                "relative_angle": 0,
                "action": "stationary",
                "dangerousness": 0.1
            }
        ]
}

def extract_architecture_name(full_name):
    # 1. Dividi la stringa al primo trattino ('-')
    architecture_name = full_name.split('-')[0]
    
    # 2. Se non c'era un trattino, la stringa intera viene restituita.
    #    Ora, dividi il risultato al primo due punti (':')
    architecture_name = architecture_name.split(':')[0]
    
    return architecture_name


# RAW DATA EXTRACTION
with open(full_input_json_path, "r") as f:
    responses = json.load(f)

models_list = list(responses.keys())
inference_times_list = []
parameters_list = []
accuracy_list = []
architecture_name_list = []

for model in models_list:
    print(model)

    # MEAN INFERENCE TIME EXTRACTION
    sum_inference_time = 0
    for trial in list(responses[model].keys()):
        sum_inference_time += responses[model][trial]["inference_time"]
    inference_times_list.append(round(sum_inference_time/len(list(responses[model].keys())),4))


    # NUMBER OF PARAMETERS EXTRACTION
 
    if model == "llama3.2-vision:latest":
            params_numb = 11.0
            parameters_list.append(params_numb)
    else:
        params_numb = re.search(r'(\d+\.?\d*)b', model)        

        if params_numb:
            parameters_list.append(float(params_numb.group(1)))
        else:
            parameters_list.append(np.nan) # Gestione errore se il nome non combacia col regex


    # COMPUTE ACCURACY
    q1_accuracy_list = []
    q2_accuracy_list = []
    q3_accuracy_list = []
    q4_accuracy_list = []

    for trial in list(responses[model].keys()):
        # Q1: farthest object
        q1_model_response = responses[model][trial]["response"]["farthest_object"]
        q1_gt = ground_truth["farthest_object"]


        q1_accuracy = 0
        for mr, gt in zip(q1_model_response.values(), q1_gt.values()):
            if mr == gt:
                q1_accuracy += 1/len(q1_gt.values())

        q1_accuracy_list.append(q1_accuracy)

        # Q2: objects list
        q2_model_response = responses[model][trial]["response"]["objects"]
        q2_gt = ground_truth["objects"]

        q2_accuracy = 0
        for mr, gt in zip(q2_model_response, q2_gt):
            if gt["object_id"] == 1:
                if "robot" in mr["name"] or "vehicle" in mr["name"] or "truck" in mr["name"]:
                    q2_accuracy += 1/len(q2_gt)
            elif gt["object_id"] == 2:
                if "person" in mr["name"] or "human" in mr["name"]:
                    q2_accuracy += 1/len(q2_gt)
            elif gt["object_id"] == 3:
                if "cylinder" in mr["name"]:
                    q2_accuracy += 1/len(q2_gt)


        q2_accuracy_list.append(q2_accuracy)

        # Q3: Most dangerous object
        q3_model_response = responses[model][trial]["response"]["most_dangerous_object"]
        q3_gt = ground_truth["most_dangerous_object"]

        q3_accuracy = 0
        for mr, gt in zip(q3_model_response.items(), q3_gt.items()):
            mr_key, mr_value = mr
            gt_key, gt_value = gt

            if gt_key == "name":
                if "robot" in mr_value or "vehicle" in mr_value or "truck" in mr_value:
                    q3_accuracy += 1/2

            elif gt_key == "dangerousness" or gt_key == "reason":
                continue
            else:
                if gt_value == mr_value:
                    q3_accuracy += 1/2

        q3_accuracy_list.append(q3_accuracy)

        # Q4: Object features [EVALUATED BY GEMINI3 - LLM as Judge]
        if model == "qwen3-vl:2b-instruct":
            q4_accuracy_list = [0, 0, 0]
        elif model == "qwen3-vl:4b-instruct":
            q4_accuracy_list = [1, 1, 1]
        elif model == "qwen3-vl:8b-instruct":
            q4_accuracy_list = [1, 1, 1]
        elif model == "llama3.2-vision:latest":
            q4_accuracy_list = [0, 0, 0]
        elif model == "llava:7b":
            q4_accuracy_list = [0, 0, 0]
        elif model == "llava:13b":
            q4_accuracy_list = [0, 0, 0]
        elif model == "gemma3:4b":
            q4_accuracy_list = [0, 0, 0]
        elif model == "gemma3:12b":
            q4_accuracy_list = [0, 0, 0]

    print(q1_accuracy_list)
    print(q2_accuracy_list)
    print(q3_accuracy_list)
    print(q4_accuracy_list)

    sum_final_accuracy = np.array(q1_accuracy_list).mean() + np.array(q2_accuracy_list).mean() + np.array(q3_accuracy_list).mean() + np.array(q4_accuracy_list).mean()
    final_accuracy = sum_final_accuracy/4
    accuracy_list.append(final_accuracy*100)


    # EXTRACTION ARCHITECTURE
    architecture_name_list.append(extract_architecture_name(model))

data = {
    'Model_Name': models_list,
    'VLM_Accuracy': accuracy_list,
    'Inference_Time_s': inference_times_list,
    'Parameters_B': parameters_list,
    'Architecture': architecture_name_list
}


with open(full_output_json_path, "w") as f:
    json.dump(data, f, indent=4)
