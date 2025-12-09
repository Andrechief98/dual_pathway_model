import json
import os
import re
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))

input_json_file_name = "test1.json"
full_input_json_path = os.path.join(script_dir, input_json_file_name)

output_json_file_name = "test1_results.json"
full_output_json_path = os.path.join(script_dir, output_json_file_name)


# GROUND TRUTH
ground_truth = {}


def extract_architecture_name(full_name):
    architecture_name = full_name.split('-')[0]
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
    for trial in list(responses[model].keys()):
        trial_inference_time_list = []
        sum_trial_inference_time = 0
        for response_trial_idx in range(len(responses[model][trial])):
            sum_trial_inference_time += responses[model][trial][response_trial_idx]["inference_time"]
        
        trial_inference_time_list.append(sum_trial_inference_time/len(responses[model][trial]))
        
    inference_times_list.append(round(np.array(trial_inference_time_list).mean(),4))


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

    # COMPUTE ACCURACY (FROM GEMINI - LLM as judge)
    accuracy_list = [0.0, 0.0, 50, 100, 0.0, 0.0, 50, 100]
    # q1_accuracy_list = []
    # q2_accuracy_list = []
    # q3_accuracy_list = []
    # q4_accuracy_list = []

    # for trial in list(responses[model].keys()):
    #     continue

    # sum_final_accuracy = np.array(q1_accuracy_list).mean() + np.array(q2_accuracy_list).mean() + np.array(q3_accuracy_list).mean() + np.array(q4_accuracy_list).mean()
    # final_accuracy = sum_final_accuracy/4
    # accuracy_list.append(final_accuracy*100)



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
