# Install required packages (uncomment if running in a new environment)
# !pip install google-genai
# !pip install google-generativeai
# !pip install roboflow
# !pip install ultralytics

import os
import json
import google.generativeai as genai
from __future__ import generator_stop
import cv2
import math
from ultralytics import YOLO
from roboflow import Roboflow

api_key = "your api"
genai.configure(api_key=api_key)
rf = Roboflow(api_key="your api")
project = rf.workspace("sindhu").project("car_dent_scratch_detection-1")
dataset = project.version(8).download("yolov8")
model = project.version(9).model

possible_classes = [
    "Bodypanel-Dent", "Front-Windscreen-Damage", "Headlight-Damage",
    "Rear-windscreen-Damage", "RunningBoard-Dent", "Sidemirror-Damage",
    "Signlight-Damage", "Taillight-Damage", "bonnet-dent", "boot-dent",
    "doorouter-dent", "fender-dent", "front-bumper-dent", "pillar-dent",
    "quaterpanel-dent", "rear-bumper-dent", "roof-dent"
]

output_dict = {key: 0.0 for key in possible_classes}

def calculate_cosine_similarity(dict1, dict2):
    """
    Calculates the cosine similarity between two dictionaries treated as vectors.
    Returns a score between 0.0 and 1.0.
    """
    all_keys = set(dict1.keys()) | set(dict2.keys())
    vector_a = [dict1.get(key, 0) for key in all_keys]
    vector_b = [dict2.get(key, 0) for key in all_keys]
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    else:
        return dot_product / (magnitude_a * magnitude_b)

prompt = input("Describe the accident: ")

prompt_template = f"""
Given the following description of a car accident, identify all types of damage from the list below and format the output as a JSON dictionary. The keys of the dictionary should be the damage types and the values should be a boolean (true if present, false if not).

Damage types: ['Bodypanel-Dent', 'Front-Windscreen-Damage', 'Headlight-Damage', 'Rear-windscreen-Damage', 'RunningBoard-Dent', 'Sidemirror-Damage', 'Signlight-Damage', 'Taillight-Damage', 'bonnet-dent', 'boot-dent', 'doorouter-dent', 'fender-dent', 'front-bumper-dent', 'pillar-dent', 'quaterpanel-dent', 'rear-bumper-dent', 'roof-dent']

Description: "{prompt}"

Example of desired output:
{{
  "Front-Windscreen-Damage": 1,
  "Headlight-Damage": 1,
  "fender-dent": 1,
  "Bodypanel-Dent": 0,
  "Rear-windscreen-Damage": 0
  ...
}}
"""

res = model.predict(
    "Car_Dent_Scratch_Detection(1)-8/test/images/car_fender--15-_jpg.rf.b8a27b92eb1fd19cddbc912d7198f167.jpg",
    confidence=40,
    overlap=30
).json()

damage_report = None
try:
    gen_model = genai.GenerativeModel('gemma-3-4b-it')
    response = gen_model.generate_content(
        prompt_template,
        generation_config={"temperature": 0.1}
    )
    output_text = response.text.strip()
    if "```json" in output_text:
        json_string = output_text.split("```json")[1].split("```")[0].strip()
    else:
        json_string = output_text
    damage_report = json.loads(json_string)
    print(json.dumps(damage_report, indent=2))
except Exception as e:
    print("Error:", e)

if 'predictions' in res and isinstance(res['predictions'], list):
    for prediction in res['predictions']:
        class_name = prediction.get('class')
        confidence = prediction.get('confidence')
        if class_name in output_dict and confidence is not None:
            output_dict[class_name] = confidence

print(json.dumps(output_dict, indent=2))

if damage_report and output_dict:
    similarity_score = calculate_cosine_similarity(damage_report, output_dict)
    print(f"CONFIDENCE : {similarity_score:.4f}")