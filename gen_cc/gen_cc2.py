from __future__ import generator_stop
import os
import json
import google.generativeai as genai
import cv2
import math
from ultralytics import YOLO
import torch
import numpy as np
import matplotlib.pyplot as plt

api_key = "AIzaSyBvUHEUIohvPADPkMZJdhcaTIFfHbRAfJU"
genai.configure(api_key=api_key)

model = YOLO("roboflow_car_damage_best.pt")

classNames = [
    'Bodypanel-Dent', 'Front-Windscreen-Damage', 'Headlight-Damage',
    'Rear-windscreen-Damage', 'RunningBoard-Dent', 'Sidemirror-Damage',
    'Signlight-Damage', 'Taillight-Damage', 'bonnet-dent', 'boot-dent',
    'doorouter-dent', 'fender-dent', 'front-bumper-dent', 'pillar-dent',
    'quaterpanel-dent', 'rear-bumper-dent', 'roof-dent'
]

output_dict = {key: 0.0 for key in classNames}

def calculate_cosine_similarity(dict1, dict2):
    """
    Calculate cosine similarity between two dictionaries.
    Converts to binary: 1 if value > 0, else 0
    """
    all_keys = set(dict1.keys()) | set(dict2.keys())
    vector_a = [1 if dict1.get(key, 0) > 0 else 0 for key in all_keys]
    vector_b = [1 if dict2.get(key, 0) > 0 else 0 for key in all_keys]
    
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))
    
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    else:
        return dot_product / (magnitude_a * magnitude_b)

# Read description from description.json
desc_path = os.path.join(os.path.dirname(__file__), "description", "description.json")
prompt = ""
try:
    with open(desc_path, "r") as f:
        desc_data = json.load(f)
    
    if "vehicle" in desc_data:
        if "description" in desc_data["vehicle"]:
            prompt = desc_data["vehicle"]["description"]
        condition_parts = []
        if "condition" in desc_data["vehicle"]:
            condition = desc_data["vehicle"]["condition"]
            def extract_damage_info(obj, prefix=""):
                parts = []
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if isinstance(value, dict):
                            parts.extend(extract_damage_info(value, f"{prefix} {key}"))
                        elif key in ["status", "severity", "location"]:
                            parts.append(f"{key}: {value}")
                return parts
            condition_parts = extract_damage_info(condition)
        
        if condition_parts:
            prompt += " " + ", ".join(condition_parts)
    elif "description" in desc_data:
        prompt = desc_data["description"]
    
    prompt = prompt.strip()
except FileNotFoundError:
    print(f"Warning: {desc_path} not found!")
except Exception as e:
    print(f"Error reading description: {e}")

print("=" * 80)
print("ACCIDENT DESCRIPTION:")
print("=" * 80)
if prompt:
    print(prompt)
else:
    print("NO DESCRIPTION PROVIDED")
print("=" * 80)
print()

# Gemini analysis
damage_report_cost = {key: 0 for key in classNames}

if prompt:
    prompt_template = f"""
You are an expert vehicle damage assessor. Analyze the following accident description and:
1. Identify which types of damage are present
2. Estimate repair costs in Indian Rupees (INR)

Damage types you can identify: {classNames}

Description: "{prompt}"

Respond ONLY with a valid JSON object with damage types as keys and estimated costs as values.
Set cost to 0 for damage types not mentioned or not present.

Cost reference ranges should be dependent on the model of the car, search them.

Output format: (example)
{{
  "Bodypanel-Dent": 0,
  "Front-Windscreen-Damage": 0,
  "Headlight-Damage": 0,
  "Rear-windscreen-Damage": 0,
  "RunningBoard-Dent": 0,
  "Sidemirror-Damage": 0,
  "Signlight-Damage": 0,
  "Taillight-Damage": 0,
  "bonnet-dent": 0,
  "boot-dent": 0,
  "doorouter-dent": 0,
  "fender-dent": 0,
  "front-bumper-dent": 0,
  "pillar-dent": 0,
  "quaterpanel-dent": 0,
  "rear-bumper-dent": 0,
  "roof-dent": 0
}}
"""

    try:
        gen_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = gen_model.generate_content(
            prompt_template,
            generation_config={"temperature": 0.2}
        )
        output_text = response.text.strip()
        
        if "```json" in output_text:
            json_string = output_text.split("```json")[1].split("```")[0].strip()
        elif "```" in output_text:
            json_string = output_text.split("```")[1].split("```")[0].strip()
        else:
            json_string = output_text
        
        damage_report_cost = json.loads(json_string)
        
        print("GEMINI DAMAGE ANALYSIS:")
        print("=" * 80)
        print(json.dumps(damage_report_cost, indent=2))
        print("=" * 80)
        print()
    except Exception as e:
        print(f"Error generating damage report: {e}")
        print()

def generate_gradcam(model, image_path, output_path):
    """Generates and saves a Grad-CAM heatmap for the given image using YOLO model."""
    try:
        import cv2
        from PIL import Image
        
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_h, original_w = img.shape[:2]
        
        results = model.predict(image_path, conf=0.4, iou=0.3, verbose=False)
        
        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            cv2.imwrite(output_path, img)
            return False
        
        yolo_model = model.model
        yolo_model.eval()
        
        img_resized = cv2.resize(img_rgb, (640, 640))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
            yolo_model = yolo_model.cuda()
        
        feature_maps = []
        def hook_fn(module, input, output):
            feature_maps.append(output)
        
        target_layer = None
        for i in range(len(yolo_model.model) - 1, -1, -1):
            layer = yolo_model.model[i]
            if hasattr(layer, 'conv') or 'Conv' in layer.__class__.__name__:
                target_layer = layer
                break
        
        if target_layer is None:
            target_layer = yolo_model.model[-4] if len(yolo_model.model) > 4 else yolo_model.model[-1]
        
        handle = target_layer.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            _ = yolo_model(img_tensor)
        
        handle.remove()
        
        if len(feature_maps) == 0:
            cv2.imwrite(output_path, img)
            return False
        
        feature_map = feature_maps[0]
        if isinstance(feature_map, tuple):
            feature_map = feature_map[0]
        
        if len(feature_map.shape) == 4:
            attention_map = torch.mean(feature_map[0], dim=0).cpu().numpy()
        else:
            attention_map = feature_map[0].cpu().numpy()
        
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        attention_map_resized = cv2.resize(attention_map, (original_w, original_h))
        heatmap = cv2.applyColorMap(np.uint8(255 * attention_map_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf.item()
            cls_idx = int(box.cls.item())
            cls_name = model.names[cls_idx]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls_name}: {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay, (x1, y1 - label_h - 5), (x1 + label_w, y1), (0, 255, 0), -1)
            cv2.putText(overlay, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.imwrite(output_path, overlay)
        return True
        
    except Exception as e:
        print(f"  Error generating Grad-CAM for {os.path.basename(image_path)}: {e}")
        import traceback
        traceback.print_exc()
        return False

images_dir = os.path.join(os.path.dirname(__file__), "images")
grade_cam_dir = os.path.join(os.path.dirname(__file__), "grade_cam")
os.makedirs(grade_cam_dir, exist_ok=True)

print("YOLO DETECTION RESULTS:")
print("=" * 80)
for img_file in sorted(os.listdir(images_dir)):
    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
        
    img_path = os.path.join(images_dir, img_file)
    results = model.predict(img_path, conf=0.4, iou=0.3, verbose=False)
    res_json = results[0].to_json()
    res = json.loads(res_json)
    
    if isinstance(res, list):
        predictions = res
    elif isinstance(res, dict) and 'predictions' in res:
        predictions = res['predictions']
    else:
        predictions = []
    
    for prediction in predictions:
        class_name = prediction.get('name') or prediction.get('class')
        confidence = prediction.get('confidence')
        
        if class_name in output_dict and confidence is not None:
            output_dict[class_name] = max(output_dict[class_name], confidence)
            print(f"  {img_file}: {class_name} (confidence: {confidence:.4f})")
    
    gradcam_path = os.path.join(grade_cam_dir, f"gradcam_{img_file}")
    generate_gradcam(model, img_path, gradcam_path)

print("=" * 80)
print()
print("FINAL YOLO DETECTION CONFIDENCES:")
print("=" * 80)
print(json.dumps(output_dict, indent=2))
print("=" * 80)
print()

print("COMPARISON RESULTS:")
print("=" * 80)

pair = None
if damage_report_cost and output_dict:
    similarity_score = calculate_cosine_similarity(damage_report_cost, output_dict)
    cost_score = sum(
        v for v in damage_report_cost.values()
        if isinstance(v, (int, float))
    )
    pair = {"confidence": similarity_score, "cost": cost_score}
    print(f"PAIR -> {pair}")
else:
    print("Could not calculate pair (missing data).")

print("=" * 80)

cc_dir = os.path.join(os.path.dirname(__file__), "cc")
os.makedirs(cc_dir, exist_ok=True)
pair_path = os.path.join(cc_dir, "cost-conf.json")
if pair:
    with open(pair_path, "w") as f:
        json.dump(pair, f, indent=2)
