# Step 1: Install all required libraries
# !pip install gradio ultralytics ultralytics-cam opencv-python numpy -q

# Step 2: Import libraries
import gradio as gr
from ultralytics import YOLO
from ultralytics_cam import CAM
import cv2
import numpy as np
from PIL import Image
import torch

# =======================================================================================
# IMPORTANT HACKATHON NOTE!
# This code is designed for a SINGLE model that detects combined classes (e.g., 'fender-dent').
#
# INSTRUCTIONS:
# 1. Train your model and save the 'best.pt' file.
# 2. Upload the file to your Colab environment.
# 3. Update the path in the line below.
# =======================================================================================
MODEL_PATH = 'yolov8n.pt'  # <-- REPLACE WITH YOUR TRAINED 'part-damage' MODEL

# Load the model and CAM instance globally
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assessor_model = YOLO(MODEL_PATH)
    yolo_cam = CAM(MODEL_PATH) # For Grad-CAM
except Exception as e:
    print(f"Error loading model: {e}")
    assessor_model = yolo_cam = None

# --- MOCK DATA FOR THE DEMO ---
# In your actual project, your model's `names` attribute will have the correct classes.
# We define them here to simulate your custom model with the generic yolov8n.pt.
# The class ID '2' from the COCO dataset ('car') will be treated as 'fender-dent'.
MOCK_CLASSES = {2: 'fender-dent', 3: 'motorcycle-scratch'}

def assess_claim(input_image: np.ndarray, confidence_threshold: float):
    """
    Main function for the single-model pipeline.
    """
    if assessor_model is None:
        return None, "Error: Model is not loaded. Please check the model path."

    original_image = Image.fromarray(input_image)
    
    # --- Single-Stage Inference ---
    results = assessor_model(original_image, verbose=False)
    
    all_findings = []
    highest_confidence_finding = {'confidence': -1}
    
    for result in results:
        for box in result.boxes:
            confidence = float(box.conf)
            if confidence > confidence_threshold:
                class_id = int(box.cls)
                
                # Use mocked class names for this demo
                class_name = MOCK_CLASSES.get(class_id, 'unknown')

                if class_name != 'unknown':
                    finding = {
                        'class_name': class_name,
                        'confidence': confidence,
                        'coords': box.xyxy[0].int().tolist(), # [x1, y1, x2, y2]
                        'class_id': class_id
                    }
                    all_findings.append(finding)
                    
                    if confidence > highest_confidence_finding['confidence']:
                        highest_confidence_finding = finding

    # --- Visualization and Explainability ---
    annotated_image = np.array(original_image.copy())
    narrative = "## 🤖 AI Assessment Report\n\n"

    if not all_findings:
        narrative += "**✅ No significant damages were detected.**"
    else:
        narrative += f"Found **{len(all_findings)}** potential issue(s):\n\n"
        
        # --- Grad-CAM on the highest confidence detection ---
        if highest_confidence_finding['confidence'] > -1:
            try:
                # Generate heatmap for the full image, targeting the specific class
                cam_results = yolo_cam.predict(
                    original_image,
                    target_class_idx=highest_confidence_finding['class_id']
                )
                heatmap_bgr = cam_results['heatmap']
                heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
                
                # Blend the full-size heatmap with the original image
                alpha = 0.5
                annotated_image = cv2.addWeighted(
                    heatmap_rgb, alpha, annotated_image, 1 - alpha, 0
                )
                narrative += f"🔬 **Explainability (Grad-CAM):** The highlighted area shows the model's focus for detecting **'{highest_confidence_finding['class_name']}'**.\n\n"
            except Exception as e:
                print(f"Could not generate Grad-CAM: {e}")

        # Draw boxes and create narrative
        for finding in all_findings:
            coords = finding['coords']
            label = f"{finding['class_name'].replace('-', ' ').title()} ({finding['confidence']:.2f})"

            # Draw bounding box (red)
            cv2.rectangle(annotated_image, (coords[0], coords[1]), (coords[2], coords[3]), (0, 0, 255), 2)
            cv2.putText(annotated_image, label, (coords[0], coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add to narrative
            narrative += f"- Detected **{finding['class_name'].replace('-', ' ')}** with **{finding['confidence']:.2f}** confidence.\n"

    return annotated_image, narrative


# --- Create Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 AI-Powered Claims Assessor (Single-Model)")
    gr.Markdown(
        "Upload an image of a vehicle to detect damages. This version uses a **single model** to identify part-damage combinations (e.g., 'fender-dent')."
        "\n\n**DEMO NOTE:** This app uses a generic `yolov8n.pt` model. It will detect 'cars' as 'fender-dents' to demonstrate the pipeline. **You must replace it with your custom-trained model.**"
    )
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Upload Vehicle Image")
            confidence_slider = gr.Slider(minimum=0.2, maximum=1.0, value=0.4, step=0.05, label="Confidence Threshold")
            submit_btn = gr.Button("Assess Claim", variant="primary")
        with gr.Column():
            output_image = gr.Image(type="numpy", label="Assessment Result")
            report_output = gr.Markdown(label="Explainability Report")
    
    submit_btn.click(
        fn=assess_claim,
        inputs=[input_image, confidence_slider],
        outputs=[output_image, report_output]
    )
    
    gr.Examples(
        examples=[
            ["https://media.wired.com/photos/59325b73a312645844994275/master/w_2560%2Cc_limit/Parking-Lot-Fender-Bender-50-50-TA.jpg", 0.4],
            ["https://www.iihs.org/cdn-cgi/image/width=1200/api/ratings/model-year-images/3173/635/", 0.4]
        ],
        inputs=[input_image, confidence_slider]
    )

# Launch the app
demo.launch(debug=True)