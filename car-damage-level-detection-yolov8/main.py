import sys
from ultralytics import YOLO

# --- 1. Check if an image path was provided ---
if len(sys.argv) < 2:
    print("Error: Please provide the image path as an argument.")
    # Exit the script if no path is given
    sys.exit(1)

# --- 2. Get the image path from the command-line argument ---
image_path = sys.argv[1]

# --- 3. Load your model and run inference ---
try:
    # Load your trained YOLOv8 classification model
    model = YOLO("/home/manan/megathon/car-damage-level-detection-yolov8/car-damage.pt")

    # Run inference on the image provided by the user
    results = model(image_path)

    # The 'results' object is a list, so we work with the first result
    first_result = results[0]

    if first_result.probs is not None:
        # Get the index of the top prediction
        top1_index = first_result.probs.top1
        
        # Get the confidence score of the top prediction
        top1_confidence = first_result.probs.top1conf.item()
        
        # Get the name of the top predicted class
        top1_classname = first_result.names[top1_index]
        
        print(f"Image Path: {image_path}")
        print(f"Top Prediction: '{top1_classname}'")
        print(f"Confidence Score: {top1_confidence:.2f}")
    else:
        print("No classification results found for the image.")

except FileNotFoundError:
    print(f"Error: The file '{image_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
