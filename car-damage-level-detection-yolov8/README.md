---
license: mit
---
# 🚗 Car Damage Level Detection ML Model (YOLOv8)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/66d27a34d51528a038d1d50c/g8PMxp1X-cVyfblPAZ6CN.png)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/66d27a34d51528a038d1d50c/wNAviMt6z6Rpdj8hAfKOm.png)

## 🧠 Model Summary

This machine learning model is developed to **detect and classify vehicle damage levels** using **YOLOv8 (You Only Look Once v8)** architecture. It categorizes car damage into **three levels**:

- `Light`
- `Moderate`
- `Severe`

The model is trained on a dataset containing images of vehicles taken from different angles, labeled based on the visible damage severity. It can be used in applications such as insurance automation, car rental inspections, and fleet monitoring systems.

- **Architecture**: YOLOv8
- **Task**: Object Detection & Multi-class Classification
- **Classes**: Light, Moderate, Severe
- **License**: MIT
- **Fine-tunable**: ✅ Yes

---

## 🚀 Usage Example

```python
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO("path/to/best.pt")  # Replace with your model path or use huggingface_hub download

# Run inference on an image
results = model("path/to/your_image.jpg")

# Visualize results
results.show()

# Print predictions
print(results[0].boxes.cls)       # class indices
print(results[0].boxes.conf)      # confidence scores
