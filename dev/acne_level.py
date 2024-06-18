from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import requests
import torch

# Load the model and feature extractor
model_name = "imfarzanansari/skintelligent-acne"
model = AutoModelForImageClassification.from_pretrained(model_name)
feature_extractor = AutoImageProcessor.from_pretrained(model_name)

# Load and preprocess the image
def load_image(image_path):
    # You can load image from file or URL
    if image_path.startswith('http'):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)
    return image

def preprocess_image(image):
    # The processor prepares the image for the model
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs

# Example image URL
image_url = r"C:\Users\prais\Downloads\AcneFace.jpeg"

# Load and preprocess the image
image = load_image(image_url)
inputs = preprocess_image(image)

# Get model predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get the predicted class
predicted_class_idx = logits.argmax(-1).item()

# Load class labels
labels = model.config.id2label

# Print the results
print(f"Predicted class: {labels[predicted_class_idx]}")

# If the model output needs post-processing, like softmax
probabilities = torch.nn.functional.softmax(logits, dim=-1)
confidence_score = probabilities[0][predicted_class_idx].item()
print(f"Confidence score: {confidence_score:.4f}")
