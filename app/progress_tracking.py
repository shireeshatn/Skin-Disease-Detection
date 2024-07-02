import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

# Specify the mean and standard deviation for normalization
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Function to load and preprocess the image
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

# Function to predict acne severity
def predict_acne_severity(image_path):
    try:
        # Load and preprocess the image
        image_tensor = load_image(image_path)

        # Load the pre-trained model
        model = torch.load('best.pt', map_location=torch.device('cpu'))  # Load model to CPU
        model.eval()  # Set the model to evaluation mode

        # Perform the prediction
        with torch.no_grad():  # No need to calculate gradients for inference
            output = model(image_tensor)

        # Get the predicted class
        _, predicted_class = torch.max(output, 1)
        predicted_class = predicted_class.item()  # Convert tensor to a scalar value

        # Map the predicted class to the severity label (0, 1, 2, 3)
        severity_labels = ['Mild', 'Moderate', 'Severe', 'Very Severe']
        severity_label = severity_labels[predicted_class]

        print(f"Predicted class: {predicted_class}")
        print(f"Acne severity: {severity_label}")

        return {'predicted_class': predicted_class, 'severity': severity_label}

    except Exception as e:
        return {'error': str(e)}

