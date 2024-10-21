import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

class_labels = {
    0: "Baked Potato",
    1: "Burger",
    2: "Crispy Chicken",
    3: "Donut",
    4: "Fries",
    5: "Hot Dog",
    6: "Pizza",
    7: "Sandwich",
    8: "Taco",
    9: "Taquito"
}

# Load your model
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)  
    
    num_classes = 10  
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 256),  
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(256, num_classes)  
    )
    
    model.load_state_dict(torch.load("resnet50.pth", map_location=torch.device('cpu')))
    model.eval() 
    return model

model = load_model()

# Define the transformations to be applied to the image
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = preprocess(image).unsqueeze(0)  
    return image

# Prediction function
def predict(image, model):
    image = preprocess_image(image)  
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        class_idx = predicted.item()
        food_name = class_labels[class_idx]
        return food_name

# Streamlit UI
st.title("Fast Food Image Classifier")
st.write("Upload an image of fast food to classify it!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    st.write("Classifying...")
    
    label = predict(image, model)
    
    st.write(f"Predicted Class: {label}")
