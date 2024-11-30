import os
import streamlit as st
import pandas as pd
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pydicom
from io import BytesIO
import timm
import torch.nn as nn

# Define label map for predictions
label_map = {0: "Normal/Mild", 1: "Moderate", 2: "Severe"}

# Define the CustomEfficientNet class
class CustomEfficientNet(nn.Module):
    def __init__(self, model_name='efficientnet_b0', num_classes=3, pretrained=True):
        super(CustomEfficientNet, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        num_ftrs = self.model.get_classifier().in_features
        self.model.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

# Load the models
@st.cache_resource
def load_models():
    with open("models.pkl", "rb") as f:
        loaded_models = pickle.load(f)
    return loaded_models

models = load_models()

# Ensure the models are on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for key, model in models.items():
    model.to(device)

# Function to load DICOM images
def load_dicom(image_path):
    dicom = pydicom.dcmread(image_path, force=True)
    pixel_array = dicom.pixel_array
    pixel_array = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
    return pixel_array

# Custom Dataset for Test Data
class SeriesDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = load_dicom(image_path)
        if self.transform:
            image = self.transform(image)
        return image

# Prediction function
def predict_series(model, dataloader):
    predictions = []
    normal_mild_probs = []
    moderate_probs = []
    severe_probs = []

    model.eval()
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            predictions.extend(predicted.cpu().numpy())
            normal_mild_probs.extend(probs[:, 0].cpu().numpy())
            moderate_probs.extend(probs[:, 1].cpu().numpy())
            severe_probs.extend(probs[:, 2].cpu().numpy())

    return predictions, normal_mild_probs, moderate_probs, severe_probs

# Define transforms for preprocessing images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# Streamlit App UI
st.title("MRI Classification with Predictions")

# Directory to save uploaded files
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Series description and condition mapping
condition_mapping = {
    'Sagittal T1': {'left': 'left_neural_foraminal_narrowing', 'right': 'right_neural_foraminal_narrowing'},
    'Axial T2': {'left': 'left_subarticular_stenosis', 'right': 'right_subarticular_stenosis'},
    'Sagittal T2/STIR': 'spinal_canal_stenosis'
}

# File upload and process
uploaded_files_by_series = {}
for series in condition_mapping.keys():
    st.header(f"Upload Images for {series}")
    uploaded_files = st.file_uploader(f"Upload files for {series}", accept_multiple_files=True, key=series)
    
    # Save files to respective directories
    if uploaded_files:
        series_folder = os.path.join(UPLOAD_FOLDER, series)
        os.makedirs(series_folder, exist_ok=True)
        for uploaded_file in uploaded_files:
            file_path = os.path.join(series_folder, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        uploaded_files_by_series[series] = [os.path.join(series_folder, f.name) for f in uploaded_files]

# Start Prediction
if st.button("Start Prediction"):
    st.write("Processing uploaded images...")

    # Combine predictions into a DataFrame
    expanded_rows = []
    for series, file_paths in uploaded_files_by_series.items():
        if not file_paths:
            st.warning(f"No images uploaded for {series}. Skipping...")
            continue

        # Create dataset and dataloader
        dataset = SeriesDataset(file_paths, transform=transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Get the model
        model = models.get(series, None)
        if model:
            st.write(f"Making predictions for {series}...")
            predictions, normal_mild_probs, moderate_probs, severe_probs = predict_series(model, dataloader)

            # Map series to conditions and generate rows
            conditions = condition_mapping[series]
            if isinstance(conditions, str):  # Single condition
                conditions = {'left': conditions, 'right': conditions}

            for side, condition in conditions.items():
                for i, file_path in enumerate(file_paths):
                    expanded_rows.append({
                        'row_id': f"{os.path.basename(file_path).split('.')[0]}_{condition}",
                        'normal_mild': normal_mild_probs[i],
                        'moderate': moderate_probs[i],
                        'severe': severe_probs[i]
                    })
        else:
            st.error(f"Model for {series} not found!")


# Create final DataFrame for results
    if expanded_rows:
        results_df = pd.DataFrame(expanded_rows)
        st.write("Final Predictions:")
        st.dataframe(results_df)

        # Generate HTML table in desired format
        html_table = results_df.to_html(index=False, escape=False)
        st.markdown(html_table, unsafe_allow_html=True)

        # Provide download option for CSV
        csv_buffer = BytesIO()
        results_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv_buffer.getvalue(),
            file_name="predictions.csv",
            mime="text/csv"
        )
