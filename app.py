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
    def _init_(self, model_name='efficientnet_b0', num_classes=3, pretrained=True):
        super(CustomEfficientNet, self)._init_()
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
# Directory to save uploaded files
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize session state for uploaded files
if "uploaded_files_by_series" not in st.session_state:
    st.session_state.uploaded_files_by_series = {}
    
# css for sidebar
st.markdown(
    """
    <style>
    
   
    /* Make sidebar title bigger */
    section[data-testid="stSidebar"] h1 {
        font-size: 50px !important;
    }
    section[data-testid="stSidebar"] h2 {
        font-size: 25px !important;
    }
    
    
    </style>
    """,
    unsafe_allow_html=True
)
# Sidebar navigation
st.sidebar.title("FEATURES")
st.sidebar.markdown("<h2> Navigate </h2>", unsafe_allow_html=True)
page = st.sidebar.radio("", ["Home", "Upload MRI Images", "Predictions"])


# Series description and condition mapping
condition_mapping = {
    'Sagittal T1': {'left': 'left_neural_foraminal_narrowing', 'right': 'right_neural_foraminal_narrowing'},
    'Axial T2': {'left': 'left_subarticular_stenosis', 'right': 'right_subarticular_stenosis'},
    'Sagittal T2/STIR': 'spinal_canal_stenosis'
}
# Add CSS for background



# Home Page
if page == "Home":
    
    # Add CSS for solid pink background
    st.markdown(
    """
    <style>
    /* Set the background color for the entire page */
    .stApp {
        background: linear-gradient(#020024, #082a34,#0e612f);
        color:#F1F1F2;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    
    

    


    st.title("LUMBAR DEGENERATIVE CLASSIFICATION")

    st.image("ap2.jpg", caption="Spine MRI", width=600)

    
    # Customizing the 'OVERVIEW' with a fade-in animation
    st.markdown("""
    <style>
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    h2 {
        font-size: 26px;
        font-family: 'Times New Roman', serif; /* Change to your desired font */
        animation: fadeIn 2s ease-out;
    }
    </style>
    <h2><u>OVERVIEW</u></h2>
    """, unsafe_allow_html=True)

    # Overview section with fade-in animation
    st.markdown("""
    <style>
    @keyframes slideUp {
        0% { transform: translateY(50%); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
    p {
        font-family: 'Times New Roman', serif; /* Change to your desired font */
        font-size: 18px;
        animation: slideUp 1s ease-out;
    }
    </style>
    <p>Welcome to the <strong>MRI Classification Platform</strong>! This tool leverages cutting-edge machine learning models to help classify degenerative spine conditions from MRI images.</p>
    """, unsafe_allow_html=True)

    # Feature points with slide-up animation
    st.markdown("""
    <style>
    @keyframes slideUp {
        0% { transform: translateY(50%); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
    p.feature {
        font-family: 'Times New Roman', serif; /* Change to your desired font */
        font-size: 18px;
        animation: slideUp 3s ease-out;
    }
    </style>
    <p class="feature">1.    <strong><u>UPLOAD MRI IMAGES </u> :</strong> Upload images for different MRI series (SAGITTAL T1, AXIAL, and SAGITTAL T2) and classify them into categories based on their condition. 
        Use the Upload MRI Images section to submit images for each series. </p>
    <p class="feature">2.    After uploading the images, go to <strong><u>PREDICTIONS</u></strong>, click on **Start Prediction** to run the model. 
        View predictions on the uploaded MRI images and download the results as a CSV file.</p>
    <p class="feature">3.    You can also remove all the uploaded pictures/files after predictions and downloadng CSV fies by clicking on the same named button <strong><u>RESET</u></strong> button.</p>

    <p class="feature">4.    The model has been trained on a robust dataset, achieving high accuracy in classification tasks.</p>

    """, unsafe_allow_html=True)
    
    # Disclaimer with fade-in animation
    st.markdown("""
    <style>
   @keyframes slideIn {
        0% { transform: translateX(-100%); opacity: 0; }
        100% { transform: translateX(0); opacity: 1; }
    }
    p.disclaimer {
        font-family: Courier new;
        font-size: 18px;
        animation: slideIn 1s ease-out;
    }
    </style>
    <p class="disclaimer"> <strong>DISCLAIMER:</strong> This tool is designed to assist in the classification of spinal conditions. It is not a substitute for professional medical diagnosis. Always consult with a healthcare professional for definitive diagnosis.</p>
    """, unsafe_allow_html=True)




# Upload Images Page
elif page == "Upload MRI Images":
    st.markdown(
    """
    <style>
    /* Set the background color for the entire page */
    .stApp {
        background: linear-gradient(#022a40, #09481d, #032913);
        color:#F1F1F2;        
        font-family: 'Times New Roman', serif; /* Change to your desired font */



    }
    
   
    </style>
    """,
    unsafe_allow_html=True
    )
    st.title("Upload Images for Classification")
    uploaded_files_by_series = st.session_state.uploaded_files_by_series
    for series in condition_mapping.keys():
        with st.expander(f"Upload Images for {series}"):
            uploaded_files = st.file_uploader(
                f"Upload files for {series}", 
                accept_multiple_files=True, 
                key=series, 
                help="You can select multiple images for this series."
            )
            if uploaded_files:
                series_folder = os.path.join(UPLOAD_FOLDER, series)
                os.makedirs(series_folder, exist_ok=True)
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(series_folder, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                st.session_state.uploaded_files_by_series[series] = [
                    os.path.join(series_folder, f.name) for f in uploaded_files
                ]
                st.success(f"{len(uploaded_files)} files uploaded successfully for {series}!")



# Predictions Page
elif page == "Predictions":
    st.markdown(
    """
    <style>
    /* Set the background color for the entire page */
    .stApp {
        background: linear-gradient(#022a40, #082219, #184258);
        color:#F1F1F2;        
        font-family: 'Times New Roman', serif; /* Change to your desired font */
    }
    
    </style>
    """,
    unsafe_allow_html=True
    )
    st.title("Run Predictions")
    if "uploaded_files_by_series" not in st.session_state or not st.session_state.uploaded_files_by_series:
        st.warning("No images uploaded yet. Please upload images in the 'Upload Images' section.")
    else:
        uploaded_files_by_series = st.session_state.uploaded_files_by_series
        if st.button("Start Prediction"):
            st.info("Processing uploaded images...")
            expanded_rows = []

            # Loop through uploaded files and predict
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
                    with st.spinner(f"Making predictions for {series}..."):
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

            # Display results
            if expanded_rows:
                results_df = pd.DataFrame(expanded_rows)
                with st.expander("View Predictions"):
                    st.dataframe(results_df)

                # Provide download option for CSV
                csv_buffer = BytesIO()
                results_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="predictions.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No predictions were made. Please check uploaded files.")
    
    # Clear All Uploaded Files Button
    if st.button("Clear All Uploaded Files"):
        # Clear all uploaded files in the session state
        st.session_state.uploaded_files_by_series = {}
        st.success("All uploaded files have been cleared.")
        