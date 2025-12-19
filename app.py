import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import cv2
import numpy as np
import json
import os

# --- 1. ID TO WORD DICTIONARY ---
# Matches your provided LSA64 table
ID_TO_WORD = {
    "01": "Opaque", "02": "Red", "03": "Green", "04": "Yellow", 
    "05": "Bright", "06": "Light-blue", "07": "Colors", "08": "Pink",
    "09": "Women", "10": "Enemy", "11": "Son", "12": "Man",
    "13": "Away", "14": "Drawer", "15": "Born", "16": "Learn",
    "17": "Call", "18": "Skimmer", "19": "Bitter", "20": "Sweet milk",
    "21": "Milk", "22": "Water", "23": "Food", "24": "Argentina",
    "25": "Uruguay", "26": "Country", "27": "Last name", "28": "Where",
    "29": "Mock", "30": "Birthday", "31": "Breakfast", "32": "Photo",
    "33": "Hungry", "34": "Map", "35": "Coin", "36": "Music",
    "37": "Ship", "38": "None", "39": "Name", "40": "Patience",
    "41": "Perfume", "42": "Deaf", "43": "Trap", "44": "Rice",
    "45": "Barbecue", "46": "Candy", "47": "Chewing-gum", "48": "Spaghetti",
    "49": "Yogurt", "50": "Accept", "51": "Thanks", "52": "Shut down",
    "53": "Appear", "54": "To land", "55": "Catch", "56": "Help",
    "57": "Dance", "58": "Bathe", "59": "Buy", "60": "Copy",
    "61": "Run", "62": "Realize", "63": "Give", "64": "Find"
}

# --- 2. MODEL DEFINITION ---
class CNN_LSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = mobilenet_v2(weights=None)
        base.classifier = nn.Identity()
        self.cnn = base
        self.lstm = nn.LSTM(input_size=1280, hidden_size=512, batch_first=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        feat = self.cnn(x)
        feat = feat.reshape(B, T, 1280)
        out, _ = self.lstm(feat)
        return self.fc(out[:, -1])

# --- 3. RESOURCE LOADING ---
@st.cache_resource
def load_resources():
    with open("class_names.json", "r") as f:
        classes = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_LSTM(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load("best_sign_model.pth", map_location=device))
    model.eval()
    return model, classes, device

# --- 4. LIVE TRANSFORMER ---
class SignLanguageTransformer(VideoTransformerBase):
    def __init__(self, model, classes, device):
        self.model = model
        self.classes = classes
        self.device = device
        self.frame_buffer = []
        self.prediction_text = "Waiting..."
        self.frame_count = 0 # To track stride

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.frame_count % 3 == 0:
            process_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            process_img = cv2.resize(process_img, (112, 112))
            tensor_img = torch.from_numpy(process_img).permute(2, 0, 1).float() / 255.0
            self.frame_buffer.append(tensor_img)
        
        # Preprocessing for MobileNetV2
        process_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        process_img = cv2.resize(process_img, (112, 112))
        tensor_img = torch.from_numpy(process_img).permute(2, 0, 1).float() / 255.0
        
        self.frame_buffer.append(tensor_img)
        
        # Prediction logic when 16 frames are collected
        if len(self.frame_buffer) == 16:
            input_tensor = torch.stack(self.frame_buffer).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(input_tensor)
                idx = torch.argmax(output, dim=1).item()
                
                # Mapping ID (e.g., "001") to Word (e.g., "Opaque")
                raw_id = self.classes[idx]
                short_id = raw_id[-2:] # Get last 2 digits
                self.prediction_text = ID_TO_WORD.get(short_id, f"ID: {raw_id}")
                
            self.frame_buffer = [] # Reset buffer

        # Display word on screen
        cv2.putText(img, f"Sign: {self.prediction_text}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img

# --- 5. STREAMLIT INTERFACE ---
st.title("🤟 Real-Time Sign Language Translator")
model, classes, device = load_resources()

webrtc_streamer(
    key="sign-translator",
    video_transformer_factory=lambda: SignLanguageTransformer(model, classes, device),
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)