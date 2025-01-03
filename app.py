import streamlit as st
import os
from dotenv import load_dotenv
import json
import torch 
import torchvision.transforms as transforms
import torchvision
import numpy as np
import torch.nn as nn
import pickle
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
import av
import io
from PIL import Image
from utils import *
import queue
from PIL import Image, ImageEnhance
from database import initialize_database, create_account, check_login , insert_freshness_record,insert_ocr_record , insert_item_counting_record
import cv2
session_state = st.session_state
if "user_index" not in st.session_state:
    st.session_state["user_index"] = 0
load_dotenv()
#genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
#OpenAI(api_key=os.environ["OPENAI_API_KEY"])

from typing import Union

import streamlit.components.v1 as components
import base64
import io

def camera_input_client():
    """Function to handle camera input using client-side camera"""
    # HTML and JavaScript for camera capture
    camera_html = """
        <!DOCTYPE html>
        <html>
        <body>
            <div style="display: flex; flex-direction: column; align-items: center; margin-bottom: 20px;">
                <video id="video" width="640" height="480" autoplay playsinline style="border-radius: 10px; margin-bottom: 10px;"></video>
                <button id="capture" style="
                    background-color: #4CAF50;
                    border: none;
                    color: white;
                    padding: 12px 30px;
                    text-align: center;
                    text-decoration: none;
                    font-size: 16px;
                    margin: 4px 2px;
                    cursor: pointer;
                    border-radius: 8px;">Capture Image</button>
                <canvas id="canvas" width="640" height="480" style="display: none;"></canvas>
                <div id="error-message" style="color: red; padding: 10px;"></div>
            </div>

            <script>
                const video = document.getElementById('video');
                const canvas = document.getElementById('canvas');
                const captureButton = document.getElementById('capture');
                const errorMessage = document.getElementById('error-message');
                let stream = null;

                // Function to initialize the camera
                async function initCamera() {
                    try {
                        stream = await navigator.mediaDevices.getUserMedia({
                            video: {
                                width: { ideal: 640 },
                                height: { ideal: 480 },
                                facingMode: { ideal: 'environment' }
                            },
                            audio: false
                        });
                        video.srcObject = stream;
                        await video.play();
                        errorMessage.textContent = '';
                    } catch (err) {
                        errorMessage.textContent = 'Error accessing camera: ' + err.message;
                        console.error('Error:', err);
                    }
                }

                // Initialize camera when page loads
                initCamera();

                // Function to capture image
                captureButton.addEventListener('click', function() {
                    try {
                        const context = canvas.getContext('2d');
                        context.drawImage(video, 0, 0, canvas.width, canvas.height);
                        
                        // Convert to base64
                        const imageData = canvas.toDataURL('image/jpeg', 0.95);
                        
                        // Send to Streamlit
                        window.Streamlit.setComponentValue(imageData);
                    } catch (err) {
                        errorMessage.textContent = 'Error capturing image: ' + err.message;
                        console.error('Capture error:', err);
                    }
                });

                // Cleanup when component is destroyed
                window.addEventListener('beforeunload', () => {
                    if (stream) {
                        stream.getTracks().forEach(track => track.stop());
                    }
                });
            </script>
        </body>
        </html>
    """

    # Create the component
    captured_image = components.html(camera_html, height=600)
    
    # Process the captured image
    if captured_image:
        try:
            # Verify we have valid base64 image data
            if isinstance(captured_image, str) and captured_image.startswith('data:image/jpeg;base64,'):
                # Extract the base64 data
                base64_data = captured_image.split(',')[1]
                # Convert to bytes
                image_bytes = base64.b64decode(base64_data)
                # Create PIL Image
                image = Image.open(io.BytesIO(image_bytes))
                # Display the captured image
                st.image(image, caption="Captured Image", width=300)
                return image
            else:
                st.error("Invalid image data received")
                return None
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return None
    
    return None
fruit_vegetable_mapping = {
    0: "apples",
    1: "banana",
    2: "bittergourd",
    3: "capsicum",
    4: "cucumber",
    5: "okra",
    6: "oranges",
    7: "potato",
    8: "tomato"
}

# Map for freshness labels (0 = Fresh, 1 = Spoiled)
freshness_mapping = {0: "Fresh", 1: "Spoiled"}
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.7
        self.base = torchvision.models.resnet18(pretrained=True)

        # Freeze some of the base layers
        for param in list(self.base.parameters())[:-15]:
            param.requires_grad = False

        self.base.classifier = nn.Sequential()
        self.base.fc = nn.Sequential()

        # Define additional blocks for the model
        self.block1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
        )

        self.block2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 9)
        )

        self.block3 = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.base(x)
        x = self.block1(x)
        y1, y2 = self.block2(x), self.block3(x)
        return y1, y2

@st.cache_resource(show_spinner=False)
def load_resource():
    model = Model()

    model.load_state_dict(torch.load('model_fruit_freshness.pth', map_location=torch.device('cpu')))


    model.eval()

    return model

# Function to make predictions on uploaded image
def make_prediction(image, model):
    # Ensure the image is converted to RGB (in case it's RGBA or other formats)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Make predictions
    with torch.no_grad():
        pred_fruit, pred_fresh = model(img_tensor)
        fruit_classes = torch.argmax(pred_fruit, dim=1).item()  # Get fruit class
        fresh_classes = torch.argmax(pred_fresh, dim=1).item()  # Get freshness class

    return fruit_classes, fresh_classes
# def initialize_database(json_file_path="data.json"):
#     try:
#         if not os.path.exists(json_file_path):
#             data = {"users": []}
#             with open(json_file_path, "w") as json_file:
#                 json.dump(data, json_file)
#     except Exception as e:
#         print(f"Error initializing database: {e}")

# def create_account(
#     name,
#     email,
#     age,
#     sex,
#     password,
#     json_file_path="data.json",
# ):
#     try:
#         if not os.path.exists(json_file_path) or os.stat(json_file_path).st_size == 0:
#             data = {"users": []}
#         else:
#             with open(json_file_path, "r") as json_file:
#                 data = json.load(json_file)

#         user_info = {
#             "name": name,
#             "email": email,
#             "age": age,
#             "sex": sex,
#             "password": password,
         
#         }
#         data["users"].append(user_info)

#         with open(json_file_path, "w") as json_file:
#             json.dump(data, json_file, indent=4)

#         return user_info
#     except json.JSONDecodeError as e:
#         st.error(f"Error decoding JSON: {e}")
#         return None
#     except Exception as e:
#         st.error(f"Error creating account: {e}")
#         return None
def signup():
    st.title("Signup Page")
    with st.form("signup_form"):
        st.write("Fill in the details below to create an account:")
        name = st.text_input("Name:")
        email = st.text_input("Email:")
        age = st.number_input("Age:", min_value=0, max_value=120)
        sex = st.radio("Sex:", ("Male", "Female", "Other"))
        password = st.text_input("Password:", type="password")
        confirm_password = st.text_input("Confirm Password:", type="password")

        if st.form_submit_button("Signup"):
            if password == confirm_password:
                user = create_account(name, email, age, sex, password)
                if user:
                    st.session_state["logged_in"] = True
                    st.session_state["user_info"] = user
                    st.success("Account created successfully! You are now logged in.")
            else:
                st.error("Passwords do not match. Please try again.")

# def check_login(username, password, json_file_path="data.json"):
#     try:
#         with open(json_file_path, "r") as json_file:
#             data = json.load(json_file)

#         for user in data["users"]:
#             if user["email"] == username and user["password"] == password:
#                 st.session_state["logged_in"] = True
#                 st.session_state["user_info"] = user
#                 st.success("Login successful!")
#                 return user
#         return None
#     except Exception as e:
#         st.error(f"Error checking login: {e}")
#         return None
def login():
    st.title("Login Page")
    username = st.text_input("Email:")
    password = st.text_input("Password:", type="password")

    login_button = st.button("Login")

    if login_button:
        user = check_login(username, password)
        if user is not None:
            st.session_state["logged_in"] = True
            st.session_state["user_info"] = user
            st.success("Login successful!")
        else:
            st.error("Invalid credentials. Please try again.")

# def process_image(pil_img):
#     # Convert PIL image to a NumPy array
#     img = np.array(pil_img)
    
#     # Convert image to grayscale
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
#     # Remove noise
#     kernel = np.ones((1, 1), np.uint8)
#     img = cv2.dilate(img, kernel, iterations=1)
#     img = cv2.erode(img, kernel, iterations=1)
    
#     # Convert back to PIL image for sharpening
#     pil_img = Image.fromarray(img)
    
#     # Increase sharpness
#     enhancer = ImageEnhance.Sharpness(pil_img)
#     img = enhancer.enhance(2.0)  # Increase sharpness

#     # Perform OCR on the processed image (now in PIL format)
#     result = pytesseract.image_to_string(img)
#     return result

def render_dashboard(user_info, json_file_path="data.json"):
    try:
        st.title(f"Welcome to the Dashboard, {user_info['name']}!")
        st.subheader("User Information:")
        st.write(f"Name: {user_info['name']}")
        st.write(f"Sex: {user_info['sex']}")
        st.write(f"Age: {user_info['age']}")

    except Exception as e:
        st.error(f"Error rendering dashboard: {e}")
def main(json_file_path="data.json"):
    st.set_page_config(page_title="SMART VISION", layout="wide", page_icon="🍎")
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://st2.depositphotos.com/6469658/9824/i/380/depositphotos_98245024-stock-photo-wave-band-abstract-background-surface.jpg");
    background-size: 180%;
    background-position: top left;
    background-repeat: no-repeat;
    background-attachment: local;
    }}

    [data-testid="stSidebar"] > div:first-child {{
    background-image: url("https://st2.depositphotos.com/6469658/9824/i/380/depositphotos_98245024-stock-photo-wave-band-abstract-background-surface.jpg");
    background-position: center; 
    background-repeat: no-repeat;
    background-attachment: fixed;
    }}

    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}

    [data-testid="stToolbar"] {{
    right: 2rem;
    }}
    </style>
    """

    st.markdown(page_bg_img, unsafe_allow_html=True)
    # Add custom CSS to style buttons, headers, and more
    st.markdown("""
        <style>
        .title {
            font-size: 36px;
            color: #007BFF;
            font-family: 'Arial Black', sans-serif;
            text-align: center;
            margin-bottom: 20px;
        }
        .subtitle {
            font-size: 24px;
            color: #4B8BFF;
            font-family: 'Arial', sans-serif;
            text-align: center;
            margin-bottom: 15px;
        }
        .section-header {
            color: #FF6F61;
            font-family: 'Georgia', serif;
            font-size: 28px;
            margin-top: 25px;
            border-bottom: 2px solid #FF6F61;
        }
        .upload-btn, .capture-btn {
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 12px 30px;
            text-align: center;
            text-decoration: none;
            font-size: 18px;
            margin: 10px auto;
            cursor: pointer;
            border-radius: 8px;
            width: 80%;
            display: block;
        }
        .results {
            border-radius: 10px;
            background-color: #F9F9F9;
            padding: 20px;
            margin-top: 20px;
            border: 1px solid #E0E0E0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            line-height: 1.6;
            font-size: 16px;
        }
        .info-box {
            padding: 15px;
            background-color: #F0F8FF;
            margin: 20px 0;
            border-radius: 10px;
            border-left: 5px solid #007BFF;
        }
        .highlight {
            background-color: #FFEB3B;
            padding: 2px 5px;
            border-radius: 5px;
        }
        .box {
            border: 2px solid #007BFF;
            background-color: #F0F8FF;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .result-box {
            border: 2px solid #4CAF50;
            background-color: #eaffea;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .highlight-result {
            background-color: #d4edda;
            padding: 10px;
            border-radius: 10px;
            border-left: 4px solid #28a745;
        }
        hr.custom-hr {
            border: none;
            border-top: 3px solid #007BFF;
            width: 60%;
            margin: 20px auto;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
        }

        
        </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("SMART VISION")
    page = st.sidebar.radio(
        "Go to",
        (
            "Signup/Login",
            "Dashboard",
            "Freshness Detector",
            "Item_Counting",
            "OCR"
        ),
        key="pages",
    )
   

   

    if page == "Signup/Login":
        st.title("Signup/Login Page")
        login_or_signup = st.radio(
            "Select an option", ("Login", "Signup"), key="login_signup"
        )
        if login_or_signup == "Login":
            login()
        else:
            signup()

    elif page == "Dashboard":
        if session_state.get("logged_in"):
            render_dashboard(session_state["user_info"])
        else:
            st.warning("Please login/signup to view the dashboard.")
            
    elif page == "Freshness Detector":
        if st.session_state.get("logged_in"):
            # Title and header styling
            st.markdown('<div class="title">Product Freshness and Shelf Life Estimator</div>', unsafe_allow_html=True)

            # Custom CSS for styling the upload section and buttons
            st.markdown("""
                <style>
                .file-uploader {
                    border: 2px dashed #007BFF;
                    background-color: #f0f8ff;
                    padding: 15px;
                    border-radius: 10px;
                    text-align: center;
                    margin-bottom: 20px;
                }
                .upload-btn, .capture-btn {
                    background-color: #4CAF50;
                    border: none;
                    color: white;
                    padding: 12px 30px;
                    text-align: center;
                    text-decoration: none;
                    font-size: 16px;
                    margin: 10px auto;
                    cursor: pointer;
                    border-radius: 8px;
                    display: inline-block;
                }
                .instructions {
                    font-size: 16px;
                    color: #007BFF;
                    font-family: 'Arial', sans-serif;
                    text-align: center;
                    margin-bottom: 20px;
                }
                .result-box {
                    border: 2px solid #4CAF50;
                    background-color: #eaffea;
                    padding: 20px;
                    border-radius: 10px;
                    margin-top: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .highlight {
                    background-color: #d4edda;
                    padding: 10px;
                    border-radius: 10px;
                    border-left: 4px solid #28a745;
                }
                hr.custom-hr {
                    border: none;
                    border-top: 3px solid #007BFF;
                    width: 60%;
                    margin: 20px auto;
                }
                </style>
            """, unsafe_allow_html=True)

            # Product Type and Capture Method selection with styling
            st.markdown('<div class="instructions">Select Product Type and how to upload the image</div>', unsafe_allow_html=True)
            product_type = st.selectbox("Select Product Type", ["Fruit/Vegetable", "Bread"])
            capture_method = st.selectbox("Capture Method", ["Upload an image", "Capture image from camera"])

            image = None

            # Upload image section with custom styling
            if capture_method == "Upload an image":
                st.markdown('<div class="file-uploader">Upload an image of the selected product (.jpg, .png, .jpeg)</div>', unsafe_allow_html=True)
                uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])  # File uploader box styling
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_column_width=False, width=300)  # Adjust the image size

            # Capture from camera section with custom button styling
            #elif capture_method == "Capture image from camera":
                # capture_image = st.button("Capture Image", key="capture_btn", help="Click to capture an image")

                # # Initialize the camera feed
                # cap = cv2.VideoCapture(2)
                # st_frame = st.empty()  # Create a placeholder for the video frame

                # while True:
                #     ret, frame = cap.read()
                #     if not ret:
                #         st.warning("Failed to access the camera.")
                #         break

                #     # Convert the frame to RGB for displaying in Streamlit
                #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #     st_frame.image(frame_rgb, channels="RGB", width=300)  # Show the camera feed

                #     # Check if the button was pressed
                #     if capture_image:
                #         image = Image.fromarray(frame_rgb)  # Convert to PIL image
                #         st.image(image, caption="Captured Image", width=300)  # Show captured image
                #         break

                # cap.release() 
            elif capture_method == "Capture image from camera":
                try:
                    image = camera_input_client()
                    if image is not None:
                        # The image is already displayed in the function
                        continue_processing = True
                except Exception as e:
                    st.error(f"Camera error: {str(e)}")
                    continue_processing = False

            # Display predicted freshness information if image is available
            if image is not None:
                st.success("Image captured successfully!")
                model = load_resource()

                if product_type == "Fruit/Vegetable":
                    fruit_label, freshness_label = make_prediction(image, model)
                    fruit_name = fruit_vegetable_mapping[fruit_label]
                    freshness = freshness_mapping[freshness_label]

                    st.markdown(f"<div class='result-box'><strong>Predicted Item:</strong> <span class='highlight'>{fruit_name}</span></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='result-box'><strong>Freshness Status:</strong> <span class='highlight'>{freshness}</span></div>", unsafe_allow_html=True)

                    shelf_life_days = None  # Initialize variable

                    if freshness_label == 0:  # Only show shelf life for fresh items
                        try:
                            shelf_life_info = fruit_vegetable_shelf_life(image)
                            shelf_life_info = shelf_life_info.replace("```json", "").replace("```", "").strip()
                            info_dict = json.loads(shelf_life_info)
                            
                            shelf_life_days = info_dict.get('shell life', 'N/A')
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"""
                                    <div style='background-color: #e3f2fd; padding: 15px; border-radius: 10px;'>
                                        <h4 style='color: #1976d2;'>Estimated Shelf Life</h4>
                                        <p>{shelf_life_days}</p>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                if info_dict.get('additional content'):
                                    st.markdown(f"""
                                        <div style='background-color: #f3e5f5; padding: 15px; border-radius: 10px;'>
                                            <h4 style='color: #7b1fa2;'>Additional Information</h4>
                                            <p>{info_dict['additional content']}</p>
                                        </div>
                                    """, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error processing freshness information: {str(e)}")
              
                    
                    insert_freshness_record(
                        produce=fruit_name,
                        freshness=freshness,
                        expected_life_span_days=shelf_life_days
                    )
                    st.success("Detection results saved to the database.")
                       
                    
                elif product_type == "Bread":
                    try:
                        bread_info = bread_freshness(image)
                        bread_info = bread_info.replace("```json", "").replace("```", "").strip()
                        info_dict = json.loads(bread_info)
                        
                        freshness_status = 'Fresh ✅' if info_dict['freshness'] == 'Yes' else 'Not Fresh ❌'
                        shelf_life_days = info_dict.get('shelf_life', 'N/A')
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"""
                                <div style='background-color: #e8f5e9; padding: 15px; border-radius: 10px;'>
                                    <h4 style='color: #2e7d32;'>Freshness Status</h4>
                                    <p>{freshness_status}</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                                <div style='background-color: #fff3e0; padding: 15px; border-radius: 10px;'>
                                    <h4 style='color: #ef6c00;'>Expected Shelf Life</h4>
                                    <p>{shelf_life_days}</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # Insert into database
                       
                            try:
                                insert_freshness_record(
                                    produce='Bread',
                                    freshness='Fresh' if info_dict['freshness'] == 'Yes' else 'Not Fresh',
                                    expected_life_span_days=shelf_life_days
                                )
                                st.success("Detection results saved to the database.")
                            except Exception as e:
                                st.error(f"Error storing data in database: {str(e)}")
                       
                    
                    except Exception as e:
                        st.error(f"Error processing bread information: {str(e)}")

        else:
            st.warning('Please Login to use the app!!')

    elif page == "Item_Counting":
        if st.session_state.get("logged_in"):
            # Title and header styling
            st.markdown('<div class="title">Shopping Cart Item Counter</div>', unsafe_allow_html=True)

            # Custom CSS for styling the upload section and buttons
            st.markdown("""
                <style>
                .file-uploader {
                    border: 2px dashed #007BFF;
                    background-color: #f0f8ff;
                    padding: 15px;
                    border-radius: 10px;
                    text-align: center;
                    margin-bottom: 20px;
                }
                .upload-btn, .capture-btn {
                    background-color: #4CAF50;
                    border: none;
                    color: white;
                    padding: 12px 30px;
                    text-align: center;
                    text-decoration: none;
                    font-size: 16px;
                    margin: 10px auto;
                    cursor: pointer;
                    border-radius: 8px;
                    display: inline-block;
                }
                .instructions {
                    font-size: 16px;
                    color: #007BFF;
                    font-family: 'Arial', sans-serif;
                    text-align: center;
                    margin-bottom: 20px;
                }
                .result-box {
                    border: 2px solid #4CAF50;
                    background-color: #eaffea;
                    padding: 20px;
                    border-radius: 10px;
                    margin-top: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                </style>
            """, unsafe_allow_html=True)

            # Capture Method selection with styling
            st.markdown('<div class="instructions">Select how you would like to upload the image of the shopping cart</div>', unsafe_allow_html=True)
            capture_method = st.selectbox("Capture Method", ["Upload an image", "Capture image from camera"])

            image = None

            # Upload image section with custom styling
            if capture_method == "Upload an image":
                st.markdown('<div class="file-uploader">Upload an image of the shopping cart (.jpg, .png, .jpeg)</div>', unsafe_allow_html=True)
                uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])  # File uploader box styling
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_column_width=False, width=300)  # Adjust the image size

            # Capture from camera section with custom button styling
            elif capture_method == "Capture image from camera":
                try:
                    image = camera_input_client()
                    if image is not None:
                        # The image is already displayed in the function
                        continue_processing = True
                except Exception as e:
                    st.error(f"Camera error: {str(e)}")
                    continue_processing = False
                # capture_image = st.button("Capture Image", key="capture_btn", help="Click to capture an image")

                # # Initialize the camera feed
                # cap = cv2.VideoCapture(2)
                # st_frame = st.empty()  # Create a placeholder for the video frame

                # while True:
                #     ret, frame = cap.read()
                #     if not ret:
                #         st.warning("Failed to access the camera.")
                #         break

                #     # Convert the frame to RGB for displaying in Streamlit
                #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #     st_frame.image(frame_rgb, channels="RGB", width=300)  # Show the camera feed

                #     # Check if the button was pressed
                #     if capture_image:
                #         image = Image.fromarray(frame_rgb)  # Convert to PIL image
                #         st.image(image, caption="Captured Image", width=300)  # Show captured image
                #         break

                # cap.release() 
                #image = camera_input_client()
            # Display detected item information if image is available
            if image is not None:
                try:
                    items_info = count_cart_items(image)  # Your function to process the image and return JSON
                    
                    # Display the raw items_info for debugging
                    #st.subheader("Raw Output from `count_cart_items`")
                    #st.text(items_info)

                    # Clean up the JSON string
                    items_info = items_info.replace("```json", "").replace("```", "")  # Remove code block markers
                    # Removed the potentially corrupting replacement
                    # items_info = items_info.replace('"s', "'s")  # Fix the apostrophe issue

                    # Display the cleaned items_info for debugging
                    #st.subheader("Cleaned JSON String")
                    #st.text(items_info)

                    # Validate JSON
                    
                    items_data = json.loads(items_info)

                    st.markdown("""
                        <div class='result-box'>
                            <h3 style='color: #2E86C1; margin-bottom: 15px;'>Items Detected</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Create two columns for the categories
                    col1, col2 = st.columns(2)
                    
                    # Categories with their corresponding emojis
                    categories = [
                        ('Fruits', '🍎'), 
                        ('Vegetables', '🥕'), 
                        ('Packed Goods', '📦'), 
                        ('Beverages', '🥤'),
                        ('Bakery Essentials', '🥖')
                    ]
                    
                    total_items_detected = 0  # Initialize total items count

                    for i, (category, emoji) in enumerate(categories):
                        with col1 if i % 2 == 0 else col2:
                            st.markdown(f"""
                                <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 5px;'>
                                    <h4 style='color: #34495E;'>{emoji} {category}</h4>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            if category in items_data and items_data[category]:
                                for item in items_data[category]:
                                    st.markdown(f"""
                                        <div style='padding: 5px 15px; margin: 2px 0;'>
                                            <span style='color: #2C3E50;'>• {item}</span>
                                        </div>
                                    """, unsafe_allow_html=True)
                                # Update total items count
                                total_items_detected += len(items_data[category])
                            else:
                                st.markdown("""
                                    <div style='padding: 5px 15px; color: #7F8C8D; font-style: italic;'>
                                        No items detected
                                    </div>
                                """, unsafe_allow_html=True)

                    # Insert the record into the database with the total count
                    insert_item_counting_record(items_data, total_items_detected)
                    st.markdown(f"<p style='font-size:16px; color:#2E86C1;'>Total Items Detected: {total_items_detected}</p>", unsafe_allow_html=True)
        
           
                except Exception as e:
                    st.error(f"Error processing items: {str(e)}")
                # For debugging
                #st.write("Debug - items_info after cleaning:", items_info)
        else:
            # If not logged in, show warning
            st.warning('Please Login to use the app!!')


    elif page == "OCR":
        if st.session_state.get("logged_in"):
            # Title and header styling
            st.markdown('<div class="title">Image Uploader and OCR App</div>', unsafe_allow_html=True)

            # Custom CSS for styling the upload section and buttons
            st.markdown("""
            
            <style>
            .file-uploader {
                border: 2px dashed #007BFF;
                background-color: #f0f8ff;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                margin-bottom: 20px;
            }
            .upload-btn, .capture-btn {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 12px 30px;
                text-align: center;
                text-decoration: none;
                font-size: 16px;
                margin: 10px auto;
                cursor: pointer;
                border-radius: 8px;
                display: inline-block;
            }
            .instructions {
                font-size: 16px;
                color: #007BFF;
                font-family: 'Arial', sans-serif;
                text-align: center;
                margin-bottom: 20px;
            }
            .result-box {
                border: 2px solid #4CAF50;
                background-color: #eaffea;
                padding: 20px;
                border-radius: 10px;
                margin-top: 40px;  /* Adjusted margin-top to give more space */
                margin-bottom: 40px;  /* Adjusted margin-bottom to give more space */
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .highlight-result {
                font-size: 16px;
                color: #333;
                font-family: 'Arial', sans-serif;
                margin-top: 20px;
                display: block;
            }
            </style>

            """, unsafe_allow_html=True)

            # Capture Method selection with styling
            st.markdown('<div class="instructions">Select how you would like to upload the image</div>', unsafe_allow_html=True)
            capture_method = st.selectbox("Capture Method", ["Upload an image", "Capture image from camera"])

            image = None

            # Upload image section with custom styling
            if capture_method == "Upload an image":
                st.markdown('<div class="file-uploader">Upload an image (.jpg, .png, .jpeg)</div>', unsafe_allow_html=True)
                uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])  # File uploader box styling
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", width=300)  # Adjust the image size

            # Capture from camera section with custom button styling
            elif capture_method == "Capture image from camera":
                try:
                    image = camera_input_client()
                    if image is not None:
                        # The image is already displayed in the function
                        continue_processing = True
                except Exception as e:
                    st.error(f"Camera error: {str(e)}")
                    continue_processing = False
                # capture_image = st.button("Capture Image", key="capture_btn", help="Click to capture an image")

                # # Initialize the camera feed
                # cap = cv2.VideoCapture(2)
                # st_frame = st.empty()  # Create a placeholder for the video frame

                # while True:
                #     ret, frame = cap.read()
                #     if not ret:
                #         st.warning("Failed to access the camera.")
                #         break

                #     # Convert the frame to RGB for displaying in Streamlit
                #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #     st_frame.image(frame_rgb, channels="RGB", width=300)  # Show the camera feed

                #     # Check if the button was pressed
                #     if capture_image:
                #         image = Image.fromarray(frame_rgb)  # Convert to PIL image
                #         st.image(image, caption="Captured Image",  width=300)  # Show captured image
                #         break

                # cap.release() 
                #image = camera_input_client()

            # Display extracted information if image is available
            if image is not None:
                try:
                    product_info = extract_product_info(image)
                    
                    # Clean and parse the JSON string
                    # Remove ```json and ``` markers and any extra whitespace
                    cleaned_json = product_info.replace("```json", "").replace("```", "").strip()
                    info_dict = json.loads(cleaned_json)

                    # Define fields with their icons
                    fields = [
                        ('Product Name', '📦'),
                        ('Brand Name', '™️'),
                        ('Type of Product', '🏷️'),
                        ('Batch Number', '#️⃣'),
                        ('Year of Manufacturing', '📅'),
                        ('Expiry Date', '⏰'),
                        ('Other relevant Details', 'ℹ️'),
                        ('Utilization time', '⌛')
                    ]

                    st.markdown("""
                        <div class='result-box'>
                            <h3 style='color: #2E86C1; margin-bottom: 15px;'>Product Information</h3>
                        </div>
                    """, unsafe_allow_html=True)

                    # Create two columns for the information
                    col1, col2 = st.columns(2)
                    
                    for idx, (field, emoji) in enumerate(fields):
                        # Check if field exists and is not NA
                        if field in info_dict :
                            with col1 if idx % 2 == 0 else col2:
                                st.markdown(f"""
                                    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 5px;'>
                                        <h4 style='color: #34495E;'>{emoji} {field}</h4>
                                        <p style='margin-top: 8px;'>{info_dict[field]}</p>
                                    </div>
                                """, unsafe_allow_html=True)

                    # Insert the record into the database
                    insert_ocr_record(info_dict)

                except json.JSONDecodeError as e:
                    st.error("Error parsing product information. Please try again.")
                except Exception as e:
                    st.error(f"Error processing product information: {str(e)}")
        else:
            # If not logged in, show warning
            st.warning('Please Login to use the app!!')

if __name__ == "__main__":
    main()
    initialize_database()
