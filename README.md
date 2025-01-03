# SMART VISION

## Overview
SMART VISION is a web application designed to help users detect the freshness of fruits, vegetables, and bread, perform OCR (Optical Character Recognition) on images, and count items in a shopping cart. The app uses deep learning models to make predictions and offers functionalities for account creation, login, and dashboard access.

## Features
- **Signup/Login**: Users can create an account or login with existing credentials to access the application.
- **Dashboard**: Displays user information like name, age, and sex after logging in.
- **Freshness Detector**: Detects the type of fruit/vegetable and its freshness from an uploaded image or captured photo. Additionally, it estimates the shelf life of the fresh produce.
- **Item Counting**: Analyzes images of shopping carts to count the number of items.
- **OCR**: Extracts text information from images using Optical Character Recognition.

## Tech Stack
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: Python, PyTorch, torchvision
- **Database**: JSON-based user management
- **Libraries**: 
  - `torch` for deep learning model inference
  - `pytesseract` for OCR processing
  - `google.generativeai` for any generative AI capabilities
  - `dotenv` for environment variable management

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your Google API key:
   ```
   GOOGLE_API_KEY=<your_api_key>
   ```

4. Ensure `model_fruit_freshness.pth` is placed in the appropriate directory for model loading.

## Usage

1. Run the Streamlit app:
   ```bash
   python -m streamlit run app.py
   ```

2. Open the provided local URL in your browser.

3. Use the **Signup/Login** page to create an account or log in.

4. After logging in, navigate through the following features:
   - **Dashboard**: View your account details.
   - **Freshness Detector**: Upload or capture an image of produce to determine its freshness and estimated shelf life.
   - **Item Counting**: Analyze an image of a shopping cart to count items.
   - **OCR**: Extract text from images of labels, receipts, or other printed material.

## Model Details

- The app uses a custom model based on ResNet18 for image classification.
- The model has been fine-tuned with two output heads:
  - **Fruit/Vegetable Classification**: Identifies 9 classes of fruits/vegetables.
  - **Freshness Detection**: Determines if the item is "Fresh" or "Spoiled".

## Project Structure
```
.
├── app.py                # Main Streamlit app
├── model_fruit_freshness.pth # Pre-trained model weights
├── data.json             # JSON file for user information
├── utils/
│   └── utils.py          # Helper functions for the app
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- The application utilizes ResNet18 architecture from `torchvision`.
- Thanks to the Streamlit community for the ease of building data apps.

