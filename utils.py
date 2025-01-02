# from dotenv import load_dotenv
# import os
# import google.generativeai as genai
# from PIL import Image
# import streamlit as st
from datetime import datetime

# Get the current date (without hours)
current_date = datetime.now().strftime("%Y-%m-%d")

# # Print the current date
# print("Current Date:", current_date)
# load_dotenv()
# genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# @st.cache_resource(show_spinner=False)
# def load_model(): 
#     model = genai.GenerativeModel('gemini-1.5-flash')
#     return model

# def fruit_vegetable_shelf_life(image):
#     prompt = f'''Here is an image of a fruit or vegetable:. Please analyze the image to assess its freshness and determine its approximate
#             remaining shelf life. If the fruit or vegetable appears fresh, provide a shelf life estimation based on visual characteristics (e.g., color, texture).
#             Use concise language (1-2 lines) in your analysis.
#             Never Tell u are an AI Agent'''
#     model = load_model()
    
#     response = model.generate_content([prompt, image])


#     return (response.text)



# def bread_freshness(image):
#     prompt = f'''Here is an image of bread:  Please analyze the image to assess whether the bread is fresh or stale.
#              If fresh, estimate its remaining shelf life based on its visual characteristics (e.g., color, texture). If the manufacturing date (MFD) is available
#              factor that in to provide a more accurate shelf life. Use concise language (1-2 lines).
#              Never Tell u are an AI Agent'''
#     model = load_model()
    
#     response = model.generate_content([prompt, image])
#     return (response.text)



# def count_cart_items(image):
#     prompt = f'''Here is an image of a shopping cart:.
#       Please analyze the image and count the total number of items present. If possible, also provide a brief list of the detected items 
#       (e.g., fruits, vegetables, packaged goods) and categorize them accordingly. Keep the response concise.'''
    
#     model = load_model()
    
#     response = model.generate_content([prompt, image])
#     return (response.text)


# def extract_product_info(image):
#     # Convert the image to the format required by the Gemini API

#     # Define the prompt for Gemini
#     prompt = '''
#     Here is an image of a product. 
#     Please extract the text from the image and identify important details. Specifically, look for the following information:
    
#     - Product Name
#     - Brand Name
#     - Type of Product
#     - Batch Number
#     - Year of Manufacturing
#     - Expiry Date
#     - Other relevant details (if available)
#     - Add Flavour name also if availbale 
#     - If expiry Date is available compare that to the current date and then tell is it good to consume or not or
#     till what point this product we can consume...(do this if expiry date is pressnt) 
#     compare the current date like what date is today and then tell suppose today is 10 oct and then compare find current date and then calculate dont show though
#     with expiry date and then tell the product can be consume like for next 8 months or so according to the expiry date
#     the above is just a exammple calculate how long we can consume according to the present date (subtact the present date by expiry date and then tell in monthsor days)
#     and give the result in the format
    
    
#     - Yes product can be utilized/consumed by the customer for ___ months. or no the product can not be utilised 
  
#     If any of this information is unreadable or not present, please return "Not readable."

 
#     Ensure the extracted information is well-formatted, with proper labels and organized for readability.
#     ''' 
#     # Assuming `gemini_api` is the interface to communicate with the Gemini model
#     gemini_model = load_model()  # Placeholder for loading the model

#     response = gemini_model.generate_content([prompt, image])
    
#     return response.text


from dotenv import load_dotenv
import os
import streamlit as st
from datetime import datetime
from openai import OpenAI
import base64
from PIL import Image
import io

# Get the current date (without hours)
current_date = datetime.now().strftime("%Y-%m-%d")

# Print the current date
print("Current Date:", current_date)

# Load environment variables
#load_dotenv()
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
# Initialize OpenAI client

def load_model():
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def encode_image_to_base64(pil_image):
    # Convert PIL Image to bytes
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    # Encode to base64
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def analyze_image_with_prompt(image, prompt):
    client = load_model()
    
    # Encode image to base64
    base64_image = encode_image_to_base64(image)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=300
    )
    
    return response.choices[0].message.content

def fruit_vegetable_shelf_life(image):
    prompt = '''Here is an image of a fruit or vegetable. Please analyze the image to assess its freshness and determine its approximate
            remaining shelf life. If the fruit or vegetable appears fresh, provide a shelf life estimation based on visual characteristics (e.g., color, texture).
            Use concise language (1-2 lines) in your analysis in json shell life and additional content. Need to give the expected shell life dont give NA for shelf life value
            json format it is{'additional content' : anything what is the additional content , 'shell life': 1-2days} it is just a example If any of this is not avilable write NA
            '''
    return analyze_image_with_prompt(image, prompt)

def bread_freshness(image):
    prompt = '''Here is an image of bread: Please analyze the image to assess whether the bread is fresh or stale.
             If fresh, estimate its remaining shelf life based on its visual characteristics (e.g., color, texture). If the manufacturing date (MFD) is available
             factor that in to provide a more accurate shelf life. Use concise language (1-2 lines).No need exact expected will work give in json format 
             json format it is{'freshness' : Yes , 'shell life': 1-2days} this is example dont give this exact If any of this is not avilable write NA '''
    return analyze_image_with_prompt(image, prompt)

def count_cart_items(image):
    prompt = '''Here is an image not specifically shopping cart in this image u need to give the list all the products present here.
      
      (e.g., fruits, vegetables, packaged goods , essentials etc.) and categorize them accordingly. Keep the response concise.
      Just give the list of items  u donot know just give empty array but dont give extra info also write the name of the fruit , vegetable etc is their I need in json as key value pairs Fruits , Vegetables , Packed Goods , Beverages , Bakery Essentials as the key and tje items as value in json format
      '''
    
    return analyze_image_with_prompt(image, prompt)

def extract_product_info(image):
    prompt = '''Here is an image of a product. 
    Please extract the text from the image and identify important details. Specifically, look for the following information:
    
    - Product Name
    - Brand Name
    - Type of Product
    - Batch Number
    - Year of Manufacturing
    - Expiry Date
    - Other relevant details (if available)
    - Add Flavour name also if available 
    - If expiry Date is available compare that to the current date ({current_date}) and then calculate how long the product can be consumed.
    
    Format the expiry analysis as:
    - Yes product can be utilized/consumed by the customer for ___ days. OR
    - No, the product cannot be utilized
  
    If any of this information is unreadable or not present, please return "Not readable."
    json format it is
    # format of result 'Product Name' : XXXX , 'Brand Name': XXX ,'Type of Product':XXX ,'Year of Manufacturing': XXX ,'Expiry Date': XX ,'Other relevant Details': XX ,'Utilzation time' : '_days'
    It is just an example Ensure the extracted information is well-formatted, with proper labels and organized for readability. It is just an example Only result should be in this format no extra thing should be their
    If any of this is not avilable write NA'''
    
    # Format the prompt with the current date
    formatted_prompt = prompt.format(current_date=current_date)
    return analyze_image_with_prompt(image, formatted_prompt)