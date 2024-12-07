from dotenv import load_dotenv
import os
import streamlit as st
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.chat_models import ChatOpenAI
from io import BytesIO
import json
import requests

load_dotenv()

# llm = ChatOpenAI(temperature=0.2, model_name="gpt-4")
# prompt = PromptTemplate(
#     input_variables=["question", "elements"],
#     template="""You are a traffic reporter that can answer question related to the traffic situation of an image. You have the ability to see the image and answer questions about it. 
#     I will give you a question and element about the image and you will answer the question.
#         \n\n
#         #Question: {question}
#         #Elements: {elements}
#         \n\n
#         Your structured response:""",
#     )

from openai import OpenAI
import base64

# Replace with your actual OpenAI API key
api_key = "your_actual_api_key_here"
client = OpenAI(api_key=api_key)

# @st.cache_resource(show_spinner="Loading model...")
# def load_model():
#     processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
#     model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
#     return model, processor

def process_query(encoded_image, query):
    # model, processor = load_model()
    # encoding = processor(image, query, return_tensors="pt")
    # outputs = model(**encoding)
    # logits = outputs.logits
    # idx = logits.argmax(-1).item()
    # chain = LLMChain(llm=llm, prompt=prompt)
    # response = chain.run(question=query, elements=model.config.id2label[idx])
    # return response

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Make sure you're using the correct model
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}",  # Embed the encoded image
                        },
                    },
                ],
            }
        ],
        max_tokens=100,
    )

    content = response.choices[0]
    return content

def convert_png_to_jpg(image):
    rgb_image = image.convert('RGB')
    byte_arr = BytesIO()
    rgb_image.save(byte_arr, format='JPEG')
    byte_arr.seek(0)
    return Image.open(byte_arr)

def find_id_by_display_name(display_name):
    file_path = 'camera_source.json'

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    for feature in data['features']:
        if feature['properties']['displayName'] == display_name:
            return feature['properties']['id']
    return None

def convert_base64_to_image(base64_str):    
    # Decode the Base64 string
    image_data = base64.b64decode(base64_str)
    
    # Create a BytesIO object and open the image
    image = Image.open(BytesIO(image_data))
    return image

def app():
    st.title("Báo cáo tình hình giao thông 🚦")
    # Sidebar contents
    with st.sidebar:
        st.title('About')
        st.markdown('''
        This app is built using:
        - [Streamlit](https://streamlit.io/)
        - [OpenAI](https://platform.openai.com/docs/models)
        ''')
        add_vertical_space(5)
        st.write('Made by [Minh Lu Xuan](https://github.com/minhluxuan)')
        st.write('Repository [Github](https://github.com/minhluxuan/traffic_reporter)')

    route_query = st.text_input('Vui lòng nhập tuyến đường trước...')

    if st.button('Kiểm tra'):
        camera_id = find_id_by_display_name(route_query)

        if not camera_id:
            return
        response = requests.get(f'https://api.bktraffic.com/api/cameras/get-camera-image?id={camera_id}')
        if not response:
            return
    
        if response.status_code == 200:
            data = response.json().get('data')
            if not data:
                return
            
            encoded_image = data.split(",")[1]
            image = convert_base64_to_image(encoded_image)

            with st.spinner('Đang xử lý...'):
                answer = process_query(encoded_image, 'Tóm tắt tình hình giao thông')
                st.write(answer.message.content)

            st.image(image, caption="Hình ảnh tình hình giao thông hiện tại", use_column_width=True)

    # uploaded_file = st.file_uploader('Upload your IMAGE', type=['png', 'jpeg', 'jpg'], key="imageUploader")

    # if uploaded_file is not None:
    #     image = Image.open(uploaded_file)
        
    #     # ViLT model only supports JPG images
    #     if image.format == 'PNG':
    #         image = convert_png_to_jpg(image)

    #     st.image(image, caption='Uploaded Image.', width=300)
        
    #     cancel_button = st.button('Cancel')
    #     query = st.text_input('Ask a question to the IMAGE')

    #     if query:
    #         with st.spinner('Processing...'):
    #             answer = process_query(image, query)
    #             st.write(answer)
          
    #     if cancel_button:
    #         st.stop()
            

app()