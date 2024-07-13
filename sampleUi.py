specific_query = "";

import os
import streamlit as st
import sys
import boto3
import vecs
import json
import base64
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from typing import Optional
from pptx import Presentation
import io
import re
from PIL import Image
from io import BytesIO
import requests
from pydub import AudioSegment
from pydub.playback import play

AudioSegment.converter = "/usr/bin/ffmpeg" 
CHUNK_SIZE = 1024
url = "https://api.elevenlabs.io/v1/text-to-speech/<voice-id>"

headers = {
  "Accept": "audio/mpeg",
  "Content-Type": "application/json",
  "xi-api-key": "sk_d72dc04e03f01f3bbf892fe030c35466e813eb1819bb6e11"
}

data = {
  "text": "adam grant",
  "model_id": "eleven_monolingual_v1",
  "voice_settings": {
    "stability": 0.5,
    "similarity_boost": 0.5
  }
}


DB_CONNECTION = "<postgresql secret>"

os.environ['AWS_ACCESS_KEY_ID'] = '<insert key id here>'
os.environ['AWS_SECRET_ACCESS_KEY'] = "<insert secret access key here>"
os.environ['AWS_SESSION_TOKEN'] = "<insert session token here>"

S3_POWERPOINTBUCKET = 'powerpointbucket'
S3_IMAGEBUCKET = 'powerpointimages'

helpmsg = "default value"


bedrock_client = boto3.client(
    'bedrock-runtime',
    region_name='us-west-2',
    # Credentials from your AWS account
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    aws_session_token=os.environ['AWS_SESSION_TOKEN'],
)


s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    aws_session_token=os.environ['AWS_SESSION_TOKEN'],
)

def upload_to_s3(file, bucket_name, object_name):
    """Upload a file to an S3 bucket"""
    try:
        s3_client.upload_fileobj(file, bucket_name, object_name)
        st.success("File uploaded successfully")
    except Exception as e:
        st.error(f"Error uploading file: {e}")

def list_images_from_s3(bucket_name, prefix="lect01-storage-images-"):
    """List images from an S3 bucket with a given prefix"""
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if 'Contents' in response:
            return [item['Key'] for item in response['Contents']]
        else:
            return []
    except Exception as e:
        st.error(f"Error listing images: {e}")
        return []


def get_image_from_s3(bucket_name, key):
    """Get an image from S3 bucket"""
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        image_data = response['Body'].read()
        return Image.open(BytesIO(image_data))
    except Exception as e:
        st.error(f"Error retrieving image: {e}")
        return None

####################################################################################

def construct_bedrock_image_body(base64_string):
    """Construct the request body.

    https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-mm.html
    """
    return json.dumps(
        {
            "inputImage": base64_string,
            "embeddingConfig": {"outputEmbeddingLength": 1024},
        }
    )

def readFileAsBase64(file_path):
    """Encode image as base64 string."""
    try:
        with open(file_path, "rb") as image_file:
            input_image = base64.b64encode(image_file.read()).decode("utf8")
        return input_image
    except Exception as e:
        print(f"bad file name: {e}")
        sys.exit(0)
        

def construct_claude_image_prompt(base64_string):
    """Construct the request body for the Claude model."""
    global specific_query 
    if specific_query != "":
        specific_query = "also explain what " + specific_query + " is."
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_string
                    }
                },
                {
                    "type": "text",
                    "text": "Human: I will give you an image of a PowerPoint slide. Generate a detailed paragraph to explain the content to students. Assume the students already know the basic topic of the slide. Start explaining directly and ensure the paragraph is fluid and cohesive, avoid any introductory remark (very important), sound friendly and easy to understand. always start the first sentence with the content. for example, if the slide is about software engineering, the start of the sentence will be: software engineering." + specific_query +  "./nAssistant:"
                }
            ]
        }
    ]
    return json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": messages
    })
    
# I will give you an image of a PowerPoint slide. Generate a detailed paragraph to explain the content to students. Assume the students already know the basic topic of the slide. Start explaining directly and ensure the paragraph is fluid and cohesive, avoid any introductory remark (very important), sound friendly and easy to understand. always start the first sentence with the content. for example, if the slide is about software engineering, the start of the sentence will be: software engineering...Keeping all of this in mind, I want you to answer the question: What is a hard disc drive\


def get_description_from_claude_sonnet(body):
    """Invoke the Claude 3 Sonnet Model via API request."""
    response = bedrock_client.invoke_model(
        body=body,
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        accept="application/json",
        contentType="application/json",
    )
    
    response_body = json.loads(response.get("body").read())
    return response_body["content"][0]["text"]


def describe_image_for_claude(file_path):
    """Generate description for the image at file_path using Claude model."""
    base64_string = readFileAsBase64(file_path)
    body = construct_claude_image_prompt(base64_string)
    description = get_description_from_claude_sonnet(body)
    return description


def display_image_with_description(file_path):
    """Display the image with its description."""
    description = describe_image_for_claude(file_path)
    image = mpimg.imread(file_path)
    plt.title(description)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    return(description)


def getdef(file_path: Optional[str] = None):
    if file_path is None:
        file_path = sys.argv[1]
    return display_image_with_description(file_path)



def seed():
    # create vector store client
    vx = vecs.create_client(DB_CONNECTION)

    # get or create a collection of vectors with 1024 dimensions
    images = vx.get_or_create_collection(name="lect01-vectors", dimension=1024)

    # Generate image embeddings with Amazon Titan Model
    img_emb0 = encode_image('./images/four.jpg')
    # img_emb1 = encode_image('./images/lect01-storage-images-1.jpg')
    # img_emb2 = encode_image('./images/lect01-storage-images-2.jpg')
    # img_emb3 = encode_image('./images/lect01-storage-images-3.jpg')
    # img_emb4 = encode_image('./images/lect01-storage-images-4.jpg')
    # img_emb5 = encode_image('./images/lect01-storage-images-5.jpg')
    # img_emb6 = encode_image('./images/lect01-storage-images-6.jpg')
    # img_emb7 = encode_image('./images/lect01-storage-images-7.jpg')
    # img_emb8 = encode_image('./images/lect01-storage-images-8.jpg')
    # img_emb9 = encode_image('./images/lect01-storage-images-9.jpg')
    # img_emb10 = encode_image('./images/lect01-storage-images-10.jpg')
    # img_emb11 = encode_image('./images/lect01-storage-images-11.jpg')
    # img_emb12 = encode_image('./images/lect01-storage-images-12.jpg')
    # img_emb13 = encode_image('./images/lect01-storage-images-13.jpg')
    # img_emb14 = encode_image('./images/lect01-storage-images-14.jpg')
    # img_emb15 = encode_image('./images/lect01-storage-images-15.jpg')
    # img_emb16 = encode_image('./images/lect01-storage-images-16.jpg')
    # img_emb17 = encode_image('./images/lect01-storage-images-17.jpg')
    # img_emb18 = encode_image('./images/lect01-storage-images-18.jpg')
    # img_emb19 = encode_image('./images/lect01-storage-images-19.jpg')
    # img_emb20 = encode_image('./images/lect01-storage-images-20.jpg')
    # img_emb21 = encode_image('./images/lect01-storage-images-21.jpg')
    # img_emb22 = encode_image('./images/lect01-storage-images-22.jpg')
    # img_emb23 = encode_image('./images/lect01-storage-images-23.jpg')
    # img_emb24 = encode_image('./images/lect01-storage-images-24.jpg')
    # img_emb25 = encode_image('./images/lect01-storage-images-25.jpg')
    # img_emb26 = encode_image('./images/lect01-storage-images-26.jpg')
    # img_emb27 = encode_image('./images/lect01-storage-images-27.jpg')
    # img_emb28 = encode_image('./images/lect01-storage-images-28.jpg')
    # img_emb29 = encode_image('./images/lect01-storage-images-29.jpg')
    # img_emb30 = encode_image('./images/lect01-storage-images-30.jpg')
    # img_emb31 = encode_image('./images/lect01-storage-images-31.jpg')
    # img_emb32 = encode_image('./images/lect01-storage-images-32.jpg')
    # img_emb33 = encode_image('./images/lect01-storage-images-33.jpg')
    # img_emb34 = encode_image('./images/lect01-storage-images-34.jpg')

    # add records to the *images* collection
    images.upsert(
        records = [
            ("four.jpg", img_emb0, {"type": "jpg"}),
            # ("lect01-storage-images-1.jpg", img_emb1, {"type": "jpg"}),
            # ("lect01-storage-images-2.jpg", img_emb2, {"type": "jpg"}),
            # ("lect01-storage-images-3.jpg", img_emb3, {"type": "jpg"}),
            # ("lect01-storage-images-4.jpg", img_emb4, {"type": "jpg"}),
            # ("lect01-storage-images-5.jpg", img_emb5, {"type": "jpg"}),
            # ("lect01-storage-images-6.jpg", img_emb6, {"type": "jpg"}),
            # ("lect01-storage-images-7.jpg", img_emb7, {"type": "jpg"}),
            # ("lect01-storage-images-8.jpg", img_emb8, {"type": "jpg"}),
            # ("lect01-storage-images-9.jpg", img_emb9, {"type": "jpg"}),
            # ("lect01-storage-images-10.jpg", img_emb10, {"type": "jpg"}),
            # ("lect01-storage-images-11.jpg", img_emb11, {"type": "jpg"}),
            # ("lect01-storage-images-12.jpg", img_emb12, {"type": "jpg"}),
            # ("lect01-storage-images-13.jpg", img_emb13, {"type": "jpg"}),
            # ("lect01-storage-images-14.jpg", img_emb14, {"type": "jpg"}),
            # ("lect01-storage-images-15.jpg", img_emb15, {"type": "jpg"}),
            # ("lect01-storage-images-16.jpg", img_emb16, {"type": "jpg"}),
            # ("lect01-storage-images-17.jpg", img_emb17, {"type": "jpg"}),
            # ("lect01-storage-images-18.jpg", img_emb18, {"type": "jpg"}),
            # ("lect01-storage-images-19.jpg", img_emb19, {"type": "jpg"}),
            # ("lect01-storage-images-20.jpg", img_emb20, {"type": "jpg"}),
            # ("lect01-storage-images-21.jpg", img_emb21, {"type": "jpg"}),
            # ("lect01-storage-images-22.jpg", img_emb22, {"type": "jpg"}),
            # ("lect01-storage-images-23.jpg", img_emb23, {"type": "jpg"}),
            # ("lect01-storage-images-24.jpg", img_emb24, {"type": "jpg"}),
            # ("lect01-storage-images-25.jpg", img_emb25, {"type": "jpg"}),
            # ("lect01-storage-images-26.jpg", img_emb26, {"type": "jpg"}),
            # ("lect01-storage-images-27.jpg", img_emb27, {"type": "jpg"}),
            # ("lect01-storage-images-28.jpg", img_emb28, {"type": "jpg"}),
            # ("lect01-storage-images-29.jpg", img_emb29, {"type": "jpg"}),
            # ("lect01-storage-images-30.jpg", img_emb30, {"type": "jpg"}),
            # ("lect01-storage-images-31.jpg", img_emb31, {"type": "jpg"}),
            # ("lect01-storage-images-32.jpg", img_emb32, {"type": "jpg"}),
            # ("lect01-storage-images-33.jpg", img_emb33, {"type": "jpg"}),
            # ("lect01-storage-images-34.jpg", img_emb34, {"type": "jpg"}),
]
    )
    print("Inserted images")

    # index the collection for fast search performance
    images.create_index()
    print("Created index")

def get_embedding_from_titan_multimodal(body):
    """Invoke the Amazon Titan Model via API request."""
    response = bedrock_client.invoke_model(
        body=body,
        modelId="amazon.titan-embed-image-v1",
        accept="application/json",
        contentType="application/json",
    )

    response_body = json.loads(response.get("body").read())
    return response_body["embedding"]

def search(query_term: Optional[str] = None):
    global specific_query 
    specific_query = ""
    if query_term is None:
        query_term = sys.argv[1]
    
    specific_query = query_term
    # create vector store client
    vx = vecs.create_client(DB_CONNECTION)
    images = vx.get_or_create_collection(name="test-vectors", dimension=1024)

    # Encode text query
    text_emb = get_embedding_from_titan_multimodal(json.dumps(
        {
            "inputText": query_term,
            "embeddingConfig": {"outputEmbeddingLength": 1024},
        }
    ))

    # query the collection filtering metadata for "type" = "jpg"
    results = images.query(
        data=text_emb,                      # required
        limit=1,                            # number of records to return
        filters={"type": {"$eq": "jpg"}},   # metadata filters
    )
    result = results[0]
    #plt.title(result)
    image = mpimg.imread('./images/' + result)
    helpmsg = './images/' + result
    help_msg = getdef('./images/' + result)
    st.session_state["help_msg"] = help_msg
    match = re.search(r'(\d+)(?=\.jpg)', result)

    if match:
        number = int(match.group(1))
        #st.write(number)
        st.session_state["current_image_index"] = number

    image = get_image_from_s3(S3_IMAGEBUCKET, result)
    if image:
        st.image(image, caption=result, use_column_width=True)
    st.session_state.help_msg = help_msg




# Streamlit UI
st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
        font-size: 2em;
        font-weight: bold;
    }
    </style>
    <h1 class="centered-title">Welcome to Aspire AI!</h1>
    """,
    unsafe_allow_html=True
)



if 'help_msg' not in st.session_state:
    st.session_state.help_msg = "generated transcript"


# Display the text area with the help_msg
st.text_area(label="transcript", value=st.session_state.help_msg, height=20, disabled=True)

image_keys = list_images_from_s3(S3_IMAGEBUCKET)

if image_keys:
    current_image_index = st.session_state.get("current_image_index", 0)
    
    # Display current image
    image = get_image_from_s3(S3_IMAGEBUCKET, image_keys[current_image_index])
    if image:
        st.image(image, caption=image_keys[current_image_index], use_column_width=True)
    
    
col1, col2, col3 = st.columns([4, 1, 4])

# Place the "Previous" button in the middle of the left half
with col1:
    st.session_state["current_image_index"] = max(0, current_image_index - 1)
    st.markdown('<div class="center-button">', unsafe_allow_html=True)
    if st.button("Previous"):
        st.session_state["current_image_index"] = max(0, current_image_index - 1)
    st.markdown('</div>', unsafe_allow_html=True)

# Place the "Next" button in the middle of the right half
with col3:
    st.markdown('<div class="center-button">', unsafe_allow_html=True)
    if st.button("Next"):
        st.session_state["current_image_index"] = min(len(image_keys) - 1, current_image_index + 1)
        st.session_state["current_image_index"] = min(len(image_keys) - 1, current_image_index + 1)
    st.markdown('</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 1, 2])  # Adjusted column widths for centering
# Add CSS to center the buttons within the columns
st.markdown(
    """
    <style>
    .center-button {
        display: flex;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    .stButton button {
        margin-top: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
col1, col2 = st.columns([4, 1])

# Add text input to the first column
with col1:
    user_input = st.text_input("Type your question here :)")

# Add submit button to the second column
with col2:
    if st.button("Submit"):
        search(user_input)
        st.experimental_rerun()
    
    
text_display = st.empty()

col1, col2, col3 = st.columns([2, 1, 2])  # Adjusted column widths for centering
# Add CSS to center the buttons within the columns
st.markdown(
    """
    <style>
    .center-button {
        display: flex;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Add CSS to center the buttons within the columns
st.markdown(
    """
    <style>
    .center-button {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
    }
    .stButton button {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Upload PowerPoint file
uploaded_file = st.file_uploader("Upload PowerPoint file", type=["pptx"])
if uploaded_file is not None:
    file_name = uploaded_file.name
    upload_to_s3(uploaded_file, S3_POWERPOINTBUCKET, file_name)


def saveaudio():
    response = requests.post(url, json=data, headers=headers)
    with open('output.mp3', 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
    #playsound('output.mp3')
    # audio = AudioSegment.from_mp3('output.mp3')
    # play(audio)



def encode_image(file_path):
    """Generate embedding for the image at file_path."""
    base64_string = readFileAsBase64(file_path)
    body = construct_bedrock_image_body(base64_string)
    emb = get_embedding_from_titan_multimodal(body)
    return emb


def main():
    specific_query = "";
    saveaudio()
    if len(sys.argv) < 3:
        print("Usage: python main.py <function_name> <argument>")
        sys.exit(1)

    function_name = sys.argv[1]
    argument = sys.argv[2]

    if function_name == "getdef":
        getdef(argument)
    elif function_name == "search":
        search(argument)
    else:
        print(f"Function {function_name} not found")


if __name__ == "__main__":
    main()







