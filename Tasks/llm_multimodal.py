import streamlit as st
import openai
import base64
import os
import time
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

# Set the OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY", "")

st.set_page_config(page_title="Multimodal Q&A", layout="centered")

st.title("Multimodal Q&A with LLM")
st.write("Upload an image and ask a question about it.")

# Sidebar for controls
st.sidebar.header("Model Settings")

# Parameters
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
top_p = st.sidebar.slider("Top-p", 0.0, 1.0, 1.0, 0.1)

# System prompt
SYSTEM_PROMPT = "You are a helpful assistant that answers questions about an image."
with st.expander("System Prompt"):
    st.code(SYSTEM_PROMPT)

# Session state for conversation history (display only)
if "history" not in st.session_state:
    st.session_state["history"] = []

# File uploader
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Question input
question = st.text_input("Ask a question about the image")

# Ask button
if st.button("Ask"):
    if uploaded_image is None:
        st.error("Please upload an image first.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        try:
            # Convert image to base64
            img_bytes = uploaded_image.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")

            # Prepare API call
            start_time = time.time()
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                temperature=temperature,
                top_p=top_p,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", 
                         "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                    ]}
                ]
            )
            end_time = time.time()

            # Extract answer
            result = response.choices[0].message.content

            # Save history (display only)
            st.session_state["history"].append({
                "question": question, 
                "answer": result
            })

            # Show response
            st.success(result)
            st.caption(f"Response time: {end_time - start_time:.2f}s | "
                       f"Temperature={temperature} | Top-p={top_p}")

        except Exception as e:
            st.error(f"Error: {str(e)}")

# Show conversation history (not sent back to model)
if st.session_state["history"]:
    st.subheader("Conversation History")
    for i, qa in enumerate(st.session_state["history"], 1):
        st.write(f"**Q{i}:** {qa['question']}")
        st.write(f"**A{i}:** {qa['answer']}")
