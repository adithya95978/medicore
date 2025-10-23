import streamlit as st
# Import both API functions
from utils.api import ask_question, get_answer_with_image

def render_chat():
    st.subheader("💬 Chat with your assistant")

    # 1. Add an image uploader
    # This widget will hold the file in session state until cleared
    image_file = st.file_uploader(
        "Upload a medical image (optional)", 
        type=["png", "jpg", "jpeg"]
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render existing chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # Input and response logic
    user_input = st.chat_input("Type your question....")
    
    if user_input:
        # Display user's text message
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # 2. Check if an image was uploaded and decide which API to call
        if image_file:
            # If image exists, call the multimodal endpoint
            with st.spinner("Analyzing image and question..."):
                response = get_answer_with_image(user_input, image_file)
        else:
            # If no image, call the text-only endpoint
            with st.spinner("Thinking..."):
                response = ask_question(user_input)

        # 3. Process the response
        if response.status_code == 200:
            data = response.json()
            
            # Handle different response keys from the two endpoints
            # /ask uses "response", /ask_with_image uses "answer"
            answer = data.get("answer", data.get("response", "Error: No answer key in response."))
            
            st.chat_message("assistant").markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Optional: Display the image description if it's in the response
            if "image_description" in data:
                st.chat_message("assistant").info(f"Image context: {data['image_description']}")

        else:
            st.error(f"Error: {response.text}")
