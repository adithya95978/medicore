import streamlit as st
from components.upload import render_uploader
from components.history_download import render_history_download
from components.chatUI import render_chat
from utils.api import get_answer_with_image


st.set_page_config(page_title="AI Medical Assistant",layout="wide")
st.title(" 🩺 Medical Assistant Chatbot")
st.markdown("---")
st.header("🔬 Ask with Image Context")

uploaded_image = st.file_uploader("Upload a medical image", type=["png", "jpg", "jpeg"])
if uploaded_image:
    st.image(uploaded_image, width=250)

prompt_with_image = st.text_input("Ask a question about the image and documents")

if st.button("Get Answer with Image"):
    if uploaded_image and prompt_with_image:
        with st.spinner("Analyzing image and finding answer..."):
            result = get_answer_with_image(prompt_with_image, uploaded_image)
            if result:
                st.success("**Answer:**")
                st.write(result.get("answer"))
                with st.expander("View AI-Generated Image Context"):
                    st.info(result.get("image_description"))
    else:
        st.warning("Please upload an image and enter a question.")


render_uploader()
render_chat()
render_history_download()   
