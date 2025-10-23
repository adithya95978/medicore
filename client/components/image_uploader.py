import streamlit as st
from utils.api import get_answer_with_image

def render_image_uploader():
    """
    Renders a dedicated section for uploading an image and asking
    a question about it.
    """
    st.header("🔬 Ask with Image Context")

    # Add 'key' to avoid conflicts with other uploaders
    uploaded_image = st.file_uploader(
        "Upload a medical image", 
        type=["png", "jpg", "jpeg"], 
        key="dedicated_image_uploader"
    )
    
    if uploaded_image:
        st.image(uploaded_image, width=250)

    prompt_with_image = st.text_input(
        "Ask a question about the image and documents", 
        key="dedicated_image_prompt"
    )

    if st.button("Get Answer with Image"):
        if uploaded_image and prompt_with_image:
            with st.spinner("Analyzing image and finding answer..."):
                
                # Call the API
                response = get_answer_with_image(prompt_with_image, uploaded_image)
                
                # --- CRITICAL FIX ---
                # Check the response status code and parse the JSON
                if response.status_code == 200:
                    result_data = response.json()
                    st.success("**Answer:**")
                    st.write(result_data.get("answer"))
                    
                    if "image_description" in result_data:
                        with st.expander("View AI-Generated Image Context"):
                            st.info(result_data.get("image_description"))
                else:
                    st.error(f"Error from API: {response.text}")
                # --- END FIX ---

        else:
            st.warning("Please upload an image AND enter a question.")
