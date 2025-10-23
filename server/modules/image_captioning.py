# modules/image_captioning.py

import torch
from PIL import Image
import io
from transformers import BlipProcessor, BlipForConditionalGeneration
from logger import logger # Use your existing logger

# --- Model Loading (Singleton Pattern) ---
# This is crucial: We load the model and processor only ONCE when
# the FastAPI server starts, not on every API call.
# This saves a massive amount of time and memory.

try:
    logger.info("Loading MEDICAL image captioning model...")
    
    # Check if a GPU is available and use it
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {DEVICE} for image captioning.")

    # --- THIS IS THE ONLY LINE YOU NEED TO CHANGE ---
    MODEL_ID = "Telemed-ai/blip-medical-captioning-large"
    # -------------------------------------------------

    # The processor prepares the image for the model
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    
    # The model generates the text
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)
    
    logger.info(f"Medical captioning model ({MODEL_ID}) loaded successfully.")

except Exception as e:
    logger.exception("CRITICAL: Failed to load image captioning model!")
    processor = None
    model = None
# ----------------------------------------------------


def generate_image_description(image_bytes: bytes) -> str:
    """
    Generates a text description for a given image in bytes.
    """
    # Fail gracefully if the model didn't load on startup
    if model is None or processor is None:
        logger.error("Image captioning model is not loaded. Cannot process image.")
        return "Error: Image processing service is unavailable."

    try:
        # 1. Open the image from the raw bytes
        # .convert('RGB') is a safety measure to ensure compatibility
        raw_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # 2. Process the image for the model
        # We don't provide text, so it's "unconditional" generation
        inputs = processor(raw_image, return_tensors="pt").to(DEVICE)

        # 3. Generate the text (caption)
        logger.debug("Generating medical image caption...")
        # You might want to experiment with max_new_tokens for medical reports
        out = model.generate(**inputs, max_new_tokens=128) 

        # 4. Decode the generated token IDs back into a string
        description = processor.decode(out[0], skip_special_tokens=True)
        logger.debug(f"Generated caption: {description}")

        return description

    except Exception as e:
        logger.exception("Error occurred during image caption generation")
        return "Error: Could not generate a description from the provided image."
