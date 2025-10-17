from fastapi import APIRouter, UploadFile, File, Form
from modules.image_captioning import generate_image_description
from modules.query_handlers import process_query # Your existing RAG function

router = APIRouter()

@router.post("/ask_with_image", tags=["Q&A"])
async def handle_ask_with_image(
    question: str = Form(...),
    image: UploadFile = File(...)
):
    """Receives image and text, combines them, and queries the RAG pipeline."""
    # 1. Generate description from the uploaded image
    image_bytes = await image.read()
    image_description = generate_image_description(image_bytes)

    # 2. Combine image description with the user's question
    combined_prompt = (
        f"Image Context: {image_description}\n\n"
        f"User's Question: {question}"
    )

    # 3. Use your existing RAG pipeline to get the final answer
    answer = process_query(combined_prompt)

    return {
        "answer": answer,
        "image_description": image_description,
    }
