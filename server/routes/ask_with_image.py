from fastapi import APIRouter, UploadFile, File, Form, JSONResponse
from modules.image_captioning import generate_image_description
from modules.query_handlers import query_chain  # <-- Corrected import
from modules.llm import get_llm_chain
from logger import logger
import os

# --- Import RAG components from ask_question.py ---
# (You could also move these to a shared module)
from pinecone import Pinecone
from langchain_core.documents import Document
from langchain.schema import BaseRetriever
from pydantic import Field
from typing import List, Optional
from modules.ask_question import VoyageAIEmbeddings # Re-using the class from your other file
# --------------------------------------------------

router = APIRouter()

# --- This is the retriever setup from ask_question.py ---
class SimpleRetriever(BaseRetriever):
    tags: Optional[List[str]] = Field(default_factory=list)
    metadata: Optional[dict] = Field(default_factory=dict)

    def __init__(self, documents: List[Document]):
        super().__init__()
        self._docs = documents

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self._docs
# ------------------------------------------------------

@router.post("/ask_with_image", tags=["Q&A"])
async def handle_ask_with_image(
    question: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Receives image and text, generates an image description,
    combines them, and queries the RAG pipeline.
    """
    try:
        # 1. Generate description from the uploaded image
        logger.info("Receiving image for captioning...")
        image_bytes = await image.read()
        image_description = generate_image_description(image_bytes)
        logger.info(f"Image Description: {image_description}")

        if "Error:" in image_description:
            # Handle cases where image captioning failed
            return JSONResponse(status_code=500, content={"error": image_description})

        # 2. Combine image description with the user's question
        combined_prompt = (
            f"Based on the context from an image, answer the user's question.\n"
            f"---Image Context---\n{image_description}\n\n"
            f"---User's Question---\n{question}"
        )
        logger.info(f"Combined Prompt: {combined_prompt}")

        # 3. Use your existing RAG pipeline to get the final answer
        # (This logic is copied from ask_question.py)
        
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

        if not all([PINECONE_API_KEY, PINECONE_INDEX_NAME]):
            raise ValueError("API keys or index name not found in environment variables.")

        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)

        embed_model = VoyageAIEmbeddings(model_name="voyage-3.5-lite", device="cpu")
        
        # We embed the COMBINED prompt to find relevant documents
        embedded_query = embed_model.embed_query(combined_prompt)
        res = index.query(vector=embedded_query, top_k=3, include_metadata=True)

        docs = [
            Document(
                page_content=match["metadata"].get("text", ""),
                metadata=match["metadata"]
            )
            for match in res["matches"]
        ]

        retriever = SimpleRetriever(docs)
        chain = get_llm_chain(retriever)
        
        # We send the COMBINED prompt to the chain
        answer = query_chain(chain, combined_prompt)

        # Add the image description to the final response for debugging/display
        answer["image_description"] = image_description
        
        logger.info("Successfully answered query with image.")
        return answer

    except Exception as e:
        logger.exception("Error in /ask_with_image endpoint")
        return JSONResponse(status_code=500, content={"error": str(e)})
