from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse

from modules.llm import get_llm_chain
from modules.query_handlers import query_chain

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone

from pydantic import Field
from typing import List
from logger import logger
import os

router = APIRouter()


class SimpleRetriever(BaseRetriever):
    docs: List[Document] = Field(default_factory=list)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self.docs


@router.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"user query: {question}")

        # Pinecone setup
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

        # Gemini embedding model
        embed_model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )

        # Embed user query
        embedded_query = embed_model.embed_query(question)

        # Search Pinecone
        res = index.query(
            vector=embedded_query,
            top_k=3,
            include_metadata=True
        )

        # Convert Pinecone matches to LangChain Documents
        docs = [
            Document(
                page_content=match["metadata"].get("text", ""),
                metadata=match["metadata"]
            )
            for match in res["matches"]
        ]

        if not docs:
            return {
                "answer": "No relevant documents found."
            }

        # Create retriever
        retriever = SimpleRetriever(docs=docs)

        # Create QA chain
        chain = get_llm_chain(retriever)

        # Ask question
        result = query_chain(chain, question)

        logger.info("query successful")

        return result

    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )