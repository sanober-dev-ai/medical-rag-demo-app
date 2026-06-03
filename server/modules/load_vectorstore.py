import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
PINECONE_ENV="us-east-1"
PINECONE_INDEX_NAME="medicalindex"

os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY

UPLOAD_DIR="./uploaded_docs"
os.makedirs(UPLOAD_DIR,exist_ok=True)


# initialize pinecone instance
pc=Pinecone(api_key=PINECONE_API_KEY)
spec=ServerlessSpec(cloud="aws",region=PINECONE_ENV)
existing_indexes=[i["name"] for i in pc.list_indexes()]

if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=3072,
        metric="dotproduct",
        spec=spec
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)


index=pc.Index(PINECONE_INDEX_NAME)

# load,split,embed and upsert pdf docs content

def load_vectorstore(uploaded_files):
    """Ingest uploaded PDFs into Pinecone.

    uploaded_files: List[fastapi.UploadFile]
    """

    embed_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    file_paths: list[str] = []

    for file in uploaded_files:
        # Persist once to disk, then parse from disk.
        # Important: don't rely on reading UploadFile streams multiple times.
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))

    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Validation logs
        total_chars = sum(len(d.page_content or "") for d in documents)
        print(
            f"📄 Loaded PDF: {file_path} | pages={len(documents)} | total_extracted_chars={total_chars}"
        )

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [
            {
                **chunk.metadata,
                "text": chunk.page_content,
            }
            for chunk in chunks
        ]
        ids = [f"{Path(file_path).stem}-{i}" for i in range(len(chunks))]

        # Additional validation logs
        non_empty_chunks = sum(1 for t in texts if (t and t.strip()))
        print(
            f"✂️ Split into {len(texts)} chunks | non_empty_chunks={non_empty_chunks}"
        )

        # If nothing was extracted, don't upsert empty vectors.
        if not non_empty_chunks:
            print(
                f"⚠️ No text extracted from {file_path}. This PDF may be scanned/image-based or parsing failed."
            )
            continue

        print(f"🔍 Embedding {len(texts)} chunks...")
        embeddings = embed_model.embed_documents(texts)

        print("📤 Uploading to Pinecone...")
        with tqdm(total=len(embeddings), desc="Upserting to Pinecone") as progress:
            index.upsert(vectors=zip(ids, embeddings, metadatas))
            progress.update(len(embeddings))

        print(f"✅ Upload complete for {file_path}")
