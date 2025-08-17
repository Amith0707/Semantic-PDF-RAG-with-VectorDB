'''
In this part of the code we will be using FAISS indexing then create 3 
type of indexing methods(basically 3index pages ->Flat,IVF,HNSW to check which
is better one)'''
import os
import pickle
from utils.logger import setup_logger
logger=setup_logger(__name__)
from dotenv import load_dotenv
load_dotenv()
import numpy as np
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings

def load_files():
    try:
        logger.info("Entering the load_files to add in vector db")
        print("Entering the load_files to add in vector db")

        # Creating a better path to access chunked files
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        CHUNKED_PATH = os.path.join(BASE_DIR, "../artifacts/chunked_data")

        TEXT_CHUNK_FILE = os.path.join(CHUNKED_PATH, "text_data_chunk.pkl")
        TABLE_CHUNK_FILE = os.path.join(CHUNKED_PATH, "table_data_chunk.pkl")
        IMAGE_CHUNK_FILE = os.path.join(CHUNKED_PATH, "images_data_chunk.pkl")

        with open(TEXT_CHUNK_FILE, "rb") as f:
            text_chunks = pickle.load(f)

        with open(TABLE_CHUNK_FILE, "rb") as f:
            table_chunks = pickle.load(f)

        with open(IMAGE_CHUNK_FILE, "rb") as f:
            image_chunks = pickle.load(f)

        #IMPORTANT- BE COMBINIG ALL THE CHUNKS TOGETHER IN VECTORDB TO AVOID overcomplicating stuff

        all_docs=text_chunks+table_chunks+image_chunks #Since all chunks are list of doc formats
        print("Retrieved all chunked data.")
        logger.info("Retrieved all chunked data.")

        # Initialize the embeddings
        embedding_model=OpenAIEmbeddings(model="text-embedding-3-small")
        print("Exiting the function.....")
        logger.info("Exiting the function.....`")
        return all_docs,embedding_model
    
    except Exception as e:
        print("Error in load_files of indexer.py")
        logger.error("Error in load_files of indexer.py")

def build_faiss_indices(documents,embedding_model):
    try:
        """
        Build Flat, HNSW, and IVF FAISS indices from Document objects.
        Returns a dictionary with keys: 'flat', 'hnsw', 'ivf'.
        """
        logger.info("Entering the build_faiss_indices() function")
        print("Entering the build_faiss_indices() function")
        # Get embedding dimension
        dim = len(embedding_model.embed_query("hello world"))
        ####################################################################
        print("1. Building Flat index....")
        flat_faiss_index = faiss.IndexFlatIP(dim)

        # Build vectorstore
        flat_index = FAISS(
            embedding_function=embedding_model,
            index=flat_faiss_index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        # Add documents
        flat_index.add_documents(documents)
        ######################################################################
        print("2. Building hnsw index....")
        hnsw_faiss_index = faiss.IndexHNSWFlat(dim,32)

        # Build vectorstore
        hnsw_index = FAISS(
            embedding_function=embedding_model,
            index=hnsw_faiss_index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        # Add documents
        hnsw_index.add_documents(documents)
        ######################################################################
        print("3. Building IVF index....")
        nlist = 100  # number of clusters (tuneable)
        quantizer = faiss.IndexFlatL2(dim)
        ivf_faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)

        # Train the IVF index
        doc_embeddings = embedding_model.embed_documents([d.page_content for d in documents])
        doc_embeddings = np.array(doc_embeddings).astype("float32")
        ivf_faiss_index.train(doc_embeddings)

        # Build FAISS vectorstore
        ivf_index = FAISS(
            embedding_function=embedding_model,
            index=ivf_faiss_index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        ivf_index.add_documents(documents)

        print("Creation of all 3 indices successful")
        logger.info("Creation of all 3 indices successful")
        return flat_index,hnsw_index,ivf_index

    except Exception as e:
        print("Error in build_faiss_indices() funcition")
        logger.error("Error in build_faiss_indices() funcition {e}",exc_info=True)


def save_indices(flat_index, hnsw_index, ivf_index):
    try:
        logger.info("Saving FAISS indices to artifacts/vector_data...")
        print("Saving FAISS indices to artifacts/vector_data...")

        # Create directory if it doesn't exist
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        SAVE_DIR = os.path.join(BASE_DIR, "../artifacts/vector_data")
        os.makedirs(SAVE_DIR, exist_ok=True)

        # Save each index
        flat_path = os.path.join(SAVE_DIR, "flat_index")
        hnsw_path = os.path.join(SAVE_DIR, "hnsw_index")
        ivf_path = os.path.join(SAVE_DIR, "ivf_index")

        flat_index.save_local(flat_path)
        hnsw_index.save_local(hnsw_path)
        ivf_index.save_local(ivf_path)

        print("All indices saved successfully!")
        logger.info("All indices saved successfully!")

    except Exception as e:
        print("Error saving FAISS indices")
        logger.error(f"Error saving FAISS indices: {e}", exc_info=True)


if __name__ == "__main__":
    print("Entered the main function in indexer.py")
    logger.info("Entered the main function in indexer.py")

    docs, embeddings_model = load_files()
    flat_idx, hnsw_idx, ivf_idx = build_faiss_indices(docs, embeddings_model)
    save_indices(flat_idx, hnsw_idx, ivf_idx)

