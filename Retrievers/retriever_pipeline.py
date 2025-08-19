import time
import os
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from utils.logger import setup_logger
# Creating a ranking
from rank_bm25 import BM25Okapi
logger = setup_logger(__name__)

def load_indices():
    """Load all three FAISS indices."""
    try:
        logger.info("Loading FAISS indices...")
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        VECTOR_DATA_PATH = os.path.join(BASE_DIR, "../artifacts/vector_data")
        
        # Load the three indices
        flat_index = FAISS.load_local(
            os.path.join(VECTOR_DATA_PATH, "flat_index"), 
            embedding_model,
            allow_dangerous_deserialization=True #-->allows de-serialization of data
        )
        
        hnsw_index = FAISS.load_local(
            os.path.join(VECTOR_DATA_PATH, "hnsw_index"), 
            embedding_model,
            allow_dangerous_deserialization=True
        )
        
        ivf_index = FAISS.load_local(
            os.path.join(VECTOR_DATA_PATH, "ivf_index"), 
            embedding_model,
            allow_dangerous_deserialization=True
        )
        
        print("Successfully loaded all indices!")
        logger.info("Successfully loaded all indices!")
        return flat_index, hnsw_index, ivf_index
        
    except Exception as e:
        logger.error(f"Error loading indices: {e}")
        return None, None, None # Added for safety

# BM25 Reranker
def bm25_rerank(query, retrieved_docs):
    corpus = [doc.page_content for doc in retrieved_docs]
    tokenized_corpus = [c.split(" ") for c in corpus]
    
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")
    scores = bm25.get_scores(tokenized_query)
    
    reranked = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked]


# MMR Reranker
def mmr_rerank(query, vectorstore, k=5, fetch_k=20):
    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": 0.5}
    )
    return retriever.invoke(query)


def retrieve_and_compare(query, flat_index, hnsw_index, ivf_index, k=5):
    """Simple retrieval comparison between all three indices, 
    plus BM25 and MMR reranking."""
    
    results = {}
    indices = {
        "flat": flat_index,
        "hnsw": hnsw_index, 
        "ivf": ivf_index
    }
    
    print(f"\nQuery: {query}")
    print("-" * 60)
    
    for name, index in indices.items():
        if index is None:
            continue
            
        try:
            # Measure retrieval time
            start_time = time.time()
            docs_with_scores = index.similarity_search_with_score(query, k=k)
            end_time = time.time()
            
            retrieval_time = (end_time - start_time) * 1000  # ms
            
            # Extract documents and scores
            documents = [doc for doc, score in docs_with_scores]
            scores = [score for doc, score in docs_with_scores]
            avg_score = sum(scores) / len(scores) if scores else 0
            
            # Store base results
            results[name] = {
                "time_ms": retrieval_time,
                "avg_score": avg_score,
                "documents": documents,
                "scores": scores,
            }
            
            # ---- BM25 Rerank ----
            bm25_docs = bm25_rerank(query, documents)
            results[name]["bm25_reranked"] = bm25_docs
            
            # ---- MMR Rerank ----
            mmr_docs = mmr_rerank(query, index, k=k, fetch_k=2*k)
            results[name]["mmr_reranked"] = mmr_docs
            
            # Print results
            print(f"{name.upper()} INDEX:")
            print(f"  Time: {retrieval_time:.2f} ms")
            print(f"  Average Score: {avg_score:.4f}")
            print(f"  Documents Retrieved: {len(documents)}")
            print(f"  BM25 Top Doc: {bm25_docs[0].page_content[:80]}...")
            print(f"  MMR  Top Doc: {mmr_docs[0].page_content[:80]}...")
            print("="*30)
            
        except Exception as e:
            print(f"Error with {name} index: {e}")
    
    return results

