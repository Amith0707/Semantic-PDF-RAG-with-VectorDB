from utils.logger import setup_logger
logger=setup_logger(__name__)
import os
from dotenv import load_dotenv
load_dotenv()
import pickle

def retrieve_chunked_data():
    try:
        logger.info("Entered retrieve_chunked_data()")

        text_chunk_path="artifacts/chunked_data/text_data_chunk.pkl"
        table_chunk_path="artifacts/chunked_data/table_data_chunk.pkl"
        images_chunk_path="artifacts/chunked_data/images_data_chunk.pkl"

        
        with open(text_chunk_path,'rb') as file:
            text_data_chunk=pickle.load(file)
        with open(table_chunk_path,'rb') as file:
            table_data_chunk=pickle.load(file)
        with open(images_chunk_path,'rb') as file:
            images_data_chunk=pickle.load(file)

        print("Loaded the chunked data successfully")

        return text_data_chunk,table_data_chunk,images_data_chunk
    except Exception as e:
        logger.error(f"Error occured in retrieve_chunked_data()")

def embedding(text_data_chunk,table_data_chunk,images_data_chunk):
    try:
        logger.info("Entered the embedding")
        # Implementing openai embeddings
        from langchain_openai.embeddings import OpenAIEmbeddings
        embedding=OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        print("Started embedding process")
        text_embedding=embedding.embed_documents([doc.page_content for doc in text_data_chunk])
        images_embedding=embedding.embed_documents([doc.page_content for doc in images_data_chunk])

        # Table embedding switching to batch wise coz openai token limit 300k exceeding
        def safe_embed(chunks, batch_size=50): # start,stop step
            embeddings = []
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                embeddings.extend(embedding.embed_documents([doc.page_content for doc in batch]))
            print("Table embedidng done")
            return embeddings
        table_embedding=safe_embed(table_data_chunk)



        print("Embedding complete")
        print("Loaded the embeded data successfully")
        logger.info("Embedding complete")
        # print("Sample Embedding: \n",text_embedding[:2])

        return text_embedding,table_embedding,images_embedding
    except Exception as e:
        logger.error(f"Error occured in embedding function.{e}",exc_info=True) #-->backup incase not able to debug
        # logger.error("Error occured in embedding function")

if __name__=="__main__":
    print("In the main funciton of embedding.py")
    text_data_chunk,table_data_chunk,images_data_chunk=retrieve_chunked_data()

    print("Main Function entering retrieving embedded vectors")
    text_embedding,table_embedding,images_embedding=embedding(text_data_chunk,table_data_chunk,images_data_chunk)

    print("Saving the embeddings in a pkl file")

    def save_documents(docs,filename):
        with open(filename,'wb') as f:
            pickle.dump(docs,f)
            print(f"Saved :{filename}")

    os.makedirs("artifacts/embedded_data", exist_ok=True)
    save_documents(text_embedding, "artifacts/embedded_data/text_embed.pkl")
    save_documents(table_embedding, "artifacts/embedded_data/table_embed.pkl")
    save_documents(images_embedding, "artifacts/embedded_data/image_embed.pkl")

    logger.info("DATA EXPORTED SUCCESSFULLY FROM embedding.py")
