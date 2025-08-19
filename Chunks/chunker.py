# This part of code is mainly for chunking the data 
from utils.logger import setup_logger
logger=setup_logger(__name__)
import os
from dotenv import load_dotenv
load_dotenv()
import pickle

def read_files():
    #reading the text based pickle file:
    try:
        logger.info("Entered the read_files in chunker.py")
        text_file_path="artifacts/loaded_data/text_docs.pkl"
        table_file_path="artifacts/loaded_data/table_docs.pkl"
        images_file_path="artifacts/loaded_data/image_docs.pkl"

        with open(text_file_path,'rb') as file:
            text_data=pickle.load(file)
        with open(table_file_path,'rb') as file:
            table_data=pickle.load(file)
        with open(images_file_path,'rb') as file:
            images_data=pickle.load(file)
        
        print("DATA LOADED ALL THREE SUCCESSFULLY")
        logger.info("DATA LOADED ALL THREE SUCCESSFULLY")
        return text_data,table_data,images_data
    
    except Exception as e:
        logger.error(f"Error caused in the read_files function of chunker.py")
        return None,None,None #-->to avpoid TypeError in main.py when unpacking if nothing comes

def chunk_files(text_data,table_data,images_data):
    try:
        logger.info("Entered the chunk_files function in chunker.py")

        #importing necessary files
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_text_splitters import CharacterTextSplitter #this is for only table data

        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,chunk_overlap=200
        )
        image_splitter=RecursiveCharacterTextSplitter(
            chunk_size=800,chunk_overlap=100,separators=["\n\n","\n"," ",""]
        )
        table_splitter=CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separator="\n"
        )
        print("Sucessfully Invoked splitter objects")
        logger.info("Sucessfully Invoked splitter objects")

        text_chunks=text_splitter.split_documents(text_data)
        table_chunks=table_splitter.split_documents(table_data)
        images_chunks=image_splitter.split_documents(images_data)

        print("Data Chunked successfully")
        logger.info("Data Chunked successfully")

        return text_chunks,table_chunks,images_chunks
    
    except Exception as e:
        logger.error(f"Error caused in chunk_files function of chunker.py")
        