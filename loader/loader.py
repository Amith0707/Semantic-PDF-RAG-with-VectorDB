# loading the text based pdfs---------------------------------------
def text_loader():
    try:
        logger.info("Entered text_loader function \n")
        import os
        from dotenv import load_dotenv
        from langchain_community.document_loaders import PyMuPDFLoader
        # 1.Importing the necessary libaraies
        #fetching the pdf path
        load_dotenv()
        TEXT_PDF_PATH=os.getenv("TEXT_PDF_PATH")
        print(TEXT_PDF_PATH)
        #loading the documents
        text_docs=PyMuPDFLoader(TEXT_PDF_PATH)
        text_loader=text_docs.load()
        #Testing if the document was loaded or not
        print("Sucessfully loaded text based pdf")
        print("-"*60)
        logger.info(f"Loaded {len(text_loader)} pages from text-based PDF")
        print(f"Loaded {len(text_loader)} pages from text-based PDF")
        print("-"*60)

        return text_loader
    
    except Exception as e:
        logger.error(f"Failed to Load Text based pdf: {e}",exc_info=True)
        # exc_info=True is used in logging to include the full traceback (error stack) in your log message.



#loading the table based pdfs------------------------------------------------------------------------------------
def table_loader():
    try:
        logger.info("Entered the table_loader() function")
        #importing the necessary modules
        import os
        from dotenv import load_dotenv
        from langchain_community.document_loaders import PDFPlumberLoader
        #reading the table data
        load_dotenv()
        TABLE_PDF_PATH=os.getenv("TABLE_PDF_PATH")
        table_docs=PDFPlumberLoader(TABLE_PDF_PATH)
        table_loader=table_docs.load()
        print("Sucessfully loaded table based pdf")
        print("-"*60)
        logger.info(f"Loaded {len(table_loader)} pages from table-based PDF")
        print(f"Loaded {len(table_loader)} pages from table-based PDF")
        print("-"*60)

        return table_loader

    except Exception as e:
        logger.error(f"Failed to load the Table Based PDF :{e}",exc_info=True)

#loading the image based pdf loader-------------------------------------------------------------------------------
def image_loader():
    try:
        logger.info("Entered image_loader() function")
        #reading the modules
        import os
        from dotenv import load_dotenv
        import fitz  #----> Just to call to read and write pdf
        from PIL import Image #-->py lib to work with img extract image pagewise
        from pytesseract import image_to_string #ofc self explanatory
        from langchain_core.documents import Document 

        #loading the environment variables
        load_dotenv()
        IMAGE_PDF_PATH=os.getenv("IMAGE_PDF_PATH")
        #reading the document
        doc=fitz.open(IMAGE_PDF_PATH)
        images_docs=[] #to store extracted page as langchain doc objects

        #page-->image extract--->text extract-->to string-->
        #loading the pages
        logger.info("Extracting text by page->image->text")
        for page in doc:
            text=page.get_text().strip() #retrieves any selectable text if exists
            if not text: #which happened in modular testing 
                pix=page.get_pixmap()
                img=Image.frombytes("RGB",[pix.width,pix.height],pix.samples)
                text=image_to_string(img)
            if text.strip(): #if any text extracted
                images_docs.append(Document(
                    page_content=text.strip(),
                    metadata={"page":page.number+1,"source":"image"}
                ))  
        logger.info("Successfully extracted text from image PDF")
        print(images_docs[0].page_content[:500])#trying to see first 500 char of page 1
        print("-"*60)

        return images_docs
    except Exception as e:
        logger.error(f"Error occured in image_loader function :{e}")


if __name__=="__main__":

    import os
    import pickle
    #importing logging utility
    from utils.logger import setup_logger
    #creating log object
    logger=setup_logger(__name__)

    text_doc=text_loader()
    table_doc=table_loader()
    images_doc=image_loader()

    #offloading the loaded documents in other location
    def save_documents(docs,filename):
        with open(filename,'wb') as f:
            pickle.dump(docs,f)
            print(f"Saved :{filename}")

    os.makedirs("artifacts/loaded_data", exist_ok=True)
    save_documents(text_doc, "artifacts/loaded_data/text_docs.pkl")
    save_documents(table_doc, "artifacts/loaded_data/table_docs.pkl")
    save_documents(images_doc, "artifacts/loaded_data/image_docs.pkl")

    logger.info("DATA EXPORTED SUCCESSFULLY FROM logger.py")

        # return text_doc,table_doc,images_doc #export point IMPORTANT
        #ALL IS WORKING WELL

        # print(f"First 500 char of text_doc: \n {text_doc[0].page_content[:500]}")
        # print("-"*100)
        # print(f"First 500 char of table_doc: \n {table_doc[0].page_content[:500]}")
        # print("-"*100)
        # print(f"First 500 char of image_doc: \n {images_doc[0].page_content[:500]}")
        # print("-"*100)

    

    
    