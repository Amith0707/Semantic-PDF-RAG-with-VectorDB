# A Python file to store constant variables and functions
import os
import pickle
import time
def save_documents(docs,filename,folder_name):
    os.makedirs(folder_name,exist_ok=True)
    with open(filename,'wb') as f:
        pickle.dump(docs,f)

    print(f"Saved the document:{docs}")

TEMPLATE="""
You are a helpful assistant answering questions based on the given documents.

Context:
{context}

Question:
{question}

Instructions:
- There are three documents from where you will get a context
    1. Journey to the center of the earth novel
    2. A Table data document on World_Development_Indicators.
    3. A pdf named "100 SQL COMMANDS"
- Use only the information provided in the context.
- If the context does not contain the answer, say "I could not find relevant information in the documents."
- Provide clear, well-structured responses.
- If numerical data is involved (tables), present it in a concise format.
- If technical (SQL/Python/etc.), keep code snippets inside ``` blocks.
Final Answer:
"""


