from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

import os
import sys
import argparse



embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def create_db(src_date_directory, db_directory):
    loader = DirectoryLoader(src_date_directory, glob = '*.txt')
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)

    db = Chroma.from_documents(
        chunks, embedding_function, persist_directory = db_directory
    )
    db.persist()

    return 



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process directories")
    
    
    parser.add_argument('text_directory', type=str, help='The directory path to the txt files')
    parser.add_argument('save_directory', type=str, help='The directory to save the database')
    args = parser.parse_args()

    
    if not os.path.isdir(args.current_directory):
        print(f"The provided current directory '{args.current_directory}' does not exist.")
        sys.exit(1)

    
    if not os.path.exists(args.save_directory):
        print(f"The save directory '{args.save_directory}' does not exist. Creating it now...")
        os.makedirs(args.save_directory)
    
    create_db(args.text_directory, args.save_directory)