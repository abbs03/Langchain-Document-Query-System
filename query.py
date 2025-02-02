from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings

import openai 

from dotenv import load_dotenv
import argparse
import os



load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']



embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
prompt = """
Answer the question based only on the following context:

{context}

---

useing the above context answer the following question: {question}
"""



def main():
    parser = argparse.ArgumentParser(description="Process directory and query.")
    

    parser.add_argument('db_directory', type=str, help='The directory to the database')
    parser.add_argument('query_text', type=str, help='A string input for query')

    args = parser.parse_args()

    db = Chroma(persist_directory=args.db_directory, embedding_function=embedding_function)
    results = db.similarity_search_with_relevance_scores(args.query_text, k=5)

    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(prompt)
    prompt = prompt_template.format(context=context_text, question=args.query_text)

    model = ChatOpenAI()
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)




if __name__ == '__main__':
    main()
