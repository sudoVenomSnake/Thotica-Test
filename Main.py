import streamlit as st
import os
from llama_index import load_index_from_storage, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from tqdm import tqdm

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

def join(source_dir, prefix, dest_file, read_size):
    output_file = open(dest_file, 'wb')
    parts = len(os.listdir(source_dir))
    for file in tqdm(range(1, parts + 1)):
        path = source_dir + prefix + str(file)
        input_file = open(path, 'rb')
        while True:
            bytes = input_file.read(read_size)
            if not bytes:
                break
            output_file.write(bytes)
        os.remove(path)
        input_file.close()
    output_file.close()

st.title('Thotica Search Experimental')

if 'docstore.json' not in os.listdir('prod_index') or 'vector_store.json' not in os.listdir('prod_index'):
    with st.spinner('Preparing Index'):
        join('temp_docstore/', 'docstore_', 'prod_index/docstore.json', 1000000)
        join('temp_vectorstore/', 'vector_store', 'prod_index/vector_store.json', 1000000)

query = st.text_input(label = 'Please enter your query - ', value = 'What causes ocean acidification?')
top_k = st.number_input(label = 'Top k - ', min_value = 2, max_value = 25, value = 5)

if query and top_k:
    index = load_index_from_storage(storage_context = StorageContext.from_defaults(
                docstore = SimpleDocumentStore.from_persist_dir(persist_dir = "prod_index"),
                vector_store = FaissVectorStore.from_persist_dir(persist_dir = "prod_index"),
                index_store = SimpleIndexStore.from_persist_dir(persist_dir = "prod_index"),
            ))
    retriever = index.as_retriever(retriever_mode = 'embedding', similarity_top_k = int(top_k))
    response = {}
    for i in retriever.retrieve(query):
        response[str(i.id_)] = {
                'Text' : i.get_text(),
                'Score' : i.get_score(),
                'Title_URL' : i.node.metadata['Title_URL'],
                'Author' : i.node.metadata['Author'],
                'Publisher' : i.node.metadata['Publisher'],
                'Title_URL' : i.node.metadata['Title_URL'],
                'Type' : i.node.metadata['Type']
            }
    print(response)
    st.json(response)