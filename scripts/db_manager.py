import os
import pyrebase

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader


class DbManager :
    CONFIG = {
        "apiKey": "AIzaSyClWABkoTyZ8XrIXaaJChZPvqTPCKdfhTA",
        "authDomain": "classproject-53b90.firebaseapp.com",
        "projectId": "classproject-53b90",
        "storageBucket": "classproject-53b90.appspot.com",
        "messagingSenderId": "117138485718",
        "appId": "1:117138485718:web:6d9156971b8bf88b3ef477",
        "measurementId": "G-2YHJSLJ8ED",
        "databaseURL": "https://classproject-53b90-default-rtdb.asia-southeast1.firebasedatabase.app/",
        "serviceAccount": "service_account.json",
    }
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    
    def __init__(self) :
        self.firebase = pyrebase.initialize_app(self.CONFIG)
        self.storage = self.firebase.storage()
        
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        self.embeddings = HuggingFaceEmbeddings(model_name=self.EMBEDDING_MODEL)
        
        self.vector_db_dir = "vector_db"
        if os.listdir(f"./{self.vector_db_dir}/") :
            self.db = self.load_vectorstore()
        else :
            self.db = self.create_vectorstore()
        
        self.save_vectorstore()
        self.old_docs = list()
    
    def create_vectorstore(self) :
        loader = PyPDFLoader("documents/file1.pdf")
        documents = loader.load_and_split()
        all_splits = self.text_splitter.split_documents(documents)
        
        # self.delete_files_in_dir()
        return FAISS.from_documents(all_splits, self.embeddings)
    
    def load_vectorstore(self) :
        return FAISS.load_local(self.vector_db_dir, self.embeddings, allow_dangerous_deserialization=True)
        
        
    def add_documents_to_vectordb(self, directory="./documents/") :
        loader = DirectoryLoader(directory, loader_cls=PyPDFLoader)
        pages = loader.load_and_split()
        docs = self.text_splitter.split_documents(pages)
        
        extension = FAISS.from_documents(docs, self.embeddings)
        self.db.merge_from(extension)
        
        # self.delete_files_in_dir()
        
    
    def save_vectorstore(self) :
        self.db.save_local(self.vector_db_dir)
        
    
    def delete_files_in_dir(self, directory="./documents/") :
        files = os.listdir(directory)
        for file in files :
            if file != "file1.pdf" :
                os.remove(directory + file)
        
        
    def download_docs_from_firebase(self, filename=None) :
        ## To list all the files in storage
        esp = self.storage.list_files()
        docs = list()
        for file in esp :
            if not filename :
                if file.name[-4:] == ".pdf" and file.name not in self.old_docs :
                    docs.append(file.name)
            else :
                file_name = [i for i in file.name.split("/")][-1]
                if file.name[:-4] == ".pdf" and file_name == filename :
                    docs.append(file.name)
        # To download a file from firebase
        for doc in docs :
            splits = [i for i in doc.split("/")]
            file_name = splits[-1]
            firebase_path = "/".join(splits[:-1])
            local_path = f"documents/{file_name}"
            self.storage.child(firebase_path).download(path=doc, filename=local_path)
            
        # Add the processed plates to old_plates list
        self.old_docs.extend(docs)
        
        
        
db_manager = DbManager()