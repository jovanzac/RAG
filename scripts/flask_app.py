from flask import Flask, jsonify, request
from flask_cors import CORS

from scripts.db_manager import db_manager
from scripts.llm import Llm


app = Flask(__name__)
CORS(app)


def add_to_vector_db_docs(filename=None, batch=False) :
        # Download all the new docs from firebase storage
        db_manager.download_docs_from_firebase(filename)
        # Split into chunks nd add all new documents to the vector db
        db_manager.add_documents_to_vectordb()
        # Clear all documents in the local dir
        db_manager.delete_files_in_dir()

@app.route("/docs_upload", methods=["POST"])
def docs_upload() :
    if request.method == "POST" :
        # response = request.json
        # add_to_vector_db_docs(filename=request["file"], batch=True)
        add_to_vector_db_docs()
        
        return jsonify(
            {
                "Status": "Success"
            }
        )
        
        
@app.route("/user_query", methods=["POST"])
def user_query() :
    if request.method == "POST" :
        print("Received the post request")
        response = request.json
        query = response["query"]
        
        response = llm.route(query)
        
        return jsonify(
            {
                "response": response
            }
        )



llm = Llm()