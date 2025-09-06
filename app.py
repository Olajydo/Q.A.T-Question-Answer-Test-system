from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from config import Config
from preprocessor import process_and_store_document
from qa_test import get_qa_response
from evaluator import evaluate_user_response

app = Flask(__name__)
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# Global variable to store the current vector db path
CURRENT_VECTOR_DB_PATH = None

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Q.A.T System is running"})

@app.route("/upload/", methods=["POST"])
def upload_file():
    global CURRENT_VECTOR_DB_PATH

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Process and get chunks + vector db path
    chunks, new_vector_db_path = process_and_store_document(filepath)
    CURRENT_VECTOR_DB_PATH = new_vector_db_path  

    return jsonify({
        "message": "PDF processed successfully",
        "chunks_created": chunks,
        "vector_db_path": CURRENT_VECTOR_DB_PATH
    })

@app.route("/query/", methods=["POST"])
def handle_query():
    global CURRENT_VECTOR_DB_PATH
    
    if not CURRENT_VECTOR_DB_PATH:
        return jsonify({"error": "Please upload a PDF first."}), 400
    
    data = request.get_json() or request.form
    user_question = data.get("question")
    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    response = get_qa_response(user_question, CURRENT_VECTOR_DB_PATH)
    return jsonify(response)

@app.route("/evaluate/", methods=["POST"])
def handle_evaluation():
    data = request.get_json() or request.form
    test_question_id = data.get("test_question_id")
    user_answer = data.get("user_answer")

    if not test_question_id or not user_answer:
        return jsonify({"error": "Missing test_question_id or user_answer"}), 400

    result = evaluate_user_response(test_question_id, user_answer)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
