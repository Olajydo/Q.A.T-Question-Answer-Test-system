import cohere
import sqlite3
import uuid
import re
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from config import Config

# Initialize Cohere client
co = cohere.Client(Config.COHERE_API_KEY)

# Initialize Cohere embeddings
embedding_model = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=Config.COHERE_API_KEY,
    user_agent="langchain"
)


def get_qa_response(user_question, vector_db_path):
    """Retrieve context, generate answer + test + store correct answer"""

    # Initialize vector store with the provided path
    vector_store = Chroma(
        persist_directory=vector_db_path,
        embedding_function=embedding_model
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(user_question)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Generate answer
    answer_prompt = f"""
    Context: {context}
    Question: {user_question}
    Answer directly and clearly:
    """
    answer_response = co.chat(
        model="command-r-plus",
        message=answer_prompt,
        temperature=0.3
    )
    final_answer = answer_response.text

    #Generate test + bullet points
    test_prompt = f"""
    You are a teacher testing whether a student understood the following answer:

    Answer: {final_answer}

    Create ONE multiple-choice test question that directly asks about the content above. 
    The question should be factual (NOT a "which is NOT" or "except" type question).
    The question MUST end with a question mark (?).
    Provide exactly 4 answer options, one of which is correct.

    Format your response EXACTLY like this:

    Test Question: <a complete question ending with ?>
    A) <option>
    B) <option>
    C) <option>
    D) <option>
    Correct Answer: <Letter>  (ONLY the letter A, B, C, or D - no extra text)
    Bullet Points:
    - <key point 1>
    - <key point 2>
    """
    test_response = co.generate(
        model="command",
        prompt=test_prompt,
        max_tokens=400
    )
    generated_text = test_response.generations[0].text.strip()

    
    lines = generated_text.split("\n")
    test_question_text = None
    options = []
    correct_answer, bullet_points = None, []

    for line in lines:
        line = line.strip()
        if line.lower().startswith("test question:"):  
            test_question_text = line.split(":", 1)[1].strip()
        elif re.match(r"^[A-Da-d]\)", line):  
            options.append(line)
        elif line.lower().startswith("correct answer:"): 
            # Extract just the letter (A, B, C, or D)
            answer_text = line.split(":", 1)[1].strip()
            # Clean up any extra text and get just the letter
            match = re.search(r"([A-Da-d])", answer_text)
            if match:
                correct_answer = match.group(1).upper()  
        elif line.startswith("-"):
            bullet_points.append(line[1:].strip())

    #if Cohere skipped the question
    if not test_question_text:
        test_question_text = f"What is the correct option based on: {final_answer}?"

    # Ensure we have exactly 4 options
    while len(options) < 4:
        options.append(f"{chr(65 + len(options))}) <Option not generated>")

    # Validate correct answer format
    if not correct_answer or correct_answer not in ['A', 'B', 'C', 'D']:
        correct_answer = 'A'  # Default to first option

    # Build the full test question with options
    test_question = f"{test_question_text}\n" + "\n".join(options)

    # Store in SQLite
    test_id = str(uuid.uuid4())
    conn = sqlite3.connect(Config.SQLITE_DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS answers (
                    id TEXT PRIMARY KEY,
                    correct_answer TEXT,
                    test_question TEXT
                )''')
    c.execute("INSERT INTO answers (id, correct_answer, test_question) VALUES (?, ?, ?)",
              (test_id, correct_answer, test_question))
    conn.commit()
    conn.close()

    return {
        "answer": final_answer,
        "bullet_points": bullet_points,
        "test_question": test_question,
        "test_question_id": test_id
    }


# verification function
def verify_answer(test_question_id, user_choice, context=None):
    """Verify if the user's answer is correct using LLM instead of just string matching"""
    conn = sqlite3.connect(Config.SQLITE_DB_PATH)
    c = conn.cursor()
    c.execute("SELECT correct_answer, test_question FROM answers WHERE id = ?", (test_question_id,))
    result = c.fetchone()
    conn.close()

    if not result:
        return {"knowledge_understood": False, "knowledge_confidence": 0}

    stored_answer, test_question = result
    user_clean = re.sub(r'[^A-Da-d]', '', user_choice).upper()

    # Use LLM to double-check
    eval_prompt = f"""
    Test Question: {test_question}
    User's Answer: {user_clean}
    Stored Correct Answer: {stored_answer}

    Decide if the user's answer matches the correct one.
    Respond with only this format:
    Understood: True/False
    Confidence: <number between 0 and 100>
    """

    eval_response = co.generate(
        model="command",
        prompt=eval_prompt,
        max_tokens=50
    )
    eval_text = eval_response.generations[0].text.strip()

    
    understood, confidence = False, 60

    # Parse LLM response
    match_understood = re.search(r"Understood:\s*(True|False)", eval_text, re.I)
    match_conf = re.search(r"Confidence:\s*(\d+)", eval_text)

    if match_understood:
        understood = match_understood.group(1).lower() == "true"
    if match_conf:
        confidence = int(match_conf.group(1))

    return {
        "knowledge_understood": understood,
        "knowledge_confidence": confidence
    }