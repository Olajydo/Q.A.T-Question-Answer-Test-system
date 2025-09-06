import sqlite3
import cohere
from config import Config

# Initializing Cohere client
co = cohere.Client(Config.COHERE_API_KEY)

def evaluate_user_response(test_question_id: str, user_answer: str):
    """
    Evaluate user answer:
    - Direct match for MCQ answers (A/B/C/D).
    - If user gives free text, use Cohere to check semantic correctness.
    """

    conn = sqlite3.connect(Config.SQLITE_DB_PATH)
    c = conn.cursor()
    c.execute("SELECT correct_answer, test_question FROM answers WHERE id = ?", (test_question_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return {
            "knowledge_understood": False,
            "knowledge_confidence": 0
        }

    correct_answer, test_question = row
    correct_answer = correct_answer.strip().upper()
    user_answer_clean = user_answer.strip().upper()

    # MCQ match
    if user_answer_clean in ["A", "B", "C", "D"]:
        if user_answer_clean == correct_answer:
            return {"knowledge_understood": True, "knowledge_confidence": 95}
        else:
            return {"knowledge_understood": False, "knowledge_confidence": 70}

    #LLM for free-text evaluation 
    eval_prompt = f"""
    Test Question: {test_question}
    Correct Answer: {correct_answer}
    User Answer: {user_answer}

    Evaluate if the user's answer means the same as the correct answer.
    Respond ONLY with JSON in this format:
    {{
        "knowledge_understood": true/false,
        "knowledge_confidence": <0-100>
    }}
    """

    response = co.chat(
        model="command-r-plus",
        message=eval_prompt,
        temperature=0.2
    )

    try:
        eval_json = eval(response.text.strip())
        return eval_json
    except:
        # Fallback if parsing fails
        return {"knowledge_understood": False, "knowledge_confidence": 60}
