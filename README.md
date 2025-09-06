# Q.A.T System (Question, Answer & Test)

This project implements a Q.A.T (Question, Answer & Test) system for research documents using Flask and a Large Language Model (LLM).
Unlike a normal Q\&A system, this application not only answers questions from uploaded documents but also generates test questions to check if the user has understood the response.

We use an LLM because it provides natural language understanding and generation, making it possible to extract meaningful answers, highlight key points, and generate evaluation questions automatically.

---

## Features

* Upload PDF documents (from any location on your machine)
* Ask questions and get structured responses:

  * Answer
  * Bullet points
  * Test question + ID
* Evaluate your response with a True/False result and confidence score
* Stores test questions & answers in a local database

---

## Setup Instructions

1. Clone the repository:

   ```bash
   git clone <your-repo-link>
   cd <your-repo-folder>
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate    
   ```

3. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```

4. Add your Cohere API Key:
   There is created `.env` file in the root directory :

   ```
   COHERE_API_KEY=your_real_api_key_here
   ```

   Get your key here: [https://dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys)

---

## Running the Application

Start the Flask server:

```bash
python app.py
```

---

## Testing the Application

Use the provided testing script:

```bash
python test_qa.py
```

Steps:

1. Enter the full path to your PDF (can be anywhere on your system)and it should not be in any quote.
2. Enter your question, also should not be in any quote.
3. Answer the generated test question.
4. Receive evaluation with:

   * `knowledge_understood` (True/False)
   * `knowledge_confidence` (%)

---

## Project Structure

* `app.py` → Flask app with 3 endpoints (`upload/`, `query/`, `evaluate/`)
* `preprocessor.py` → Process PDF into chunks
* `qa_test.py` → Generate answers and test questions with LLM
* `evaluator.py` → Evaluate user’s answers
* `test_qa.py` → Command line tester
* `test_answers.db` → SQLite database (auto-created)
* `.env` → Your Cohere API key
* `confg` → Loads settings like API key, file paths.

---
