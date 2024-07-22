from flask import Flask, render_template, request
import os
import google.generativeai as genai
from dotenv import load_dotenv


app = Flask(__name__)
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

math_keywords = [
    'math', 'formula', 'solve', 'calculate', 'equation', 'integral', 'derivative', 
    'function', 'algebra', 'geometry', 'trigonometry', 'statistics', 'probability', 
    'matrix', 'calculus', 'theorem', 'proof', 'numeric', 'variable', 'sequence', 
    'series', 'limit', 'vector', 'set', 'linear', 'quadratic', 'exponential', 
    'logarithm', 'angle', 'ratio', 'proportion', 'fraction', 'theorem', 'proof',
]

important_text_keywords = [
    'description', 'essay', 'agreements', 'application', 'emails', 
    'report', 'proposal', 'contract', 'terms', 'conditions', 'summary',
    'memo', 'letter', 'statement', 'thesis', 'research', 'analysis',
    'presentation', 'discussion', 'review', 'instruction', 'manual',
    'policy', 'procedure', 'recommendation', 'feedback', 'evaluation',
    'correspondence', 'communication', 'announcement', 'notice',
    'agenda', 'minutes', 'memorandum', 'official', 'document',
]

keywords = math_keywords + important_text_keywords
keywords_str = ', '.join(keywords)

system_instruction = f"Provide accurate, relevant, and high-quality information. Include key mathematical and important text concepts such as: {keywords_str}. Ensure the generated text is in plain format and easily copyable and also write text in organized way like paragraph headings line by line programming code."

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    safety_settings=safety_settings,
    generation_config=generation_config,
    system_instruction=system_instruction,
)

history = []

@app.route("/")
def index():
    return render_template("index.html", history=history)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form["user_input"]
    history.append({"role": "user", "parts": [user_input]})

    try:
        chat_session = model.start_chat(history=history)
        response = chat_session.send_message(user_input)
        model_response = response.text
        history.append({"role": "model", "parts": [model_response]})
    except Exception as e:
        history.append({"role": "model", "parts": ["Sorry, there was an error processing your request."]})
        print(f"Error: {e}")

    return render_template("index.html", history=history)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)