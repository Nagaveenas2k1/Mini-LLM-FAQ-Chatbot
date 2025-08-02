import random
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Input FAQs once at startup from terminal for demo purposes
faqs = []
n = int(input("How many FAQs do you want to add? "))
for _ in range(n):
    q = input("Enter FAQ question: ")
    a = input("Enter FAQ answer: ")
    faqs.append({"question": q, "answer": a})

random.shuffle(faqs)

model = SentenceTransformer('all-MiniLM-L6-v2')

def find_best_answer(user_query, faqs):
    questions = [faq['question'] for faq in faqs]
    question_embeddings = model.encode(questions, convert_to_tensor=True)
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, question_embeddings)[0]
    best_idx = scores.argmax()
    return faqs[best_idx]['answer']

@app.route("/", methods=["GET"])
def home():
    # Pass only the questions to the frontend
    faq_questions = [faq["question"] for faq in faqs]
    return render_template("index.html", faq_questions=faq_questions)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_q = data.get('question')
    if not user_q:
        return jsonify({"error": "No question provided"}), 400
    answer = find_best_answer(user_q, faqs)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)


