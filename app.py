import re
import string
import math
import pickle

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allows your frontend (running on a different port) to call this API


# ---------------------------------------------------------------------------
# Load the trained model once, at startup
# ---------------------------------------------------------------------------

with open('spam_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

log_prob_spam = model_data['log_prob_spam']
log_prob_ham = model_data['log_prob_ham']
prior_spam = model_data['prior_spam']
prior_ham = model_data['prior_ham']
vocab = model_data['vocab']


# ---------------------------------------------------------------------------
# Same tokenizer used during training -- must match exactly
# ---------------------------------------------------------------------------

def tokenize(text):
    text = text.lower()
    text = re.sub(r"\d+", " NUM ", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    return text.split()


def classify_message(text):
    tokens = tokenize(text)

    score_spam = math.log(prior_spam)
    score_ham = math.log(prior_ham)

    trigger_words = []

    for word in tokens:
        if word in vocab:
            score_spam += log_prob_spam[word]
            score_ham += log_prob_ham[word]
            # a word is a "spam trigger" if it's meaningfully more likely under spam
            if log_prob_spam[word] - log_prob_ham[word] > 0:
                trigger_words.append(word)

    max_score = max(score_spam, score_ham)
    exp_spam = math.exp(score_spam - max_score)
    exp_ham = math.exp(score_ham - max_score)
    total = exp_spam + exp_ham

    prob_spam = exp_spam / total
    prob_ham = exp_ham / total

    label = 'spam' if score_spam > score_ham else 'ham'
    confidence = prob_spam if label == 'spam' else prob_ham

    return {
        'label': label,
        'confidence': round(confidence, 4),
        'trigger_words': trigger_words[:5],
    }


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()

    if not data or 'message' not in data:
        return jsonify({'error': 'Request must include a "message" field'}), 400

    message = data['message'].strip()

    if not message:
        return jsonify({'error': 'Message cannot be empty'}), 400

    result = classify_message(message)
    return jsonify(result)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
