import re
import string
import math
import pickle
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Load & clean data
# ---------------------------------------------------------------------------

df = pd.read_csv('spam.csv', encoding='latin-1').drop(columns=['v3', 'v4', 'v5'])
X = df['v2']
Y = df['v1']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2, stratify=Y
)


# ---------------------------------------------------------------------------
# Tokenize
# ---------------------------------------------------------------------------

def tokenize(text):
    text = text.lower()
    text = re.sub(r"\d+", " NUM ", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    return text.split()


X_train_tokens = X_train.apply(tokenize)
X_test_tokens = X_test.apply(tokenize)


# ---------------------------------------------------------------------------
# Count words per class
# ---------------------------------------------------------------------------

spam_counter = Counter()
ham_counter = Counter()

n_spam = 0
n_ham = 0

for tokens, label in zip(X_train_tokens, Y_train):
    if label == 'spam':
        spam_counter.update(tokens)
        n_spam += 1
    else:
        ham_counter.update(tokens)
        n_ham += 1

# priors: overall probability a message is spam / ham
total_messages = n_spam + n_ham
prior_spam = n_spam / total_messages
prior_ham = n_ham / total_messages

# vocabulary: every unique word seen across both classes
vocab = set(spam_counter.keys()) | set(ham_counter.keys())
vocab_size = len(vocab)

total_spam_words = sum(spam_counter.values())
total_ham_words = sum(ham_counter.values())


# ---------------------------------------------------------------------------
# Convert counts to log-probabilities, with Laplace smoothing
# ---------------------------------------------------------------------------

def word_log_prob(word, class_counter, total_words_in_class):
    count = class_counter.get(word, 0)
    # Laplace smoothing: add 1 to numerator, vocab_size to denominator
    prob = (count + 1) / (total_words_in_class + vocab_size)
    return math.log(prob)


log_prob_spam = {word: word_log_prob(word, spam_counter, total_spam_words) for word in vocab}
log_prob_ham = {word: word_log_prob(word, ham_counter, total_ham_words) for word in vocab}


# ---------------------------------------------------------------------------
# Classify a tokenized message
# ---------------------------------------------------------------------------

def classify_tokens(tokens):
    score_spam = math.log(prior_spam)
    score_ham = math.log(prior_ham)

    for word in tokens:
        if word in vocab:
            score_spam += log_prob_spam[word]
            score_ham += log_prob_ham[word]
        # unseen words (not in vocab at all) are skipped rather than smoothed,
        # since they carry no information either class trained on

    # convert log-scores to a normalized confidence between 0 and 1
    max_score = max(score_spam, score_ham)
    exp_spam = math.exp(score_spam - max_score)
    exp_ham = math.exp(score_ham - max_score)
    total = exp_spam + exp_ham

    prob_spam = exp_spam / total
    prob_ham = exp_ham / total

    label = 'spam' if score_spam > score_ham else 'ham'
    confidence = prob_spam if label == 'spam' else prob_ham

    return label, confidence


def classify_message(text):
    tokens = tokenize(text)
    return classify_tokens(tokens)


# ---------------------------------------------------------------------------
# Validate on the held-out test set
# ---------------------------------------------------------------------------

true_positive = false_positive = true_negative = false_negative = 0

for tokens, true_label in zip(X_test_tokens, Y_test):
    predicted_label, _ = classify_tokens(tokens)

    if predicted_label == 'spam' and true_label == 'spam':
        true_positive += 1
    elif predicted_label == 'spam' and true_label == 'ham':
        false_positive += 1
    elif predicted_label == 'ham' and true_label == 'ham':
        true_negative += 1
    else:
        false_negative += 1

accuracy = (true_positive + true_negative) / len(Y_test)
precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("=" * 50)
print("VALIDATION RESULTS (on held-out test set)")
print("=" * 50)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"\nConfusion matrix:")
print(f"  True Positive (correctly caught spam):  {true_positive}")
print(f"  False Positive (ham wrongly flagged):    {false_positive}")
print(f"  True Negative (correctly passed ham):    {true_negative}")
print(f"  False Negative (spam missed):            {false_negative}")


# ---------------------------------------------------------------------------
# Save the trained model
# ---------------------------------------------------------------------------

model_data = {
    'log_prob_spam': log_prob_spam,
    'log_prob_ham': log_prob_ham,
    'prior_spam': prior_spam,
    'prior_ham': prior_ham,
    'vocab': vocab,
}

with open('spam_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\nSaved trained model to spam_model.pkl")
