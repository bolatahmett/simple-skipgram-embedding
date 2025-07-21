import numpy as np

# Vocabulary & mapping
vocab = ["apple", "pear", "banana", "cherry", "strawberry", "fruit", "red", "sweet",
         "eat", "love", "go", "run", "beautiful", "weather", "sun",
         "cloud", "cold", "hot", "happy", "sad", "and", "are", "to", "i", "fast", "far", "make", "the", "is"]

word_to_index = {w: i for i, w in enumerate(vocab)}
index_to_word = {i: w for w, i in word_to_index.items()}

sentences = [
    "apple sweet and red",
    "beautiful weather and sun",
    "run love",
    "fruit eat happy makes",
    "cloud cold weather makes",
    "apple and banana are fruit",
    "i love to eat sweet strawberry",
    "the weather is hot and sunny",
    "run fast and go far",
    "cloud and cold make weather sad"
]

embedding_dim = 16
vocab_size = len(vocab)
learning_rate = 0.05
window_size = 2  # expanded context window

# Initialize embeddings and output weights randomly
embeddings = np.random.rand(vocab_size, embedding_dim)
output_weights = np.random.rand(embedding_dim, vocab_size)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def one_hot(idx, size):
    vec = np.zeros(size)
    vec[idx] = 1
    return vec

# Prepare training data (target-context pairs)
training_pairs = []
for sentence in sentences:
    words = sentence.lower().split()
    words = [w for w in words if w in word_to_index]  # filter unknown words
    for i, target_word in enumerate(words):
        target_idx = word_to_index[target_word]
        start = max(0, i - window_size)
        end = min(len(words), i + window_size + 1)
        for j in range(start, end):
            if i != j:
                context_idx = word_to_index[words[j]]
                training_pairs.append((target_idx, context_idx))

print(f"Total training pairs: {len(training_pairs)}")

# Training loop
for epoch in range(5000):
    loss = 0
    for target_idx, context_idx in training_pairs:
        # Forward pass
        v_target = embeddings[target_idx]
        scores = np.dot(v_target, output_weights)
        y_pred = softmax(scores)

        y_true = one_hot(context_idx, vocab_size)

        # Cross-entropy loss
        loss -= np.log(y_pred[context_idx] + 1e-7)

        # Backpropagation
        error = y_pred - y_true
        dW = np.outer(v_target, error)
        dv = np.dot(output_weights, error)

        # Update weights
        output_weights -= learning_rate * dW
        embeddings[target_idx] -= learning_rate * dv

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Function to predict likely context words given a word
def predict_context_words(word, top_k=5):
    if word not in word_to_index:
        return []
    idx = word_to_index[word]
    v = embeddings[idx]
    scores = np.dot(v, output_weights)
    probs = softmax(scores)
    top_indices = probs.argsort()[-top_k:][::-1]
    return [(index_to_word[i], probs[i]) for i in top_indices]

# Example predictions
word = "apple"
predictions = predict_context_words(word)
print(f"\nWords likely to appear near '{word}':")
for w, p in predictions:
    print(f"{w}: {p:.4f}")
