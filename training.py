import json
import nltk
nltk.download('wordnet')
import random
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Initialize lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Load intents data
intents = json.loads(open('intents.json').read())

# Initialize word lists and document structures
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Tokenize and process the patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        word_list = [stemmer.stem(lemmatizer.lemmatize(word.lower())) for word in word_list if word not in ignore_letters]
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Sort and deduplicate the words and classes
words = sorted(set([lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]))
classes = sorted(set(classes))

# Save the words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [stemmer.stem(lemmatizer.lemmatize(word.lower())) for word in word_patterns]
    bag = [1 if word in word_patterns else 0 for word in words]

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag + output_row)

# Shuffle and convert to numpy array
random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(trainY[0]), activation='softmax')
])

# Compile the model
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save('chatbot_model.h5', hist)
print("Training Complete")


# Load the trained model for evaluation
model = tf.keras.models.load_model('chatbot_model.h5')

# Make predictions on the training set (change to your test set if needed)
y_pred_prob = model.predict(trainX)  # Predicted probabilities
y_pred = np.argmax(y_pred_prob, axis=1)  # Predicted class indices

# Convert one-hot encoded trainY to class indices
y_true = np.argmax(trainY, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)

# Calculate precision, recall, and F1 score with weighted averaging
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Display the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)