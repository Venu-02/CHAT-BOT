#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import json
import pickle
import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents and saved data
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Clean up and tokenize sentences
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Create a bag of words from a given sentence
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for word in sentence_words:
        for i, known_word in enumerate(words):
            if word == known_word:
                bag[i] = 1
    return np.array(bag)

# Predict the class of a given sentence
def predict_class(sentence):
    # Create a bag-of-words from the input sentence
    bow = bag_of_words(sentence)
    
    # Get prediction probabilities
    prediction = model.predict(np.array([bow]))[0]

    # Define an error threshold for filtering low-probability results
    ERROR_THRESHOLD = 0.25
    
    # Extract classes with probabilities above the threshold
    results = [
        {"index": i, "probability": prob}
        for i, prob in enumerate(prediction)
        if prob > ERROR_THRESHOLD
    ]
    
    # Sort results by probability in descending order
    results.sort(key=lambda x: x["probability"], reverse=True)
    
    # Check if there are results that meet the threshold
    if not results:
        return []

    # Prepare the return list with intent and probability
    return_list = [
        {
            "intent": classes[result["index"]],
            "probability": str(result["probability"]),
        }
        for result in results
    ]

    return return_list

# Get a response based on the predicted class
def get_response(intents_list, intents_json):
    # Ensure the intents_list is not empty
    if not intents_list:
        raise ValueError("The intents_list is empty, cannot determine a response.")

    tag = intents_list[0]['intent']
    list_of_intents = intents_json.get("intents", [])

    # Find the intent with the given tag
    for intent in list_of_intents:
        if intent.get("tag") == tag:
            responses = intent.get("responses", [])
            if responses:
                return random.choice(responses)
            else:
                raise ValueError(f"No responses found for tag: {tag}")

    # If no matching tag is found, raise an error
    raise ValueError(f"No intent found with tag: {tag}")

# Bot main loop
print("GO! Bot is running!")

# while True:
#     message = input("You: ")  # Prompt to indicate user input
#     predicted_intents = predict_class(message)
#     if predicted_intents:
#         response = get_response(predicted_intents, intents)
#         print("Bot:", response)  # Bot's response
#     else:
#         print("Bot: I'm not sure I understand. Can you clarify?")


# In[ ]:




