from flask import Flask, render_template, request, jsonify
import chatbot

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/chat")
def chat():
    return render_template("chat.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    predicted_intents = chatbot.predict_class(userText)
    if predicted_intents:
        response = chatbot.get_response(predicted_intents, chatbot.intents)
    else:
        response = "I'm not sure I understand. Can you clarify?"
    return str(response)


if __name__ == "__main__":
    app.run()
