import google.generativeai as genai
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
import sqlite3
from keras.models import load_model
import json
import random
from dotenv import load_dotenv
from flask_cors import CORS

# Load environment variables
load_dotenv()

conn1 = sqlite3.connect('example.db')
conn1.cursor().execute("CREATE TABLE IF NOT EXISTS chats (id INTEGER PRIMARY KEY, sessionid INTEGER, email TEXT, usermessage TEXT,botmessage TEXT, createdAt DATETIME DEFAULT CURRENT_TIMESTAMP)")
conn1.commit()

# Load words from a pickle file
with open('texts8.pkl', 'rb') as f:
    words = pickle.load(f)

# Load classes from a pickle file
with open('labels8.pkl', 'rb') as f:
    classes = pickle.load(f)

# Load your Keras model
model = load_model('model8.h5')

# Define the lemmatizer object
lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
    return(np.array(bag))

def predict_class(sentence, model, words):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Load intents from data.json
with open('data.json', 'r') as f:
    intents = json.load(f)

def getResponse(ints, intents_json):
    if not ints:
        return "I did not understand. Please ask me questions related to mental health only."
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg, words):
    ints = predict_class(msg, model, words)
    response = getResponse(ints, intents)
    return response

app = Flask(__name__)
app.static_folder = 'static'
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.post("/sendmessages")
def get_bot_response():
    body = request.get_json(force=True)
    userText = body['message']
    email = body['email']
    sessionId = body['sessionId']
    if sessionId is None:
        print ("returning log")
        return ''
    response = chatbot_response(userText, words)
    # Replace newline characters with <br> tags for web display
    response = response.replace('\n', '<br>')
    conn = sqlite3.connect('example.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chats (sessionid, email ,usermessage, botmessage) VALUES(?, ?, ?, ?)", [sessionId, email, userText, response])
    conn.commit()
    return response

@app.post("/getmessages")
def get_messages():
    body = request.get_json(force=True)
    email = body['email']
    conn = sqlite3.connect('example.db')
    cursor = conn.cursor()
    dbRes = cursor.execute("select sessionid, usermessage, botmessage from chats where email = ? order by createdAt desc", [email]).fetchall()
    # Replace newline characters with <br> tags for web display
    formatted_messages = []
    for message in dbRes:
        sessionid, usermessage, botmessage = message
        usermessage = usermessage.replace('\n', '<br>')
        botmessage = botmessage.replace('\n', '<br>')
        formatted_messages.append((sessionid, usermessage, botmessage))
    return formatted_messages

@app.post("/createnewchat")
def create_new_chat_session():
    body = request.get_json(force=True)
    email = body["email"]
    
    with sqlite3.connect('example.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT max(sessionId) FROM chats WHERE email = ?", (email,))
        max_sess_id = cursor.fetchone()[0]
            
        if max_sess_id is None:
             max_sess_id = 0
            
        new_sess_id = max_sess_id + 1
        cursor.execute("INSERT INTO chats (sessionId, email, usermessage, botmessage) VALUES (?, ?, ?, ?)", (new_sess_id, email, "", ""))
        conn.commit()
            
        return {"success": True, "sessionId": new_sess_id}

if __name__ == "__main__":
    app.run(debug=True)
