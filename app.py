import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np


import sqlite3

conn1 = sqlite3.connect('example.db')
conn1.cursor().execute("CREATE TABLE IF NOT EXISTS chats (id INTEGER PRIMARY KEY, sessionid INTEGER, email TEXT, usermessage TEXT,botmessage TEXT, createdAt DATETIME DEFAULT CURRENT_TIMESTAMP)")
conn1.commit()



from keras.models import load_model
model = load_model('model.h5')
import json
import random
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    # Check if the intents list is empty
    if not ints:
        # Return a default response if the list is empty
        return "I did not understand. Please ask me questions related to mental health only."
    
    # If the list is not empty, proceed as before
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


from flask import Flask, render_template, request
from flask_cors import CORS

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
    response = chatbot_response(userText)
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
    return dbRes


@app.post("/createnewchat")
def create_new_chat_session():
    body = request.get_json(force=True)
    email = body["email"]
    conn = sqlite3.connect('example.db')
    cursor = conn.cursor()
    max_sess_id = cursor.execute("select max(sessionid) from chats where email = ?", [email]).fetchone()[0]
    print("maxsessid",str(max_sess_id))
    cursor.execute("INSERT INTO chats (sessionId, email ,usermessage, botmessage) VALUES(?, ?, ?, ?)", [max_sess_id + 1, email, "", ""])
    conn.commit()
    return {"success": True}

if __name__ == "__main__":
    app.run(debug=True)