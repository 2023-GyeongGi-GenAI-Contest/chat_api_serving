from flask import Flask, jsonify
from flask_cors import CORS
from flask import request
import json

from Agent.chat_agent import get_reply

app = Flask(__name__)

#CORS(app, origins='http://203.250.148.52:28881', supports_credentials=True)
CORS(app)

@app.route("/gg/genai/chat", methods=['POST'])
def gg_make_chat():
    msgNum = int(request.json.get("msgNum"))
    sessionId = request.json.get("sessionId")
    clientId = request.json.get("clientId")
    msg = request.json.get("text")
    reply = get_reply(msg)
    response = {"sessionId": sessionId, "msgNum": str(msgNum+1),"msgType": "1", "text": reply,"clientId": clientId}
    return jsonify(response)

# @app.route("/gg/genai/keyword", methods=['POST'])
# def gg_make_keyword():
#     return None
