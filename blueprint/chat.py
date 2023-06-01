import json
from flask import Blueprint, render_template, request
from Agent.custom_agent import get_reply

api = Blueprint('API', __name__)

@api.route('/chat', methods=['GET'])
def chat_page():
    return render_template('index.html')


@api.route('/response', methods=['POST'])
def get_response():
    message = request.json.get('message')

    print('INPUT: ')
    print(message)

    reply = get_reply(message)
    return json.dumps({"reply": reply}, ensure_ascii=False)


