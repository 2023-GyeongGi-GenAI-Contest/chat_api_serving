from flask import Flask
from flask_cors import CORS

from blueprint import chat

app = Flask(__name__)

app.register_blueprint(chat.api, url_prefix='/chat')

cors = CORS(app, origins='http://127.0.0.1:5000', supports_credentials=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')