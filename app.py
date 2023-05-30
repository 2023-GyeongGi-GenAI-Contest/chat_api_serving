import redis
from flask import Flask
from flask_cors import CORS
from flask_session import Session

from view import Kiin

app = Flask(__name__)

app.register_blueprint(Kiin.Kiin, url_prefix='/Kiin')

app.config.from_mapping(
    SECRET_KEY='dev',  # 다른 보안키를 사용해주세요
    SESSION_TYPE='redis',
    SESSION_REDIS=redis.Redis(host='localhost', port=6379)  # redis server의 호스트와 포트를 적어주세요
)
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = False

cors = CORS(app, origins='http://127.0.0.1:5000', supports_credentials=True)
Session(app)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')