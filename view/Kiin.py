import json
import uuid
from flask import Blueprint, render_template, request, session, make_response
from controller import build_memory, Kiin_AI_Response
from datetime import datetime, timedelta

Kiin = Blueprint('Kiin', __name__)

def generate_new_session_id():
    session_id = str(uuid.uuid4())
    return session_id

@Kiin.route('/chat', methods=['GET'])
def chat_page():
    session_id = request.cookies.get('session')
    if session_id is None:
        session_id = generate_new_session_id()  # 새로운 세션 ID 생성
        session['session'] = session_id
        print('최초접속 -> 새로운 세션을 생성합니다.')
    else:
        session['session'] = session_id
        print('session: ' + session_id)

    # session['UserChat'] = session.get('UserChat', [{"chatnum": 0, "chatmessage": "this is default userchat"}])
    # session['AIChat'] = session.get('AIChat', [{"chatnum": 0, "chatmessage": "this is default aichat"}])
    session['UserChat'] = session.get('UserChat', [])
    session['AIChat'] = session.get('AIChat', [])
    session['memory'] = build_memory.build_memory(session_id)

    # 세션 쿠키를 영구 쿠키로 변경
    expires = datetime.now() + timedelta(days=30)  # 30일 후로 설정
    response = make_response(render_template('index.html', UserChat=session['UserChat'], AIChat=session['AIChat']))
    response.set_cookie('session', session_id, expires=expires)

    return response


@Kiin.route('/response', methods=['POST'])
def get_response():
    session_id = request.cookies.get('session')
    print('session '+session_id)

    memory = session.get('memory')

    chatnum = len(session['UserChat'])
    message = request.json.get('message')

    print('INPUT: ')
    print(message)

    session['UserChat'].append({"chatnum": chatnum, "chatmessage": message})
    session['AIChat'].append({"chatnum": chatnum, "chatmessage": Kiin_AI_Response.Agent(session['UserChat'][-1]["chatmessage"], memory)})
    session.modified = True

    print('POST SUCCESS')
    print('RESULT: ')
    print({"UserChat": session['UserChat'], "AIChat": session['AIChat']})
    return json.dumps({"UserChat": session['UserChat'], "AIChat": session['AIChat']}, ensure_ascii=False)


