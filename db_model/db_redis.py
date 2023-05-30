import pickle
import redis

def get_chat(session_id):
    r = redis.Redis(host='localhost', port=6379, db=0)
    value = r.get('session:'+session_id)
    if value == None:
        return None
    decoded_value = pickle.loads(value)
    return decoded_value



