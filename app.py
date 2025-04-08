import logging
import sys
import threading

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit

from domain.message import Message, MessageSender
from history_model import HistoryModel

app = Flask(__name__)

socketio = SocketIO(app)

logger = logging.getLogger()

handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

base_llm_model_name = "gpt2"
history_model = HistoryModel(base_llm_model_name)

chat_histories = {}
chat_histories_locks = {}


@socketio.on('connect')
def handle_connect():
    sid = request.sid
    logger.log(logging.INFO, "Client connected with sid=%s", sid)
    chat_histories[sid] = []
    chat_histories_locks[sid] = threading.Lock()


@socketio.on('message')
def handle_message(message):
    logger.log(logging.INFO, "Received message on WS: %s", message)
    user_message = message.get('message', '')

    message = Message(user_message, MessageSender.USER)

    print(f"Acquiring lock for {request.sid}")
    chat_histories_locks[request.sid].acquire()
    chat_histories[request.sid].append(message)
    number_of_messages = len(chat_histories[request.sid])
    chat_histories_locks[request.sid].release()

    if user_message:
        response = history_model.generate_response(user_message)
        response = "Response to your latest (" + str(number_of_messages) + "th) message: " + response
        emit('message', {'message': response})
    else:
        emit('message', {'error': 'No message provided'})


@app.route('/get_llm_name', methods=['GET'])
def get_llm_name():
    return jsonify({"llm_name": base_llm_model_name}), 200


if __name__ == '__main__':
    # app.run(debug=True)
    socketio.run(app, debug=True)
