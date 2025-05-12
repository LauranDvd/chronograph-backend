import logging
import logging
import sys
import threading

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from history_model import HistoryModel
from human_feedback.human_feedback import save_feedback_to_csv
from project_constants import LLM_MODEL_NAME

app = Flask(__name__)

CORS(app, origins=["http://localhost:5173"])

socketio = SocketIO(app, cors_allowed_origins="*")

logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

history_model = HistoryModel(LLM_MODEL_NAME)

chat_histories = {}
chat_histories_locks = {}


def add_to_history(message: str, type: str, sid: str) -> None:
    if type != "user" and type != "assistant":
        raise ValueError("Invalid message type. Must be 'user' or 'assistant'.")

    message = {"role": type, "content": message}

    print(f"Acquiring lock for {request.sid}")
    chat_histories_locks[sid].acquire()
    chat_histories[sid].append(message)
    chat_histories_locks[sid].release()


@socketio.on("connect")
def handle_connect():
    sid = request.sid
    logger.log(logging.INFO, "Client connected with sid=%s", sid)
    chat_histories[sid] = []
    chat_histories_locks[sid] = threading.Lock()


@socketio.on("message")
def handle_message(message):
    logger.log(logging.INFO, "Received message on WS: %s", message)
    user_message = message.get("message", "")

    if user_message:
        add_to_history(user_message, "user", request.sid)
        response = history_model.generate_response(chat_histories[request.sid])
        add_to_history(response, "assistant", request.sid)

        emit("message", {"message": response})
    else:
        emit("message", {"error": "No message provided"})


@socketio.on("like")
def handle_like(data):
    logger.log(logging.INFO, f"Received like request: {data}")
    sid = request.sid
    index = data.get("index")

    try:
        message = chat_histories[sid][index]
        logger.log(logging.INFO, f"User liked message at index {index} from session {sid}: {message['content']}")
        save_feedback_to_csv(sid, chat_histories[sid], index, "like")
    except KeyError:
        logger.log(logging.INFO, f"Like failed: invalid session {sid}")
    except IndexError:
        logger.log(logging.INFO, f"Like failed: invalid index {index}")


@socketio.on("dislike")
def handle_dislike(data):
    logger.log(logging.INFO, f"Received dislike request: {data}")
    sid = request.sid
    index = data.get("index")

    try:
        message = chat_histories[sid][index]
        logger.log(logging.INFO, f"User disliked message at index {index} from session {sid}: {message['content']}")
        save_feedback_to_csv(sid, chat_histories[sid], index, "dislike")
    except KeyError:
        logger.log(logging.INFO, f"Dislike failed: invalid session {sid}")
    except IndexError:
        logger.log(logging.INFO, f"Dislike failed: invalid index {index}")


@app.route("/get_llm_name", methods=["GET"])
def get_llm_name():
    return jsonify({"llm_name": LLM_MODEL_NAME}), 200


if __name__ == "__main__":
    # app.run(debug=True)
    socketio.run(app, debug=True)
