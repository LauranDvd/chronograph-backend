import os
import logging
import sys
import threading

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from langchain_chroma import Chroma

import embedding_model
from domain.message import Message, MessageSender
from history_model import HistoryModel
from flask_cors import CORS
from project_constants import LLM_MODEL_NAME, DATABASE_PATH, EMBEDDING_MODEL_NAME

app = Flask(__name__)

CORS(app, origins=["http://localhost:5173"])

socketio = SocketIO(app, cors_allowed_origins="*")

logger = logging.getLogger()

handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

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


@app.route("/get_llm_name", methods=["GET"])
def get_llm_name():
    return jsonify({"llm_name": LLM_MODEL_NAME}), 200


if __name__ == "__main__":
    # app.run(debug=True)
    # check if the quering form the db works

    socketio.run(app, debug=True)
