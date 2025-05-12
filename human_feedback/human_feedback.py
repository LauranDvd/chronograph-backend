import csv
import os
from datetime import datetime

FEEDBACK_LOG_FILE = "chat_feedback_log.csv"


def save_feedback_to_csv(session_id, chat_history, index, feedback_type):
    if index >= len(chat_history):
        return

    assistant_response = chat_history[index]["content"]

    user_message = ""
    for i in range(index - 1, -1, -1):
        if chat_history[i]["role"] == "user":
            user_message = chat_history[i]["content"]
            break
    user_message = user_message.split("Here are some chunks")[0].strip()

    file_exists = os.path.isfile(FEEDBACK_LOG_FILE)
    with open(FEEDBACK_LOG_FILE, mode="a", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["feedback", "user_message", "assistant_response"])

        writer.writerow([
            feedback_type,
            user_message,
            assistant_response
        ])
