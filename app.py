from flask import Flask, request, jsonify

from history_model import HistoryModel

app = Flask(__name__)

base_llm_model_name = "gpt2"
history_model = HistoryModel(base_llm_model_name)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_message = data.get("message", "")

    if user_message:
        response = history_model.generate_response(user_message)
        return jsonify({"message": response}), 200
    else:
        return jsonify({"error": "No message provided"}), 400


@app.route('/get_llm_name', methods=['GET'])
def get_llm_name():
    return jsonify({"llm_name": base_llm_model_name}), 200


if __name__ == '__main__':
    app.run(debug=True)
