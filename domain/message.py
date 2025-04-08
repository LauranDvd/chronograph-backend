from enum import Enum


class MessageSender(Enum):
    USER = 1
    MODEL = 2


class Message:
    def __init__(self, content: str, sender: MessageSender):
        self.content = content
        self.sender = sender

    def __repr__(self):
        return f"{self.sender}: {self.content}"

