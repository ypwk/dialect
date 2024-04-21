#! /usr/bin/python3
import time
from flask import Flask, Response, jsonify
from flask_socketio import SocketIO, emit
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)
from langchain.llms.base import LLM
from langchain.vectorstores.pgvector import PGVector
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from typing import Optional
from threading import Thread
import psycopg2 as pg2
import torch.cuda

print("Finished imports")

# Choose a suitable model for the task
model_id = "Intel/neural-chat-7b-v3-1"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"On {device}")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16
).to(device)

print("Finished importing model")

# Room-based setup for handling multiple conversations
conversations = {}

philosophers = ["Aristotle", "Hume", "Kant", "Mill", "Schopenhauer"]

global stop_threads, dialogue_thread, num_clients_connected
num_clients_connected = 0
dialogue_thread = None
stop_threads = False

embedding_model_id = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_id,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

vdb_params = {
    "driver": "psycopg2",
    "host": "localhost",
    "port": 5432,
    "database": "vdb",
    "user": "lumi",
    "password": "lumipass",
}

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=vdb_params["driver"],
    host=vdb_params["host"],
    port=vdb_params["port"],
    database=vdb_params["database"],
    user=vdb_params["user"],
    password=vdb_params["password"],
)

COLLECTION_NAME = "chat_logs"
store = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=embedding_model
)


class CustomSocketLLM(LLM):
    streamer: Optional[TextIteratorStreamer] = None
    history = ["What makes a person morally good?"]
    question = "What makes a person morally good?"
    session_saved = False

    def _call(self, prompt, stop=None, run_manager=None) -> str:
        self.history.append("")
        self.streamer = TextIteratorStreamer(
            tokenizer=tokenizer, skip_prompt=True, timeout=5
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        kwargs = dict(
            input_ids=inputs.input_ids,
            max_new_tokens=500,
            streamer=self.streamer,
            pad_token_id=tokenizer.eos_token_id,
        )
        thread = Thread(target=model.generate, kwargs=kwargs)
        thread.start()
        return ""

    @property
    def _llm_type(self) -> str:
        return "custom"

    def stream_tokens(self):
        if not self.streamer.text_queue.empty():
            for token in self.streamer:
                time.sleep(0.05)
                self.history[-1] += token
                socketio.emit('new_token', {'token': token})
                print(token)
        else:
            time.sleep(1)
            self.stream_tokens()

    def build_context(self):
        """Construct the context string from the cache."""
        context_parts = []
        for idx in range(min(len(self.history), 5)):
            context_parts.append(self.history[-idx])
        context_parts.pop()
        return "\n".join(context_parts)


tokenizer.pad_token_id = model.config.eos_token_id

template = """You are the famous philosopher {philosopher}. You are engaging in a debate with a fellow philosopher from another time period.
Here is the previous response: {prompt}
{context}
Always respond politely and engage the opponent with clear, understandable language.
Your response:"""
prompt = PromptTemplate.from_template(template)

socket_llm = CustomSocketLLM()
socket_chain = prompt | socket_llm

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


def dialogue(phil_idx: int):
    context_string = "Context:\n"
    context_string += socket_llm.build_context()
    prompt = socket_llm.history[-1]
    docs_with_score = store.similarity_search_with_score(prompt)
    for doc, score in docs_with_score:
        if doc.metadata.get("author") == philosophers[phil_idx]:
            context_string += doc.page_content
    print(context_string)
    socket_chain.invoke(input=dict({"context": context_string, "prompt": prompt, "philosopher": philosophers[phil_idx]}))
    socketio.emit('new_philosopher', {'token': f"{philosophers[phil_idx]}:"}, namespace='/')
    socket_llm.stream_tokens()


def start_dialogue():
    global stop_threads
    curr_idx = 0
    while not stop_threads:
        dialogue(curr_idx)
        curr_idx = (curr_idx + 1) % 2
        time.sleep(2)  # Add a small delay to prevent CPU overload.
        socket_llm.history.pop(0)
        print(socket_llm.history)
    stop_threads = False
    print("nobody here no more")


@app.route('/')
def index():
    return "waugh"


@socketio.on('join')
def on_join():
    print("SOMEONE JPIEND YEAR")
    global num_clients_connected
    num_clients_connected += 1
    if num_clients_connected == 1:
        dialogue_thread = Thread(target=start_dialogue)
        dialogue_thread.start()


@socketio.on('leave')
def on_leave():
    global num_clients_connected
    num_clients_connected -= 1
    if num_clients_connected == 0:
        global stop_threads
        stop_threads = True


if __name__ == '__main__':
    socketio.run(app, debug=True)
