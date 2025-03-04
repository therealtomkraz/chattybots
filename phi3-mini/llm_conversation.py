import os
import requests
import time
import json
import subprocess
import random
import sys
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

def check_ollama_status():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        print("Ollama status:\n", result.stdout)
    except Exception as e:
        print("Error checking Ollama status:", e)

def check_api_response():
    url = "http://localhost:11434/api/generate"
    payload = {"model": "phi3:mini", "prompt": "Hello, world!"}
    response = requests.post(url, json=payload, stream=True)
    print(f"API Response Status: {response.status_code}")
    print("API Response Content:")
    for line in response.iter_lines():
        if line:
            print(line.decode('utf-8'))

def initialize_memory():
    return ConversationBufferMemory(return_messages=True)

def generate_response(prompt, memory):
    url = "http://localhost:11434/api/generate"
    history = memory.chat_memory.messages
    context = "\n".join([f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in history])
    full_prompt = f"Conversation history:\n{context}\n\nHuman: {prompt}\nAI:"

    payload = {"model": "phi3:mini", "prompt": full_prompt}
    response = requests.post(url, json=payload, stream=True)

    full_response = ""
    for line in response.iter_lines():
        if line:
            try:
                json_response = json.loads(line)
                if 'response' in json_response:
                    full_response += json_response['response']
                if json_response.get('done', False):
                    break
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {line}")

    return full_response.strip() if full_response else None

def print_slowly(text, speaker, delay=0.05):
    if speaker == "LLM1":
        sys.stdout.write("\033[1;33m")  # Bold yellow for LLM1
    elif speaker == "LLM2":
        sys.stdout.write("\033[1;36m")  # Bold cyan for LLM2

    sys.stdout.write(f"{speaker}: ")
    sys.stdout.flush()
    time.sleep(delay)

    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)

    sys.stdout.write("\033[0m\n")  # Reset color
    sys.stdout.flush()

def conversation(turns=25, topic=None):
    memory = initialize_memory()

    if not topic:
        topics = ["technology", "philosophy", "space exploration", "artificial intelligence", "history"]
        topic = random.choice(topics)

    print_slowly(f"Topic: {topic}", "System")

    prompt1 = f"Let's talk about {topic}. What are your thoughts?"
    prompt2 = ""

    for i in range(turns):
        print_slowly(prompt1, "LLM1")
        memory.chat_memory.add_user_message(prompt1)

        prompt2 = generate_response(prompt1, memory)
        if prompt2:
            print_slowly(prompt2, "LLM2")
            memory.chat_memory.add_ai_message(prompt2)
        else:
            print("LLM2 failed to respond.")
            break

        prompt1 = generate_response(prompt2, memory)
        if prompt1:
            memory.chat_memory.add_user_message(prompt1)
        else:
            print("LLM1 failed to respond.")
            break

    print_slowly("Conversation ended.", "System")

if __name__ == "__main__":
    print_slowly("Starting conversation...", "System")

    # Check Ollama status before starting the conversation
    check_ollama_status()

    # Verify API response before starting the conversation
    check_api_response()

    time.sleep(10)  # Give Ollama some time to start up

    # Get topic from environment variable or default to None
    topic = os.getenv("TOPIC")

    # Start conversation with optional topic
    conversation(topic=topic)
