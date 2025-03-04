# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install curl and Docker client
RUN apt-get update && apt-get install -y curl docker.io

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install Python dependencies
RUN pip install requests langchain

# Copy the updated Python script into the container
COPY llm_conversation.py .

# Set a default topic environment variable (can be overridden)
ENV TOPIC=""

# Create a script to start Ollama, check its status, and run the conversation
RUN echo '#!/bin/bash\n\
ollama serve &\n\
sleep 10\n\
ollama pull phi3:mini\n\
python llm_conversation.py' > /app/start.sh && chmod +x /app/start.sh

# Run the start script when the container launches
CMD ["/app/start.sh"]
~                     
