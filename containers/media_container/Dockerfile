FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y python3 python3-pip git wget

WORKDIR /app

# Clone ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI .

# Install requirements
RUN pip3 install -r requirements.txt

# Clone popular extensions
RUN mkdir -p custom_nodes
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git custom_nodes/ComfyUI-Manager

# Install cloudflared for tunneling
RUN wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb && \
    dpkg -i cloudflared-linux-amd64.deb && \
    rm cloudflared-linux-amd64.deb

# Copy launcher script
COPY comfyui_launcher.py .

# Expose port
EXPOSE 8188

# Start command
CMD ["python3", "comfyui_launcher.py"]
