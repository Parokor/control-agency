import subprocess
import threading
import time
import socket
import urllib.request

def iframe_thread(port):
    """Wait for ComfyUI to start and then launch cloudflared tunnel"""
    while True:
        time.sleep(0.5)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', port))
        if result == 0:
            break
        sock.close()
    
    print("\nComfyUI finished loading, launching cloudflared tunnel\n")
    
    # Launch cloudflared tunnel
    p = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Monitor output to extract URL
    for line in p.stderr:
        l = line.decode()
        if "trycloudflare.com " in l:
            print("ComfyUI access URL:", l[l.find("http"):], end='')

# Start ComfyUI in a separate thread
def start_comfyui():
    """Start ComfyUI with appropriate parameters"""
    subprocess.run(["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188"])

if __name__ == "__main__":
    # Start the tunnel thread
    threading.Thread(target=iframe_thread, daemon=True, args=(8188,)).start()
    
    # Start ComfyUI
    start_comfyui()
