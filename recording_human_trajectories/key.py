from pynput.keyboard import Listener
import socket

# Listen for keyboard input on Linux
def on_key_event(key):
    # Send key information to the server
    try:
        key = key.char.encode()  # Regular keys
    except AttributeError:
        key = str(key).encode()  # Special keys like Ctrl, Alt, etc.
    client_socket.send(key)

# Assume the IP address of WSL is 172.26.5.84, and the port number is 8066
WSL_PORT = 8066
# Create a client socket and connect to the server within WSL
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('172.26.5.84', WSL_PORT))

# Listen for keyboard input
with Listener(on_press=on_key_event) as listener:
    listener.join()