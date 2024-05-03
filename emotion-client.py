import socket

# Socket setup
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = 'server_ip_here'  # Replace 'server_ip_here' with the server's IP address
port = 8080
client_socket.connect((server_address, port))

try:
    while True:
        data = client_socket.recv(1)
        if data:
            print("Emotion Code:", data.decode())
except KeyboardInterrupt:
    pass

client_socket.close()
