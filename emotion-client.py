import socket

HOST = 'server_ip_address'  # The server's hostname or IP address
PORT = 65432                # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    while True:
        data = s.recv(1024)
        if not data:
            break
        print("Received:", data.decode())
