import socket
import threading

class ChatServer:
    def __init__(self, host, port):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen(5)
        self.clients = {}  # Store client sockets and nicknames
        self.accept_clients()

    def accept_clients(self):
        print("Server is running and listening for connections...")
        while True:
            client_socket, client_address = self.server_socket.accept()
            print(f"New connection from {client_address}")
            threading.Thread(target=self.handle_client, args=(client_socket,)).start()

    def handle_client(self, client_socket):
        nickname = client_socket.recv(1024).decode('utf-8').split(':')[1]
        self.clients[nickname] = client_socket  # Add client socket to dictionary
        print(f"{nickname} connected")
        while True:
            try:
                msg = client_socket.recv(1024).decode('utf-8')
                if msg.startswith("/msg"):
                    parts = msg.split(' ', 3)  # Split into 3 parts
                    if len(parts) >= 3:  # Check if split parts contain all required components
                        recipient = parts[1]
                        message = ' '.join(parts[2:])  # Concatenate all elements from index 2 onwards
                        self.send_private_message(nickname, recipient, message)
                else:
                    self.broadcast_message(nickname, msg)
                    print(f"Received message from {nickname}: {msg}")
            except:
                print(f"{nickname} disconnected")
                del self.clients[nickname]  # Remove client from dictionary
                break
            
    def broadcast_message(self, sender, msg):
        for nick, client_socket in self.clients.items():
            if nick != sender:  # Only send message to other clients
                client_socket.send(f"{sender}: {msg}".encode('utf-8'))

    def send_private_message(self, sender, recipient, message):
        if recipient in self.clients:
            recipient_socket = self.clients[recipient]
            recipient_socket.send(f"[Private from {sender}]: {message}".encode('utf-8'))
            sender_socket = self.clients[sender]
            sender_socket.send(f"[Private to {recipient}]: {message}".encode('utf-8'))
        else:
            sender_socket = self.clients[sender]
            sender_socket.send(f"User {recipient} not found.".encode('utf-8'))



if __name__ == "__main__":
    chat_server = ChatServer('127.0.0.1', 5555)
