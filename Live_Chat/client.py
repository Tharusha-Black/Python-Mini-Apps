import socket
import threading
import sys

def receive_messages(client_socket):
    while True:
        try:
            msg = client_socket.recv(1024).decode('utf-8')
            print(msg)
        except:
            break

def start_client():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1', 5555))

    nickname = input("Enter your nickname: ")
    client_socket.send(f"NICK:{nickname}".encode('utf-8'))

    receive_thread = threading.Thread(target=receive_messages, args=(client_socket,))
    receive_thread.start()

    while True:
        msg = input()
        if msg.lower() in ["/quit", "/dc", "/disconnect"]:
            confirm = input("Are you sure you want to disconnect? (y/n): ")
            if confirm.lower() == "y":
                client_socket.send("/quit".encode('utf-8'))
                client_socket.close()
                print("Disconnected from server.")
                sys.exit(0)
            else:
                continue
        elif msg.startswith("/msg"):
            parts = msg.split(' ', 2)
            if len(parts) == 3:
                client_socket.send(f"MSG:{parts[1]}:{parts[2]}".encode('utf-8'))
        elif msg.startswith("/join"):
            parts = msg.split(' ', 2)
            if len(parts) == 2:
                client_socket.send(f"JOIN:{parts[1]}".encode('utf-8'))
        else:
            client_socket.send(f"MSG:ALL:{msg}".encode('utf-8'))  # Send messages to all clients

start_client()
