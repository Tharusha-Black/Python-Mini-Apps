import socket
import threading
import sys

# Function to continuously receive messages from the server
def receive_messages(client_socket):
    while True:
        try:
            msg = client_socket.recv(1024).decode('utf-8')
            print(msg)  # Display all messages (modify as needed)
        except ConnectionError:
            break

# Function to start the chat client
def start_client():
    # Prompt the user to enter their username
    nickname = input("Enter your User Name: ")

    # Create a socket and connect to the server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1', 5555))

    # Send the username to the server
    client_socket.send(f"User Name:{nickname}".encode('utf-8'))

    # Start a thread to continuously receive messages from the server
    receive_thread = threading.Thread(target=receive_messages, args=(client_socket,))
    receive_thread.start()

    # Main loop for sending messages
    while True:
        msg = input()  # Get input from the user

        if msg.lower() in ["/quit", "/dc", "/disconnect"]:
            # Ask for confirmation before disconnecting
            confirm = input("Are you sure you want to disconnect? (y/n): ")
            if confirm.lower() == "y":
                # Send quit command and close the socket
                client_socket.send("/quit".encode('utf-8'))
                client_socket.close()
                print("Disconnected from server.")
                sys.exit(0)
        elif msg.startswith("/msg"):
            # Extract recipient and message content from input
            parts = msg.split(' ', 2)
            if len(parts) == 3:
                # Send the formatted private message
                client_socket.send(f"/msg {parts[1]} {parts[2]}".encode('utf-8'))
            else:
                print("Invalid private message format. Use /msg <recipient> <message>")
        else:
            # Send a message to all users
            client_socket.send(f"{msg}".encode('utf-8'))

# Entry point of the script
if __name__ == "__main__":
    start_client()
