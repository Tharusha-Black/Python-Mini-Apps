import socket
import threading
import tkinter as tk

class ServerGUI:
    def __init__(self, host, port):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen(5)

        self.root = tk.Tk()
        self.root.title("Chat Server")

        self.client_listbox = tk.Listbox(self.root)
        self.client_listbox.pack(fill=tk.BOTH, expand=True)

        self.receive_thread = threading.Thread(target=self.accept_clients)
        self.receive_thread.start()

        self.root.mainloop()

    def accept_clients(self):
        while True:
            client_socket, client_address = self.server_socket.accept()
            threading.Thread(target=self.handle_client, args=(client_socket, client_address)).start()

    def handle_client(self, client_socket, client_address):
        while True:
            try:
                msg = client_socket.recv(1024).decode('utf-8')
                if msg:
                    self.client_listbox.insert(tk.END, f"{client_address[0]}:{client_address[1]} - {msg}")
            except:
                break

if __name__ == "__main__":
    ServerGUI('127.0.0.1', 5555)
