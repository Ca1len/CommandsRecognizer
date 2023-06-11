import socket
import traceback
import numpy as np
import json
from include.data_loader import read_json, remove_file
from include.model_script import predict, M5


MESSAGE_LENGTH = 8 * 1024 ** 2


class Server():
    def __init__(self, app):
        self.app = app

    def handle_request(self, message):
        try:
            np_array = read_json(message)
            pred, prob = predict(np_array, self.app.model)
            remove_file(message)
            return json.dumps({"prediction": pred, "probability": prob})
        except Exception as e:
            return json.dumps({"Error": str(traceback.format_exc())})

    def on_new_client(self, client_sock, addr):
        message = bytearray()
        while True:
            new_message = client_sock.recv(MESSAGE_LENGTH)
            # print(new_message[-3:])
            if not len(new_message):
                break
            message.extend(new_message)
            print(f"Message from {addr}:") #{message}")
            return_message = self.handle_request(message.decode())
            client_sock.send(return_message.encode())

    def start_server(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        HOST, PORT = socket.gethostbyname(socket.gethostname()), 8080

        server_socket.bind((HOST, PORT))
        server_socket.listen()

        while(True):
            (client_connection, client_address) = server_socket.accept()
            print("Accepted a connection request from %s:%s"%(client_address[0], client_address[1]))
            try:
                self.on_new_client(client_connection, client_address)
            except Exception as e:
                print(f"MAIN APP: {e}")

            print("End a connection request from %s:%s"%(client_address[0], client_address[1]))
