import socket
import json
import numpy as np


# PORT = 8080 # For local server
PORT = 5000 # For docker server
HOST = socket.gethostbyname(socket.gethostname())


# Data example for request for NN. Must be List type. 
# If you got data stored in file, then read it and make one-dimensional List.
data = np.random.rand(8000).tolist()


#===============================================
def connect():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))
    return client_socket


MESSAGE_LENGTH = 8 * 1024 ** 2


# Function for sending data to server
def send_and_recive(sock, request):
    file_path = "data/temp_data/request_array.json"
    with open(file_path, "w") as file:
        json.dump(request, file)

    sock.send(file_path.encode())
    response = sock.recv(MESSAGE_LENGTH).decode()
    response = json.loads(response)
    return response


def main():
    sock = connect()

    json_data = {"data": data}

    response = send_and_recive(sock, json_data)
    print(response)


if __name__ == "__main__":
    main()
