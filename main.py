from include.app import App
from include.server import Server
from include.model_script import create_model, M5


model = create_model()
app = App(model)
server_app = Server(app)


def main():
    # print(model)
    print("Server starts")
    server_app.start_server()


if __name__ == "__main__":
    main()