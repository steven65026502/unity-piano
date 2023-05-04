import socket
import json

def sendjson(c, musicdata):
    musicdata = musicdata.encode()
    c.sendall(len(musicdata).to_bytes(4, 'big'))
    c.sendall(musicdata)

def wait_for_start_signal(c):
    data = c.recv(1024).decode()
    print("Received data:", data)  # Add this line
    if data:
        try:
            received_data = json.loads(data)
            print("Received JSON:", received_data)  # Add this line
            if received_data["signal"] == "start":
                return True, received_data["temperature"], received_data["p"], received_data["min_length"]
        except json.JSONDecodeError:
            print("Failed to decode JSON.")  # Add this line
            pass
    return False, None, None, None


if __name__ == '__main__':
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 1234))
        s.listen()
        c, addr = s.accept()
        with c:
            print(addr, "connected.")
            while True:
                start_signal, received_temperature, received_p, received_min_length = wait_for_start_signal(c)
                if start_signal:
                    with open(input("json file:"), 'r') as f:
                        musicdata = f.read()
                    sendjson(c, musicdata)
