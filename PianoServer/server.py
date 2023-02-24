import socket
import json



with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("127.0.0.1", 1234))
    s.listen()
    c, addr = s.accept()
    with c:
        print(addr, "connected.")

        while True:
            with open(input("json file:"), 'r') as f:
                musicdata = f.read().encode()
            # data = c.recv(1024)
            # if not data:
            #    break
            c.sendall(len(musicdata).to_bytes(4, 'big'))
            c.sendall(musicdata)
