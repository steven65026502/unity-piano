import socket


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("127.0.0.1",1234))
    s.listen()
    c, addr = s.accept()
    with c:
        print(addr,"connected.")

        while True:
            json = input("piano json:").encode()
            #data = c.recv(1024)
            #if not data:
            #    break
            c.sendall(len(json).to_bytes(4, 'big'))
            c.sendall(json)
