import socket

def sendjson(c, musicdata):
    musicdata = musicdata.encode()
    c.sendall(len(musicdata).to_bytes(4, 'big'))
    c.sendall(musicdata)

def wait_for_start_signal(c):
    data = c.recv(1024).decode()
    if data == 'start':
        return True
    return False

if __name__ == '__main__':
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 1234))
        s.listen()
        c, addr = s.accept()
        with c:
            print(addr, "connected.")
            while True:
                with open(input("json file:"), 'r') as f:
                    musicdata = f.read()
                sendjson(c, musicdata)
