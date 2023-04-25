import socket

c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
c.connect(("127.0.0.1", 1234))

def sendjson(musicdata):
    musicdata = musicdata.encode()
    c.sendall(len(musicdata).to_bytes(4, 'big'))
    c.sendall(musicdata)

if __name__ == '__main__':
    while True:
        with open(input("json file:"), 'r') as f:
            musicdata = f.read()
        # data = c.recv(1024)
        # if not data:
        #    break
        sendjson(musicdata)

