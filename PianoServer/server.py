import socket
import json

def send_json(conn, musicdata):
    musicdata_bytes = musicdata.encode()
    conn.sendall(len(musicdata_bytes).to_bytes(4, 'big'))
    conn.sendall(musicdata_bytes)

def recv_json(conn):
    length_data = recvall(conn, 4)
    if not length_data:
        return None

    length = int.from_bytes(length_data, byteorder='big')
    data_str = recvall(conn, length).decode('utf-8')

    print(f"Received raw data: {data_str}")

    received_data = json.loads(data_str)
    return received_data

def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

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
                    print("test")
                send_json(c, musicdata)
