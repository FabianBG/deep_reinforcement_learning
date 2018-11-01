import socket
import json
import struct
import random

HOST = 'it01-r4-ln-01.yachay.ep'  # The server's hostname or IP address
PORT = 40000        # The port used by the server
ENV_ID = "A"

def connect():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    return s

def init(s):
    s.send(bytes('init_' + ENV_ID, "utf-8"))
    print(repr(s.recv(1024)))

def step_sample(s):
    s.send(bytes('sample_' + ENV_ID, "utf-8"))
    data = recv_msg(s).decode("utf-8")
    return  json.loads(data) 

def step(s, action):
    s.send(bytes('step_' + ENV_ID + '_' + json.dumps(action), "utf-8"))
    data = recv_msg(s).decode("utf-8")
    return  json.loads(data) 

def steps(s, action):
    data = ""
    total_rw = 0
    for _ in range(random.choice([2,3,4])):
        s.send(bytes('step_' + ENV_ID + '_' + json.dumps(action), "utf-8"))
        data = recv_msg(s).decode("utf-8")
        data = json.loads(data)
        total_rw = data[1] + total_rw
    data[1] = total_rw
    return  data

def reset(s):
    s.send(bytes('reset_' + ENV_ID, "utf-8"))
    print(repr(s.recv(1024)))

def close(s):
    s.send(bytes('close_' + ENV_ID, "utf-8"))
    print(repr(s.recv(1024)))

def send_msg(sock, msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

