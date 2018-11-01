import socket
import gym
import json
import struct

HOST = '10.20.8.66'  # Standard loopback interface address (localhost)
PORT = 40000        # Port to listen on (non-privileged ports are > 1023)

ENVIROMENTS = {}

def serve():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            while True:
                cmd = conn.recv(1024).decode("utf-8").split("_")
                if len(cmd) > 1: 
                    response = handle_cmd(cmd)
                    #conn.send(bytes(response, "utf-8"))
                    send_msg(conn, bytes(response, "utf-8"))

def handle_cmd(cmd):
    if cmd[0] == "init":
        return init_gym("CarRacing-v0", cmd[1])
    elif cmd[0] == "sample":
        return step_sample(cmd[1])
    elif cmd[0] == "step":
        return step(cmd[1], cmd[2])
    elif cmd[0] == "reset":
        return reset(cmd[1])
    elif cmd[0] == "close":
        return close(cmd[1])


def init_gym(game, env_id):
    ENVIROMENTS[env_id] = gym.make(game)
    ENVIROMENTS[env_id].reset()
    return env_id

def step_sample(env_id):
    ENVIROMENTS[env_id].render(mode="rgb_array")
    action = ENVIROMENTS[env_id].action_space.sample()
    ob, rw, done, _ = ENVIROMENTS[env_id].step(action)
    return  json.dumps([ob.tolist(), rw, done, action.tolist()]) 

def step(env_id, action):
    action = json.loads(action)
    ENVIROMENTS[env_id].render(mode="rgb_array")
    ob, rw, done, _ = ENVIROMENTS[env_id].step(action)
    return  json.dumps([ob.tolist(), rw, done]) 

def reset(env_id):
    ENVIROMENTS[env_id].reset()
    return "Reseted " + env_id

def close(env_id):
    ENVIROMENTS[env_id].close()
    return "Closed " + env_id

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

serve()