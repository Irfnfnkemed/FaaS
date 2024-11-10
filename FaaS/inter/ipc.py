import json
import socket
import subprocess
from typing import Any, Tuple


def get_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception as e:
        ip = '127.0.0.1'
    s.close()
    return ip


def get_free_port(ip: str) -> int:
    if ip == "" or ip == '127.0.0.1' or ip == 'localhost':
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return port
    else:
        code = """
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
                """
        cmd = [
            "ssh", ip,
            "python3", "-c",
            f"'{code}'"
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return int(result.stdout)
        else:
            return 0


class ServerIPC:
    def __init__(self, conn: socket.socket, ip: str):
        self._conn = conn
        self._ip = ip

    def send(self, cmd: str, data: Any):
        print("send", cmd, data)
        self._conn.send(json.dumps({'cmd': cmd, 'data': data}).encode())

    def recv(self) -> Tuple[str, Any]:
        request = self._conn.recv(1024)
        print("recv", request)
        received_data = json.loads(request.decode())
        return received_data['cmd'], received_data['data']

    def close(self):
        self._conn.close()


class Server:

    def __init__(self, port: int):
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._port = port

    def serve(self):
        self._server.bind(('localhost', self._port))
        self._server.listen(5)

    def accept(self) -> ServerIPC:
        conn, ip = self._server.accept()
        return ServerIPC(conn, ip)

    def close(self):
        self._server.close()

    def get_port(self) -> int:
        return self._server.getsockname()[1]


class ClientIPC:
    def __init__(self):
        self._conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self, host: str, port: int):
        self._conn.connect((host, port))

    def send(self, cmd: str, data: Any):
        self._conn.send(json.dumps({'cmd': cmd, 'data': data}).encode())

    def recv(self) -> Tuple[str, Any]:
        request = self._conn.recv(1024)
        received_data = json.loads(request.decode())
        return received_data['cmd'], received_data['data']

    def close(self):
        self._conn.close()
