import json
import socket
from typing import Any, Tuple


class ServerIPC:
    def __init__(self, conn: socket.socket, ip: socket._RetAddress):
        self._conn = conn
        self._ip = ip

    def send(self, cmd: str, data: Any):
        self._conn.send(json.dumps({'cmd': cmd, 'data': data}).encode())

    def recv(self) -> Tuple[str, Any]:
        request = self._conn.recv(1024)
        received_data = json.loads(request.decode())
        return received_data['cmd'], received_data['data']

    def close(self):
        self._conn.close()


class Server:

    def __init__(self):
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def serve(self):
        self._server.bind(('localhost', 12345))
        self._server.listen(5)

    def accept(self) -> ServerIPC:
        conn, ip = self._server.accept()
        return ServerIPC(conn, ip)


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
