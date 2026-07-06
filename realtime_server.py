import base64
import hashlib
import json
import os
import socket
import socketserver
import threading
from http.server import SimpleHTTPRequestHandler


class RealtimeServer:
    def __init__(self, host="127.0.0.1", port=8765, dashboard_dir="dashboard"):
        self.host = host
        self.port = port
        self.dashboard_dir = dashboard_dir
        self.clients = set()
        self.lock = threading.Lock()
        self.httpd = None
        self.thread = None

    def start(self):
        server = self
        dashboard_path = os.path.abspath(self.dashboard_dir)

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=dashboard_path, **kwargs)

            def log_message(self, format, *args):
                return

            def do_GET(self):
                if self.path == "/ws":
                    self._handle_websocket()
                    return
                if self.path == "/":
                    self.path = "/index.html"
                super().do_GET()

            def _handle_websocket(self):
                key = self.headers.get("Sec-WebSocket-Key")
                if not key:
                    self.send_error(400, "Missing WebSocket key")
                    return

                accept = base64.b64encode(
                    hashlib.sha1(
                        (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")
                    ).digest()
                ).decode("ascii")

                self.send_response(101, "Switching Protocols")
                self.send_header("Upgrade", "websocket")
                self.send_header("Connection", "Upgrade")
                self.send_header("Sec-WebSocket-Accept", accept)
                self.end_headers()

                sock = self.connection
                sock.settimeout(1.0)
                with server.lock:
                    server.clients.add(sock)

                try:
                    while True:
                        try:
                            data = sock.recv(2)
                        except socket.timeout:
                            continue
                        if not data:
                            break
                        opcode = data[0] & 0x0F
                        if opcode == 8:
                            break
                finally:
                    with server.lock:
                        server.clients.discard(sock)

        class ThreadedHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self.httpd = ThreadedHTTPServer((self.host, self.port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def broadcast(self, payload):
        message = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        frame = self._frame(message)

        with self.lock:
            clients = list(self.clients)

        for client in clients:
            try:
                client.sendall(frame)
            except OSError:
                with self.lock:
                    self.clients.discard(client)

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()

    def url(self):
        return f"http://{self.host}:{self.port}"

    @staticmethod
    def _frame(message):
        length = len(message)
        header = bytearray([0x81])

        if length < 126:
            header.append(length)
        elif length < 65536:
            header.append(126)
            header.extend(length.to_bytes(2, "big"))
        else:
            header.append(127)
            header.extend(length.to_bytes(8, "big"))

        return bytes(header) + message
