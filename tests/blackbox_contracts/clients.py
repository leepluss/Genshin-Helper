"""Pure-stdlib black-box protocol clients (HTTP, RFC 6455 WebSocket, Native Messaging)."""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import http.client
import json
import os
import queue
import socket
import struct
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class HttpResponse:
    status: int
    reason: str
    headers: Dict[str, str]
    raw_headers: List[Tuple[str, str]]
    body: bytes

    def header(self, name: str, default: Optional[str] = None) -> Optional[str]:
        return self.headers.get(name.lower(), default)

    def text(self, encoding: str = "utf-8") -> str:
        return self.body.decode(encoding, errors="replace")

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))


class HttpClient:
    """Black-box HTTP/1.1 client supporting large (>2 MiB) bodies and non-2xx responses."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, timeout: float = 10.0) -> None:
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)

    def request(
        self,
        method: str,
        path: str,
        *,
        body: Optional[bytes | str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> HttpResponse:
        raw_body: Optional[bytes]
        if body is None:
            raw_body = None
        elif isinstance(body, bytes):
            raw_body = body
        else:
            raw_body = body.encode("utf-8")

        req_headers = dict(headers or {})
        conn = http.client.HTTPConnection(
            self.host,
            self.port,
            timeout=self.timeout if timeout is None else float(timeout),
        )
        try:
            conn.request(method.upper(), path, body=raw_body, headers=req_headers)
            resp = conn.getresponse()
            resp_body = resp.read()
            raw_hdrs = list(resp.getheaders())
            hdr_map = {k.lower(): v for k, v in raw_hdrs}
            return HttpResponse(
                status=int(resp.status),
                reason=str(resp.reason),
                headers=hdr_map,
                raw_headers=raw_hdrs,
                body=resp_body,
            )
        finally:
            conn.close()

    def get(
        self,
        path: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> HttpResponse:
        return self.request("GET", path, headers=headers, timeout=timeout)

    def post_json(
        self,
        path: str,
        data: Any = None,
        *,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> HttpResponse:
        payload = b"{}" if data is None else json.dumps(data, ensure_ascii=False).encode("utf-8")
        hdrs = {"Content-Type": "application/json; charset=utf-8"}
        if headers:
            hdrs.update(headers)
        return self.request("POST", path, body=payload, headers=hdrs, timeout=timeout)

    def options(
        self,
        path: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> HttpResponse:
        return self.request("OPTIONS", path, headers=headers, timeout=timeout)


class WsHandshakeError(Exception):
    def __init__(self, status: int, headers: Dict[str, str], body: bytes) -> None:
        super().__init__(f"WebSocket handshake failed with HTTP {status}")
        self.status = status
        self.headers = headers
        self.body = body


class WsClient:
    """Pure-stdlib RFC 6455 WebSocket client over raw TCP socket."""

    _WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

    def __init__(
        self,
        host: str,
        port: int,
        path: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 5.0,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.path = path
        self.sock = socket.create_connection((self.host, self.port), timeout=timeout)
        self.sock.settimeout(timeout)
        self._buf = bytearray()
        self.response_headers: Dict[str, str] = {}
        self._handshake(headers or {})

    def _handshake(self, extra_headers: Dict[str, str]) -> None:
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        lines = [
            f"GET {self.path} HTTP/1.1",
            f"Host: {self.host}:{self.port}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {key}",
            "Sec-WebSocket-Version: 13",
        ]
        for k, v in extra_headers.items():
            lines.append(f"{k}: {v}")
        raw_req = ("\r\n".join(lines) + "\r\n\r\n").encode("utf-8")
        self.sock.sendall(raw_req)

        while b"\r\n\r\n" not in self._buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("Socket closed during WebSocket handshake")
            self._buf.extend(chunk)

        sep = bytes(self._buf).index(b"\r\n\r\n")
        head_bytes = bytes(self._buf[:sep])
        del self._buf[: sep + 4]

        head_lines = head_bytes.decode("utf-8", errors="replace").split("\r\n")
        status_parts = head_lines[0].split(" ", 2)
        status = int(status_parts[1]) if len(status_parts) >= 2 and status_parts[1].isdigit() else 0
        hdrs: Dict[str, str] = {}
        for hline in head_lines[1:]:
            if ":" in hline:
                k, v = hline.split(":", 1)
                hdrs[k.strip().lower()] = v.strip()

        if status != 101:
            content_len = int(hdrs.get("content-length", "0") or "0")
            while len(self._buf) < content_len:
                chunk = self.sock.recv(4096)
                if not chunk:
                    break
                self._buf.extend(chunk)
            body = bytes(self._buf[:content_len]) if content_len > 0 else bytes(self._buf)
            self.sock.close()
            raise WsHandshakeError(status, hdrs, body)

        expected_accept = base64.b64encode(
            hashlib.sha1((key + self._WS_GUID).encode("ascii")).digest()
        ).decode("ascii")
        if hdrs.get("sec-websocket-accept") != expected_accept:
            self.sock.close()
            raise AssertionError(
                f"Invalid Sec-WebSocket-Accept: got {hdrs.get('sec-websocket-accept')!r}, "
                f"expected {expected_accept!r}"
            )
        self.response_headers = hdrs

    def _read_exact(self, n: int, timeout: Optional[float] = None) -> bytes:
        if timeout is not None:
            self.sock.settimeout(timeout)
        while len(self._buf) < n:
            chunk = self.sock.recv(max(4096, n - len(self._buf)))
            if not chunk:
                raise EOFError("WebSocket closed before frame completed")
            self._buf.extend(chunk)
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out

    def recv_frame(self, timeout: float = 5.0) -> Tuple[int, bytes]:
        b1, b2 = self._read_exact(2, timeout=timeout)
        opcode = b1 & 0x0F
        masked = bool(b2 & 0x80)
        length = b2 & 0x7F
        if length == 126:
            (length,) = struct.unpack("!H", self._read_exact(2, timeout=timeout))
        elif length == 127:
            (length,) = struct.unpack("!Q", self._read_exact(8, timeout=timeout))
        mask_key = self._read_exact(4, timeout=timeout) if masked else b""
        payload = bytearray(self._read_exact(length, timeout=timeout))
        if masked:
            for idx in range(length):
                payload[idx] ^= mask_key[idx % 4]
        return opcode, bytes(payload)

    def recv_json(self, timeout: float = 5.0) -> Any:
        deadline = time.monotonic() + timeout
        while True:
            rem = max(0.1, deadline - time.monotonic())
            opcode, payload = self.recv_frame(timeout=rem)
            if opcode == 0x9:  # Ping -> respond with Pong
                self.send_frame(0xA, payload)
                continue
            if opcode == 0x8:
                raise EOFError("Received WebSocket close frame")
            if opcode == 0x1:
                return json.loads(payload.decode("utf-8"))

    def send_frame(self, opcode: int, payload: bytes = b"") -> None:
        header = bytearray([0x80 | (opcode & 0x0F)])
        n = len(payload)
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", n))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", n))
        mask = os.urandom(4)
        header.extend(mask)
        masked_payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self.sock.sendall(bytes(header) + masked_payload)

    def send_text(self, text: str) -> None:
        self.send_frame(0x1, text.encode("utf-8"))

    def send_json(self, data: Any) -> None:
        self.send_text(json.dumps(data, ensure_ascii=False))

    def send_ping(self, payload: bytes = b"ping") -> None:
        self.send_frame(0x9, payload)

    def close(self) -> None:
        try:
            self.send_frame(0x8, b"")
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass

    def __enter__(self) -> "WsClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


class NativeHostClient:
    """4-byte little-endian stdio driver for Chrome Native Messaging Host."""

    def __init__(
        self,
        cmd: List[str],
        *,
        port: int,
        extra_args: Optional[List[str]] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        full_cmd = [*cmd, "--port", str(port), *(extra_args or [])]
        flags = 0x08000000 if sys.platform == "win32" else 0
        self.proc = subprocess.Popen(
            full_cmd,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            creationflags=flags,
        )
        self._stdout_queue: "queue.Queue[bytes]" = queue.Queue()
        self._reader_thread = threading.Thread(
            target=self._pump_stdout,
            daemon=True,
        )
        self._reader_thread.start()
        self._stdout_buf = bytearray()

    def _pump_stdout(self) -> None:
        assert self.proc.stdout is not None
        try:
            while True:
                chunk = self.proc.stdout.read(4096)
                if not chunk:
                    break
                self._stdout_queue.put(chunk)
        finally:
            self._stdout_queue.put(b"")

    def _read_stdout_exact(self, n: int, timeout: float = 5.0) -> bytes:
        deadline = time.monotonic() + timeout
        while len(self._stdout_buf) < n:
            rem = deadline - time.monotonic()
            if rem <= 0:
                raise TimeoutError(
                    f"Timed out reading {n} stdout bytes (buffered={len(self._stdout_buf)}: "
                    f"{bytes(self._stdout_buf[:64])!r})"
                )
            chunk = self._stdout_queue.get(timeout=rem)
            if chunk == b"":
                raise EOFError(
                    f"EOF on host stdout while expecting {n} bytes "
                    f"(buffered={bytes(self._stdout_buf)!r})"
                )
            self._stdout_buf.extend(chunk)
        out = bytes(self._stdout_buf[:n])
        del self._stdout_buf[:n]
        return out

    def send_raw(self, raw: bytes) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(raw)
        self.proc.stdin.flush()

    def send_frame(self, payload: bytes) -> None:
        self.send_raw(struct.pack("<I", len(payload)) + payload)

    def send_json(self, obj: Any) -> None:
        self.send_frame(json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def recv_frame(self, timeout: float = 5.0) -> bytes:
        hdr = self._read_stdout_exact(4, timeout=timeout)
        (msg_len,) = struct.unpack("<I", hdr)
        if msg_len == 0 or msg_len > 1_048_576:
            raise AssertionError(
                f"Invalid framed length {msg_len} from host stdout (header={hdr!r})"
            )
        return self._read_stdout_exact(msg_len, timeout=timeout)

    def recv_json(self, timeout: float = 5.0) -> Dict[str, Any]:
        raw = self.recv_frame(timeout=timeout)
        if raw.startswith(b"\xef\xbb\xbf"):
            raise AssertionError("Unexpected UTF-8 BOM in Native Messaging frame")
        parsed = json.loads(raw.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise AssertionError(f"Expected JSON object frame, got {type(parsed).__name__}")
        return parsed

    def close_and_wait(self, timeout: float = 5.0) -> Tuple[int, bytes, bytes]:
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        rc = self.proc.wait(timeout=timeout)
        self._reader_thread.join(timeout=2.0)
        while not self._stdout_queue.empty():
            chunk = self._stdout_queue.get_nowait()
            if chunk:
                self._stdout_buf.extend(chunk)
        stderr_bytes = b""
        if self.proc.stderr is not None:
            try:
                stderr_bytes = self.proc.stderr.read()
            except Exception:
                pass
            try:
                self.proc.stderr.close()
            except Exception:
                pass
        if self.proc.stdout is not None:
            try:
                self.proc.stdout.close()
            except Exception:
                pass
        return int(rc), bytes(self._stdout_buf), stderr_bytes

    def force_close(self) -> None:
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
        if self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2.0)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass

    def __enter__(self) -> "NativeHostClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.force_close()
