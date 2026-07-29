"""极简 WebSocket 客户端，探测 Polymarket CLOB 的 market 频道是否需要认证。

不用第三方库：手写 RFC6455 握手 + 帧解析（只需要读，不需要分片/续帧的完整实现）。
"""
import base64
import json
import os
import socket
import ssl
import struct
import sys
import time

HOST = "ws-subscriptions-clob.polymarket.com"
PATHS = ["/ws/market", "/ws/user", "/ws/"]


def ws_connect(host, path, timeout=12):
    raw = socket.create_connection((host, 443), timeout=timeout)
    s = ssl.create_default_context().wrap_socket(raw, server_hostname=host)
    key = base64.b64encode(os.urandom(16)).decode()
    req = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "Origin: https://polymarket.com\r\n"
        "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/120\r\n"
        "\r\n"
    )
    s.sendall(req.encode())
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = s.recv(4096)
        if not chunk:
            break
        buf += chunk
    head = buf.split(b"\r\n\r\n")[0].decode("utf-8", "replace")
    return s, head, buf.split(b"\r\n\r\n", 1)[1] if b"\r\n\r\n" in buf else b""


def send_text(s, text):
    payload = text.encode()
    hdr = bytearray([0x81])
    n = len(payload)
    mask = os.urandom(4)
    if n < 126:
        hdr.append(0x80 | n)
    elif n < 65536:
        hdr.append(0x80 | 126)
        hdr += struct.pack("!H", n)
    else:
        hdr.append(0x80 | 127)
        hdr += struct.pack("!Q", n)
    hdr += mask
    hdr += bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    s.sendall(bytes(hdr))


def read_frames(s, seconds=15, leftover=b""):
    s.settimeout(2.0)
    buf = bytearray(leftover)
    out = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            data = s.recv(65536)
            if not data:
                break
            buf += data
        except socket.timeout:
            pass
        except Exception:
            break
        while len(buf) >= 2:
            b0, b1 = buf[0], buf[1]
            opcode = b0 & 0x0F
            ln = b1 & 0x7F
            off = 2
            if ln == 126:
                if len(buf) < 4:
                    break
                ln = struct.unpack("!H", buf[2:4])[0]
                off = 4
            elif ln == 127:
                if len(buf) < 10:
                    break
                ln = struct.unpack("!Q", buf[2:10])[0]
                off = 10
            if len(buf) < off + ln:
                break
            payload = bytes(buf[off:off + ln])
            del buf[:off + ln]
            if opcode in (1, 2):
                out.append(payload.decode("utf-8", "replace"))
            elif opcode == 8:
                out.append("<CLOSE frame>")
                return out
    return out


if __name__ == "__main__":
    tok = sys.argv[1] if len(sys.argv) > 1 else None
    for path in PATHS:
        try:
            s, head, rest = ws_connect(HOST, path)
        except Exception as e:  # noqa: BLE001
            print(f"{path}: 连接失败 {e}")
            continue
        status = head.split("\r\n")[0]
        print(f"\n=== {path} -> {status}")
        if "101" not in status:
            print(head[:400])
            s.close()
            continue
        if path == "/ws/market" and tok:
            sub = {"assets_ids": [tok], "type": "market"}
            send_text(s, json.dumps(sub))
            print("已发送订阅:", json.dumps(sub)[:120])
        msgs = read_frames(s, seconds=20, leftover=rest)
        print(f"20 秒内收到 {len(msgs)} 条消息")
        for m in msgs[:3]:
            print("  ", m[:400])
        s.close()
