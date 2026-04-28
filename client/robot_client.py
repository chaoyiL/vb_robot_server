from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

from openpi_client import msgpack_numpy
from websockets.exceptions import ConnectionClosed
from websockets.datastructures import Headers
from websockets.http11 import Request, Response
from websockets.sync.server import Server, ServerConnection, serve


class RobotClient:
    """Persistent websocket bridge between robot-side control and remote policy."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        allowed_tokens: list[str] | tuple[str, ...] | set[str] | None = None,
    ):
        self.host = host
        self.port = port
        self._allowed_tokens = None if allowed_tokens is None else {str(token) for token in allowed_tokens}

        self._condition = threading.Condition()
        self._packer = msgpack_numpy.Packer()
        self._server: Server | None = None
        self._thread: threading.Thread | None = None
        self._stopped = False
        self._connected = False

        self._config: Any = None
        self._state_updates: deque[Any] = deque()

        self._latest_obs: dict[str, Any] | None = None
        self._latest_obs_seq = -1

        self._latest_action: Any = None
        self._latest_action_obs_seq = -1

    def _process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        del connection

        if self._allowed_tokens is None:
            return None

        auth_header = request.headers.get("Authorization")
        if auth_header is None:
            return Response(
                401,
                "Unauthorized",
                Headers({"WWW-Authenticate": 'Bearer realm="robot-bridge"'}),
                b"Missing Authorization header.\n",
            )

        scheme, _, token = auth_header.partition(" ")
        if scheme.lower() != "bearer" or not token or token not in self._allowed_tokens:
            return Response(
                401,
                "Unauthorized",
                Headers({"WWW-Authenticate": 'Bearer realm="robot-bridge"'}),
                b"Invalid bearer token.\n",
            )

        return None

    def _handle_connection(self, websocket: ServerConnection) -> None:
        with self._condition:
            self._connected = True
            self._condition.notify_all()

        last_sent_obs_seq = -1

        try:
            websocket.send(
                self._packer.pack(
                    {
                        "type": "hello",
                        "protocol": "robot-bridge-v1",
                    }
                )
            )

            while True:
                outbound = None
                with self._condition:
                    if self._stopped:
                        break
                    if self._latest_obs is not None and self._latest_obs_seq > last_sent_obs_seq:
                        outbound = self._latest_obs

                if outbound is not None:
                    websocket.send(self._packer.pack(outbound))
                    last_sent_obs_seq = int(outbound["obs_seq"])

                try:
                    raw_message = websocket.recv(timeout=0.05)
                except TimeoutError:
                    continue

                if isinstance(raw_message, str):
                    raise RuntimeError("Robot bridge expects binary websocket frames.")

                message = msgpack_numpy.unpackb(raw_message)
                self._handle_message(message)
        except ConnectionClosed:
            pass
        finally:
            with self._condition:
                self._connected = False
                self._latest_action = None
                self._latest_action_obs_seq = -1
                self._condition.notify_all()

    def _handle_message(self, message: dict[str, Any]) -> None:
        message_type = message.get("type")

        with self._condition:
            if message_type == "config":
                self._config = message["config"]
            elif message_type == "state":
                self._state_updates.append(message["state"])
            elif message_type == "action":
                self._latest_action = message["action"]
                self._latest_action_obs_seq = int(message["obs_seq"])
            else:
                raise ValueError(f"Unsupported websocket message type: {message_type}")

            self._condition.notify_all()

    def run(self) -> None:
        with serve(
            self._handle_connection,
            host=self.host,
            port=self.port,
            process_request=self._process_request,
            compression=None,
            max_size=None,
            # This bridge can legitimately spend long periods inside blocking
            # policy inference or user input before the next recv() call.
            # Disable websocket keepalive to avoid false timeouts on the sync API.
            ping_interval=None,
        ) as server:
            with self._condition:
                self._server = server
                self._condition.notify_all()
            server.serve_forever()

    def start_background(self, daemon: bool = True) -> threading.Thread:
        if self._thread is not None and self._thread.is_alive():
            return self._thread

        with self._condition:
            self._stopped = False

        self._thread = threading.Thread(
            target=self.run,
            name=f"RobotClient-{self.port}",
            daemon=daemon,
        )
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            server = self._server
            self._condition.notify_all()

        if server is not None:
            server.shutdown()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def wait_for_connection(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout

        with self._condition:
            while not self._connected and not self._stopped:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return self._connected

    def wait_for_config(self, timeout: float | None = None) -> Any:
        deadline = None if timeout is None else time.monotonic() + timeout

        with self._condition:
            while self._config is None and not self._stopped:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)
            return self._config

    def get_state_update(self) -> Any:
        with self._condition:
            if self._state_updates:
                return self._state_updates.popleft()
            return None

    def publish_obs(self, obs: Any) -> int:
        with self._condition:
            self._latest_obs_seq += 1
            self._latest_obs = {
                "type": "obs",
                "obs_seq": self._latest_obs_seq,
                "obs": obs,
            }
            self._condition.notify_all()
            return self._latest_obs_seq

    def wait_for_action(self, obs_seq: int, timeout: float | None = None) -> Any:
        deadline = None if timeout is None else time.monotonic() + timeout

        with self._condition:
            while not self._stopped:
                if self._latest_action is not None and self._latest_action_obs_seq >= obs_seq:
                    action = self._latest_action
                    self._latest_action = None
                    return action

                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)

        return None
