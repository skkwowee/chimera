"""
CS2 Network Console (netcon) TCP client.

Connects to CS2's network console exposed via the -netconport launch option.
Sends console commands as newline-terminated text over a raw TCP socket.

Usage:
    from cs2_tools.netcon import CS2Netcon

    con = CS2Netcon(port=2121)
    con.connect()
    con.playdemo("furia-vs-vitality-m1-mirage")
    con.goto_tick(5000)
    con.spec_player("ZywOo")
    con.screenshot("mirage_r01_t005000_ZywOo")
    con.disconnect()

Requires CS2 launched with: -netconport 2121 -console
"""

import socket
import time


class CS2Netcon:
    """TCP client for CS2's network console."""

    def __init__(self, host: str = "127.0.0.1", port: int = 2121, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: socket.socket | None = None

    def connect(self) -> None:
        """Connect to CS2 netcon. Retries up to 3 times."""
        for attempt in range(3):
            try:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._sock.settimeout(self.timeout)
                self._sock.connect((self.host, self.port))
                # Drain any welcome banner
                try:
                    self._sock.recv(4096)
                except socket.timeout:
                    pass
                print(f"Connected to CS2 netcon at {self.host}:{self.port}")
                return
            except (ConnectionRefusedError, OSError) as e:
                if self._sock:
                    self._sock.close()
                    self._sock = None
                if attempt < 2:
                    wait = 2 ** attempt
                    print(f"Connection failed ({e}), retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise ConnectionError(
                        f"Could not connect to CS2 netcon at {self.host}:{self.port}. "
                        f"Is CS2 running with -netconport {self.port}?"
                    ) from e

    def disconnect(self) -> None:
        """Close the TCP connection."""
        if self._sock:
            self._sock.close()
            self._sock = None

    def send(self, command: str) -> None:
        """Send a console command."""
        if not self._sock:
            raise ConnectionError("Not connected. Call connect() first.")
        try:
            self._sock.sendall((command + "\n").encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError):
            print(f"Connection lost, reconnecting...")
            self.connect()
            self._sock.sendall((command + "\n").encode("utf-8"))

    def send_and_wait(self, command: str, settle: float = 0.0) -> None:
        """Send a command and wait for rendering/processing to settle."""
        self.send(command)
        if settle > 0:
            time.sleep(settle)

    # -- CS2 demo helpers --

    def playdemo(self, demo_path: str, load_wait: float = 10.0) -> None:
        """Load a demo file. Waits for it to finish loading."""
        self.send_and_wait(f"playdemo {demo_path}", settle=load_wait)

    def goto_tick(self, tick: int, settle: float = 1.0) -> None:
        """Jump to a specific demo tick and pause.

        The third arg (1) tells CS2 to pause after reaching the tick.
        """
        self.send_and_wait(f"demo_gototick {tick} 0 1", settle=settle)

    def spec_player(self, player_name: str, settle: float = 0.5) -> None:
        """Spectate a specific player by name."""
        self.send_and_wait(f"spec_player {player_name}", settle=settle)

    def screenshot(self, name: str, settle: float = 0.2) -> None:
        """Capture a JPEG screenshot.

        CS2's jpeg command writes to the game's screenshots directory.
        The name becomes the filename (without extension).
        """
        self.send_and_wait(f"jpeg {name}", settle=settle)

    def pause(self) -> None:
        """Pause demo playback."""
        self.send("demo_pause")

    def resume(self) -> None:
        """Resume demo playback."""
        self.send("demo_resume")

    def exec_cfg(self, commands: list[str]) -> None:
        """Send a batch of console commands for initial setup."""
        for cmd in commands:
            self.send(cmd)
            time.sleep(0.1)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()
