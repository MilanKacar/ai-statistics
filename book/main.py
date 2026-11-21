#!/usr/bin/env python3
"""
Statistics for AI -- main entry point.

This is the ONE command that builds the interactive site from the book's
Markdown chapters and serves it locally:

    python3 main.py

Then open the URL it prints (defaults to http://localhost:8001/).

Flags:
    --no-build      skip rebuilding the site (just serve whatever is already in book/site/)
    --port N        serve on a specific port (default: 8001; auto-picks the next free one if busy)
    --no-browser    don't automatically open your default web browser
"""
import argparse
import http.server
import os
import socket
import socketserver
import subprocess
import sys
import webbrowser

BOOK_DIR = os.path.dirname(os.path.abspath(__file__))
SITE_DIR = os.path.join(BOOK_DIR, "site")
BUILD_SCRIPT = os.path.join(BOOK_DIR, "code", "_build_site.py")
REQUIREMENTS = os.path.join(os.path.dirname(BOOK_DIR), "requirements.txt")


def check_requirements():
    try:
        import markdown  # noqa: F401
    except ImportError:
        print("Missing required package: markdown")
        print()
        print("Set up (or activate) your virtual environment first, e.g.:")
        print(f"  source {os.path.join(os.path.dirname(BOOK_DIR), '.venv', 'bin', 'activate')}")
        print(f"  pip install -r \"{REQUIREMENTS}\"")
        sys.exit(1)


def build_site():
    print("Building the site from the Markdown chapters...\n")
    result = subprocess.run([sys.executable, BUILD_SCRIPT])
    if result.returncode != 0:
        print("Site build failed -- see the errors above.")
        sys.exit(1)
    print()


def find_free_port(preferred):
    port = preferred
    for _ in range(25):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    return preferred


def is_wsl():
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except Exception:
        return False


def lan_ip():
    """Best-effort LAN IP of this machine, so a phone on the same Wi-Fi can
    reach the server. Doesn't actually send packets (UDP connect just picks
    the right outbound interface and records its address)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return None
    finally:
        s.close()


def print_phone_instructions(port):
    """Explain how to reach the server from a phone, accounting for WSL."""
    ip = lan_ip()
    if is_wsl():
        print("  You're inside WSL2, so this 'IP' is NOT reachable from your phone:")
        if ip:
            print(f"      (WSL virtual IP: {ip} -- phones can't route to it directly)")
        print("  In WINDOWS PowerShell (as Administrator), forward the port once:")
        print(f"      $ip = (wsl hostname -I).Trim().Split()[0]")
        print(f"      netsh interface portproxy add v4tov4 listenport={port} listenaddress=0.0.0.0 "
              f"connectport={port} connectaddress=$ip")
        print(f"      New-NetFirewallRule -DisplayName \"WSL Book {port}\" -Direction Inbound "
              f"-LocalPort {port} -Protocol TCP -Action Allow")
        print(f"  Then, on your phone (same Wi-Fi): http://<your PC's Wi-Fi IPv4>:{port}/")
        print("      (find it with `ipconfig` in Windows -- the Wi-Fi adapter's IPv4 Address)")
    elif ip:
        print(f"  On your phone (same Wi-Fi): http://{ip}:{port}/")


def serve(port, open_browser, host="0.0.0.0"):
    if not os.path.isdir(SITE_DIR):
        print(f"No built site found at {SITE_DIR}. Run again without --no-build.")
        sys.exit(1)

    os.chdir(SITE_DIR)
    handler = http.server.SimpleHTTPRequestHandler
    free_port = find_free_port(port)
    if free_port != port:
        print(f"Port {port} is busy -- using {free_port} instead.")

    # allow_reuse_address lets us rebind immediately after Ctrl+C instead of
    # hitting "OSError: Address already in use" while the OS waits out the
    # socket's TIME_WAIT state.
    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    httpd = ReusableTCPServer((host, free_port), handler)
    try:
        url = f"http://localhost:{free_port}/"
        print(f"Serving the book at {url}")
        print_phone_instructions(free_port)
        print("Press Ctrl+C to stop.\n")
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        # close the listening socket so the port is released right away
        httpd.server_close()


def main():
    parser = argparse.ArgumentParser(description="Build and serve the Statistics for AI book.")
    parser.add_argument("--no-build", action="store_true", help="skip rebuilding the site")
    parser.add_argument("--port", type=int, default=8001, help="port to serve on (default: 8001)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="interface to bind (default: 0.0.0.0, reachable from your phone on the same Wi-Fi)")
    parser.add_argument("--no-browser", action="store_true", help="don't auto-open a browser")
    args = parser.parse_args()

    check_requirements()
    if not args.no_build:
        build_site()
    serve(args.port, open_browser=not args.no_browser, host=args.host)


if __name__ == "__main__":
    main()
