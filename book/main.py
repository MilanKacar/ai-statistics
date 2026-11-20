#!/usr/bin/env python3
"""
Statistics for AI -- main entry point.

This is the ONE command that builds the interactive site from the book's
Markdown chapters and serves it locally:

    python3 main.py

Then open the URL it prints (defaults to http://localhost:8000/).

Flags:
    --no-build      skip rebuilding the site (just serve whatever is already in book/site/)
    --port N        serve on a specific port (default: 8000; auto-picks the next free one if busy)
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
            if s.connect_ex(("localhost", port)) != 0:
                return port
        port += 1
    return preferred


def serve(port, open_browser):
    if not os.path.isdir(SITE_DIR):
        print(f"No built site found at {SITE_DIR}. Run again without --no-build.")
        sys.exit(1)

    os.chdir(SITE_DIR)
    handler = http.server.SimpleHTTPRequestHandler
    free_port = find_free_port(port)
    if free_port != port:
        print(f"Port {port} is busy -- using {free_port} instead.")

    with socketserver.TCPServer(("", free_port), handler) as httpd:
        url = f"http://localhost:{free_port}/"
        print(f"Serving the book at {url}")
        print("Press Ctrl+C to stop.\n")
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


def main():
    parser = argparse.ArgumentParser(description="Build and serve the Statistics for AI book.")
    parser.add_argument("--no-build", action="store_true", help="skip rebuilding the site")
    parser.add_argument("--port", type=int, default=8000, help="port to serve on (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="don't auto-open a browser")
    args = parser.parse_args()

    check_requirements()
    if not args.no_build:
        build_site()
    serve(args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
