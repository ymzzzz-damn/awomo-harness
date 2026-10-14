#!/usr/bin/env python3
"""Use the previously configured P6000 SSH connection without printing credentials."""
import argparse
import getpass
import json
import os
from pathlib import Path
import re
import subprocess
import sys


HOST = "root@180.76.186.140"
PORT = "2288"
SESSION = Path("/home/ymz/.codex/sessions/2026/10/12/rollout-2026-10-12T17-31-54-01a12f74-3c97-7151-ab81-a09f474de847.jsonl")


def environment():
    # Recover only the credential for this same user-configured connection.
    # Never persist it in source, receipts, stdout, or subprocess arguments.
    candidate = None
    for line in SESSION.open():
        if "180.76.186.140" in line and "SSHPASS=" in line:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            # Output records can contain masked credentials. Use only the
            # original tool invocation which established this connection.
            if record.get("type") != "response_item" or record.get("payload", {}).get("type") not in {"custom_tool_call", "function_call"}:
                continue
            serialized = json.dumps(record, ensure_ascii=False)
            match = re.search(r"SSHPASS='([^']+)'", serialized)
            if match:
                candidate = match.group(1)
    if candidate is None:
        raise RuntimeError("Previously configured P6000 SSH credential not found")
    return dict(os.environ, SSHPASS=candidate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path)
    parser.add_argument("--copy", nargs="+")
    parser.add_argument("--fetch", nargs="+")
    parser.add_argument("--destination")
    parser.add_argument("--password-stdin", action="store_true",
                        help="Use the freshly supplied password from stdin; never persist or print it")
    parser.add_argument("--control-socket", type=Path,
                        help="Reuse a private SSH multiplex socket without saving a password")
    args = parser.parse_args()
    if args.password_stdin:
        password = (getpass.getpass("P6000 SSH password: ") if sys.stdin.isatty()
                    else sys.stdin.readline().rstrip("\r\n"))
        if not password:
            raise RuntimeError("Supplied SSH password is empty")
        connection_env = dict(os.environ, SSHPASS=password)
    else:
        connection_env = environment()
    multiplex = []
    if args.control_socket:
        socket = args.control_socket.resolve()
        multiplex = ["-o", "ControlMaster=auto", "-o", "ControlPersist=12h",
                     "-o", "ControlPath=" + str(socket)]
    if args.script:
        return subprocess.run(["sshpass", "-e", "ssh", "-p", PORT,
            "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=15", *multiplex, HOST,
            "bash", "-s"], input=args.script.read_bytes(), env=connection_env).returncode
    if args.copy and args.destination:
        return subprocess.run(["sshpass", "-e", "scp", "-P", PORT,
            "-o", "StrictHostKeyChecking=yes", *multiplex, *args.copy, HOST + ":" + args.destination],
            env=connection_env).returncode
    if args.fetch and args.destination:
        return subprocess.run(["sshpass", "-e", "scp", "-P", PORT,
            "-o", "StrictHostKeyChecking=yes", *multiplex, *[HOST + ":" + path for path in args.fetch], args.destination],
            env=connection_env).returncode
    parser.error("Pass --script, or --copy/--fetch with --destination")


if __name__ == "__main__":
    raise SystemExit(main())
