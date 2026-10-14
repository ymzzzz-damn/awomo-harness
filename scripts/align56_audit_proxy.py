#!/usr/bin/env python3
"""Loopback-only Responses relay for this bounded simulation experiment.

Uses the existing encrypted gpt6 credential without exporting it. Stores no
headers or credentials; logs requested/returned model and raw model replies.
It rejects other models and API paths, and does not alter prompts or actions.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

from cryptography.fernet import Fernet
from fastapi import FastAPI, Request
from fastapi.responses import Response
import httpx
import uvicorn

MODEL = 'gpt-5.6-sol'
SETTINGS = Path('/home/ymz/.awomopaw.secret/providers/custom/gpt6.json')


def credential():
    settings = json.loads(SETTINGS.read_text())
    key = settings['api_key']
    if key.startswith('ENC:'):
        master = bytes.fromhex((SETTINGS.parents[2] / '.master_key').read_text().strip())
        key = Fernet(base64.urlsafe_b64encode(master)).decrypt(key[4:].encode()).decode()
    return settings['base_url'].rstrip('/'), key


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def probe(output):
    base, key = credential()
    body = {'model': MODEL, 'input': 'Call ping once.',
            'tools': [{'type': 'function', 'name': 'ping', 'description': 'Connectivity probe only',
                       'parameters': {'type': 'object', 'properties': {}, 'required': []}}],
            'tool_choice': {'type': 'function', 'name': 'ping'}, 'max_output_tokens': 256}
    request = urllib.request.Request(base + '/responses', data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            reply = json.load(response)
        save(output, {'requested_model': MODEL, 'response': reply, 'base_url': base})
        calls = [x.get('name') for x in reply.get('output', []) if x.get('type') == 'function_call']
        print(json.dumps({'requested_model': MODEL, 'returned_model': reply.get('model'),
                          'function_calls': calls, 'response_id': reply.get('id')}), flush=True)
        if not str(reply.get('model', '')).startswith('gpt-5.6') or calls != ['ping']:
            raise RuntimeError('Gateway did not return GPT-5.6 with the expected tool call')
    except urllib.error.HTTPError as error:
        save(output, {'requested_model': MODEL, 'status': error.code,
                      'error_body': error.read().decode(errors='replace')})
        raise RuntimeError(f'GPT-5.6 gateway probe failed: HTTP {error.code}') from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    if args.probe:
        return probe(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    base, key = credential()
    app = FastAPI()
    counter = 0

    @app.get('/health')
    async def health():
        return {'ready': True, 'allowed_model': MODEL, 'upstream': base, 'requests': counter}

    @app.post('/v1/responses')
    async def relay(request: Request):
        nonlocal counter
        raw = await request.body()
        body = json.loads(raw)
        if body.get('model') != MODEL or body.get('stream'):
            return Response('Only non-streaming GPT-5.6 evaluation requests allowed', status_code=400)
        if request.headers.get('authorization') != 'Bearer local-align56-audit':
            return Response('Unauthorized local experiment client', status_code=401)
        counter += 1
        sequence = counter
        context_file = args.output.parent / 'audit-context.json'
        context = json.loads(context_file.read_text()) if context_file.exists() else {}
        started = datetime.now(timezone.utc).isoformat()
        try:
            async with httpx.AsyncClient(timeout=240) as client:
                response = await client.post(base + '/responses', content=raw,
                    headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
        except httpx.HTTPError as error:
            save(args.output / f'{sequence:05d}.json', {'sequence': sequence, 'context': context,
                'started': started, 'ended': datetime.now(timezone.utc).isoformat(),
                'status': 502, 'requested_model': body['model'], 'returned_model': None,
                'request_sha256': hashlib.sha256(raw).hexdigest(),
                'transport_error_type': type(error).__name__})
            return Response('Upstream transport error', status_code=502)
        try:
            reply = response.json()
        except ValueError:
            reply = {'parse_error': True, 'body': response.text}
        record = {'sequence': sequence, 'context': context, 'started': started,
                  'ended': datetime.now(timezone.utc).isoformat(), 'status': response.status_code,
                  'requested_model': body['model'], 'returned_model': reply.get('model'),
                  'request_sha256': hashlib.sha256(raw).hexdigest(), 'response': reply}
        save(args.output / f'{sequence:05d}.json', record)
        print(json.dumps({k: record[k] for k in ['sequence', 'context', 'status',
                         'requested_model', 'returned_model']}), flush=True)
        return Response(response.content, status_code=response.status_code,
                        media_type='application/json')

    uvicorn.run(app, host='127.0.0.1', port=8101, access_log=False)


if __name__ == '__main__':
    main()
