#!/usr/bin/env python3
"""Offline sealed-box export for one approved GitHub Actions destination.

This utility never retrieves secrets, calls GitHub, decrypts data or installs it.
Only the export step receives the fixed allowlist through its environment.
"""
from __future__ import annotations
import argparse
import base64
import binascii
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import sys
from typing import Mapping

SOURCE = 'Siyuan-Zhu/readme-live-sandbox'
TARGET = 'Siyuan-Zhu/Siyuan-Zhu'
REQUIRED = ('APIFY_TOKEN', 'GEMINI_API_KEY', 'BFL_API_KEY', 'UNSPLASH_ACCESS_KEY')
OPTIONAL = ('UNSPLASH_APP_ID', 'UNSPLASH_SECRET_KEY')


class ExportDenied(RuntimeError):
    """Messages are fixed strings; never embed configuration or secret values."""


def build_export(env: Mapping[str, str], *, include_secrets=()):
    """Read only fixed environment keys and return ciphertext plus public binding."""
    if env.get('PRODUCTION_SECRET_EXPORT_ENABLED') != 'true':
        raise ExportDenied('Secret export is disabled.')
    if (env.get('GITHUB_REPOSITORY') != SOURCE or
        env.get('SECRET_EXPORT_SOURCE_PRIVATE') != 'true' or
        env.get('GITHUB_EVENT_NAME') != 'workflow_dispatch'):
        raise ExportDenied('Export requires the fixed private source and manual dispatch.')
    sha = env.get('GITHUB_SHA', '')
    approved_sha = env.get('PRODUCTION_SECRET_EXPORT_APPROVED_SHA', '')
    if not re.fullmatch('[0-9a-f]{40}', sha) or not hmac.compare_digest(sha, approved_sha):
        raise ExportDenied('Source commit is not explicitly approved for export.')
    if any(env.get(name, '').lower() in ('1', 'true') for name in ('RUNNER_DEBUG', 'ACTIONS_STEP_DEBUG', 'ACTIONS_RUNNER_DEBUG')):
        raise ExportDenied('Export is unavailable with runner or step debugging enabled.')
    if env.get('PRODUCTION_SECRET_EXPORT_TARGET') != TARGET:
        raise ExportDenied('Secret export destination is not the fixed approved repository.')
    names = tuple(include_secrets)
    if len(set(names)) != len(names) or any(name not in OPTIONAL for name in names):
        raise ExportDenied('Requested export contains an unapproved or duplicate secret name.')
    key_id = env.get('PRODUCTION_ACTIONS_PUBLIC_KEY_ID', '')
    approved_id = env.get('PRODUCTION_SECRET_EXPORT_APPROVED_KEY_ID', '')
    if not re.fullmatch('[A-Za-z0-9_-]{1,128}', key_id) or not hmac.compare_digest(key_id, approved_id):
        raise ExportDenied('Destination key identifier does not match the approved identifier.')
    try:
        public_key = base64.b64decode(env.get('PRODUCTION_ACTIONS_PUBLIC_KEY', ''), validate=True)
    except (ValueError, binascii.Error):
        raise ExportDenied('Invalid destination public encryption key.') from None
    fingerprint = hashlib.sha256(public_key).hexdigest()
    approved_fingerprint = env.get('PRODUCTION_SECRET_EXPORT_APPROVED_PUBLIC_KEY_SHA256', '')
    if (len(public_key) != 32 or not re.fullmatch('[0-9a-f]{64}', approved_fingerprint)
        or not hmac.compare_digest(fingerprint, approved_fingerprint)):
        raise ExportDenied('Destination public key does not match the approved fingerprint.')
    # Validate every source before creating any output. No environment iteration,
    # alias discovery, secret hashes, raw lengths or plaintext manifests.
    values = {}
    for name in REQUIRED + names:
        value = env.get(name, '')
        if not isinstance(value, str) or not value.strip():
            raise ExportDenied('A required source secret is missing.')
        encoded = value.encode('utf-8')
        if len(encoded) > 48 * 1024 or '\0' in value:
            raise ExportDenied('Source secret is not an admissible GitHub Actions value.')
        values[name] = encoded
    from nacl.public import PublicKey, SealedBox
    box = SealedBox(PublicKey(public_key))
    encrypted = {name: base64.b64encode(box.encrypt(value)).decode('ascii')
                 for name, value in values.items()}
    return {'format_version': 1, 'encryption': 'libsodium-sealed-box',
      'target_repository': TARGET, 'key_id': key_id,
      'public_key_sha256': fingerprint, 'source_commit': sha,
      'secrets': {name: {'encrypted_value': value} for name, value in encrypted.items()}}


def write_export(output: Path, envelope):
    """Exclusive 0600 ciphertext file; no plaintext temporary file ever exists."""
    payload = (json.dumps(envelope, sort_keys=True, indent=2) + '\n').encode('utf-8')
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # Only this newly created ciphertext file is owned by the operation.
        output.unlink(missing_ok=True)
        raise


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        # argparse normally echoes unknown argument values. Keep failures static.
        self.exit(2, 'Secret export rejected: invalid command configuration.\n')


def main(argv=None):
    parser = SafeParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-secret', action='append', default=[])
    args = parser.parse_args(argv)
    try:
        envelope = build_export(os.environ, include_secrets=args.include_secret)
        write_export(args.output, envelope)
    except ExportDenied as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception:
        # No traceback or exception repr can expose runtime environment values.
        print('Secret export failed; no plaintext output was produced.', file=sys.stderr)
        return 2
    print('Encrypted export created for the approved destination.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
