"""Synthetic-only sealed-box verification; no real environment or API access."""
import base64
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

try:
    from nacl.public import PrivateKey, SealedBox
    from nacl.exceptions import CryptoError
except ImportError:
    PrivateKey = None

from scripts import seal_production_secrets as handoff


@unittest.skipUnless(PrivateKey is not None, 'PyNaCl is installed only in the dedicated handoff test environment')
class SealedHandoffTests(unittest.TestCase):
    def setUp(self):
        self.private = PrivateKey.generate()
        public = bytes(self.private.public_key)
        self.env = {
            'PRODUCTION_SECRET_EXPORT_ENABLED': 'true',
            'GITHUB_REPOSITORY': handoff.SOURCE, 'SECRET_EXPORT_SOURCE_PRIVATE': 'true',
            'GITHUB_EVENT_NAME': 'workflow_dispatch', 'GITHUB_SHA': 'a'*40,
            'PRODUCTION_SECRET_EXPORT_APPROVED_SHA': 'a'*40,
            'PRODUCTION_SECRET_EXPORT_TARGET': handoff.TARGET,
            'PRODUCTION_ACTIONS_PUBLIC_KEY': base64.b64encode(public).decode(),
            'PRODUCTION_ACTIONS_PUBLIC_KEY_ID': 'test-key-id',
            'PRODUCTION_SECRET_EXPORT_APPROVED_KEY_ID': 'test-key-id',
            'PRODUCTION_SECRET_EXPORT_APPROVED_PUBLIC_KEY_SHA256': hashlib.sha256(public).hexdigest(),
            **{name: 'synthetic-only-test-value-' + name for name in handoff.REQUIRED},
        }

    def test_github_compatible_sealed_box_round_trip_and_wrong_recipient_rejection(self):
        envelope = handoff.build_export(self.env)
        self.assertEqual(set(envelope['secrets']), set(handoff.REQUIRED))
        for name, record in envelope['secrets'].items():
            ciphertext = base64.b64decode(record['encrypted_value'], validate=True)
            self.assertEqual(SealedBox(self.private).decrypt(ciphertext), self.env[name].encode())
            with self.assertRaises(CryptoError):
                SealedBox(PrivateKey.generate()).decrypt(ciphertext)
        encoded = json.dumps(envelope)
        for name in handoff.REQUIRED:
            self.assertNotIn(self.env[name], encoded)
        self.assertEqual(envelope['target_repository'], handoff.TARGET)
        self.assertEqual(envelope['key_id'], 'test-key-id')

    def test_optional_secrets_need_explicit_request_and_salt_is_never_read(self):
        class ReadTracked(dict):
            reads = []
            def get(self, key, default=None):
                self.reads.append(key)
                return super().get(key, default)
            def __iter__(self):
                raise AssertionError('Environment enumeration is forbidden')
        env = ReadTracked(self.env)
        for name in handoff.OPTIONAL: env[name] = 'synthetic-optional-' + name
        env['PULSE_VOTE_SALT'] = 'synthetic-salt-must-remain-private-source-only'
        base = handoff.build_export(env)
        self.assertEqual(set(base['secrets']), set(handoff.REQUIRED))
        self.assertNotIn('PULSE_VOTE_SALT', env.reads)
        self.assertNotIn('UNSPLASH_SECRET_KEY', env.reads)
        with_optional = handoff.build_export(env, include_secrets=handoff.OPTIONAL)
        self.assertEqual(set(with_optional['secrets']), set(handoff.REQUIRED + handoff.OPTIONAL))

    def test_default_disabled_and_binding_failures_are_rejected(self):
        cases = [
            ('PRODUCTION_SECRET_EXPORT_ENABLED', ''),
            ('PRODUCTION_SECRET_EXPORT_TARGET', 'someone/else'),
            ('GITHUB_REPOSITORY', handoff.TARGET),
            ('SECRET_EXPORT_SOURCE_PRIVATE', 'false'),
            ('GITHUB_EVENT_NAME', 'push'),
            ('PRODUCTION_SECRET_EXPORT_APPROVED_SHA', 'b'*40),
            ('PRODUCTION_ACTIONS_PUBLIC_KEY_ID', 'different-key'),
            ('PRODUCTION_SECRET_EXPORT_APPROVED_PUBLIC_KEY_SHA256', 'c'*64),
            ('PRODUCTION_ACTIONS_PUBLIC_KEY', base64.b64encode(b'x'*31).decode()),
            ('PRODUCTION_ACTIONS_PUBLIC_KEY', 'invalid-base64!'),
            ('RUNNER_DEBUG', '1'), ('ACTIONS_STEP_DEBUG', 'true'),
            ('GEMINI_API_KEY', ''),
        ]
        for field, value in cases:
            with self.subTest(field=field):
                env = dict(self.env); env[field] = value
                with self.assertRaises(handoff.ExportDenied):
                    handoff.build_export(env)

    def test_unknown_duplicate_and_salt_exports_fail(self):
        for names in [('PULSE_VOTE_SALT',), ('UNKNOWN',), ('UNSPLASH_APP_ID',)*2]:
            with self.subTest(names=names), self.assertRaises(handoff.ExportDenied):
                handoff.build_export(self.env, include_secrets=names)

    def test_exclusive_0600_output_contains_ciphertext_only(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'encrypted-secrets.json'
            handoff.write_export(output, handoff.build_export(self.env))
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            body = output.read_text()
            self.assertEqual(set(json.loads(body)['secrets']), set(handoff.REQUIRED))
            for name in handoff.REQUIRED: self.assertNotIn(self.env[name], body)
            with self.assertRaises(FileExistsError):
                handoff.write_export(output, handoff.build_export(self.env))

    def test_cli_success_and_failure_never_emit_source_values(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'encrypted-secrets.json'
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch.dict(handoff.os.environ, self.env, clear=True), redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(handoff.main(['--output', str(output)]), 0)
            with patch.dict(handoff.os.environ, {}, clear=True), redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(handoff.main(['--output', str(output)]), 2)
            for name in handoff.REQUIRED:
                self.assertNotIn(self.env[name], stdout.getvalue()+stderr.getvalue())
            self.assertNotIn('Traceback', stderr.getvalue())

    def test_unexpected_exception_and_bad_arguments_are_not_echoed(self):
        sentinel = 'synthetic-never-echo-this-value'
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr), patch.object(handoff, 'build_export', side_effect=RuntimeError(sentinel)):
            self.assertEqual(handoff.main(['--output', '/unused']), 2)
        with redirect_stderr(stderr), self.assertRaises(SystemExit):
            handoff.main(['--unexpected', sentinel])
        self.assertNotIn(sentinel, stdout.getvalue()+stderr.getvalue())
        self.assertNotIn('Traceback', stderr.getvalue())


class WorkflowBoundaryTests(unittest.TestCase):
    def test_manual_private_default_denied_ciphertext_only_workflow(self):
        import re
        path = Path(__file__).resolve().parents[1]/'.github/workflows/sealed_production_secret_export.yml'
        workflow = path.read_text()
        self.assertIn('on:\n  workflow_dispatch:', workflow)
        self.assertNotRegex(workflow, r'(?m)^  (push|pull_request|schedule|workflow_call):')
        self.assertIn('permissions:\n  contents: read\n', workflow)
        self.assertNotIn('id-token:', workflow)
        for gate in ("vars.PRODUCTION_SECRET_EXPORT_ENABLED == 'true'", 'github.event.repository.private == true',
                     'vars.PRODUCTION_SECRET_EXPORT_APPROVED_SHA == github.sha'):
            self.assertIn(gate, workflow)
        self.assertNotIn('PULSE_VOTE_SALT', workflow)
        self.assertIn('retention-days: 1', workflow)
        self.assertIn('path: ' + chr(36) + '{{ runner.temp }}/sealed-provider-handoff/encrypted-secrets.json', workflow)
        for action in re.findall(r'uses: (\S+)', workflow):
            self.assertIsNotNone(re.fullmatch(r'[^@]+@[0-9a-f]{40}', action))
        self.assertNotIn('secrets: inherit', workflow)

    def test_python_workflow_orchestration_keeps_private_paths_and_explicit_optional_scope(self):
        import os
        import re
        import subprocess
        import sys
        import textwrap
        workflow = (Path(__file__).resolve().parents[1]/'.github/workflows/sealed_production_secret_export.yml').read_text()
        bodies = re.findall(r'        shell: python\n        run: \|\n((?:          [^\n]*\n)+)', workflow)
        self.assertEqual(len(bodies), 3)
        bodies = [textwrap.dedent(body) for body in bodies]
        with patch.dict(os.environ, {'RUNNER_DEBUG': '1'}, clear=True), self.assertRaises(SystemExit):
            exec(compile(bodies[0], '<workflow-debug>', 'exec'), {})
        for include in ('false', 'true'):
            with self.subTest(include=include), tempfile.TemporaryDirectory() as directory:
                env = {'RUNNER_TEMP': directory, 'INCLUDE_UNSPLASH_OAUTH': include}
                previous_umask = os.umask(0o077)
                try:
                    with patch.dict(os.environ, env, clear=True), patch.object(subprocess, 'run') as run:
                        exec(compile(bodies[1], '<workflow-create>', 'exec'), {})
                        folder = Path(directory)/'sealed-provider-handoff'
                        self.assertEqual(stat.S_IMODE(folder.stat().st_mode), 0o700)
                        output = folder/'encrypted-secrets.json'
                        expected = [sys.executable, 'scripts/seal_production_secrets.py', '--output', str(output)]
                        if include == 'true':
                            expected += ['--include-secret', 'UNSPLASH_APP_ID', '--include-secret', 'UNSPLASH_SECRET_KEY']
                        run.assert_called_once_with(expected, check=True)
                        # Existing private directories cannot silently reuse an old handoff.
                        with self.assertRaises(FileExistsError):
                            exec(compile(bodies[1], '<workflow-repeat>', 'exec'), {})
                        self.assertEqual(run.call_count, 1)
                        output.write_text('synthetic ciphertext fixture')
                        exec(compile(bodies[2], '<workflow-cleanup>', 'exec'), {})
                        self.assertFalse(output.exists())
                finally:
                    os.umask(previous_umask)



if __name__ == '__main__':
    unittest.main()
