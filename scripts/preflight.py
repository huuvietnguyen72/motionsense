"""Read-only installation checks; no downloads, training, Node or DB startup."""

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def utf8_env() -> dict:
    return dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')


def utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')


def venv_python(root: Path) -> Path:
    return root.resolve() / '.venv' / 'Scripts' / 'python.exe'


def locked_versions(root: Path, filename: str = 'requirements.lock.txt') -> dict:
    text = (root / filename).read_text(encoding='utf-8')
    versions = dict(re.findall(r'^([\w.-]+)==([^\s\\]+)', text, re.MULTILINE))
    if not versions or any(not re.search(r'--hash=sha256:[0-9a-f]{64}(?:\s|$)', block) for block in
                           re.split(r'(?m)^[\w.-]+==', text)[1:]):
        raise ValueError('Runtime lock phải có phiên bản chính xác và SHA-256.')
    return versions


def inspect_environment(root: Path) -> dict:
    command = [str(venv_python(root)), '-X', 'utf8', '-c',
               ('import sys,sysconfig,platform,json,importlib.metadata as m; '
               'print(json.dumps(dict(implementation=platform.python_implementation(),'
               'version=list(sys.version_info[:2]),platform=sysconfig.get_platform(),'
               'prefix=sys.prefix,base_prefix=sys.base_prefix,'
               'packages={d.metadata["Name"].lower().replace("_","-"):d.version '
                'for d in m.distributions()})))')]
    result = subprocess.run(command, capture_output=True, encoding='utf-8',
                            env=utf8_env(), cwd=root, timeout=30, check=True)
    return json.loads(result.stdout)


def valid_interpreter(info: dict, root: Path) -> bool:
    return (info['implementation'] == 'CPython' and info['version'] == [3, 13]
            and info['platform'] == 'win-amd64'
            and Path(info['prefix']).resolve() == (root / '.venv').resolve()
            and info['prefix'] != info['base_prefix'])


def environment_ready(root: Path) -> bool:
    try:
        info = inspect_environment(root)
        return valid_interpreter(info, root) and all(
            info['packages'].get(name.lower().replace('_', '-')) == version
            for name, version in locked_versions(root).items())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return False


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build_inputs(root: Path) -> dict:
    frontend = root / 'frontend'
    files = [p for p in frontend.iterdir() if p.is_file() and
             (p.suffix in {'.json', '.ts', '.html'} or p.name == 'package-lock.json')]
    for folder in ('src', 'public'):
        if (frontend / folder).is_dir():
            files.extend(p for p in (frontend / folder).rglob('*') if p.is_file())
    return {p.relative_to(frontend).as_posix(): digest(p) for p in sorted(files)}


def build_outputs(root: Path) -> dict:
    dist = root / 'frontend' / 'dist'
    return {p.relative_to(dist).as_posix(): digest(p) for p in sorted(dist.rglob('*'))
            if p.is_file() and p.name not in {'build-manifest.json', '.build-manifest.tmp'}}


def write_build_manifest(root: Path) -> None:
    dist = root / 'frontend' / 'dist'
    manifest = {'schema_version': 1, 'inputs': build_inputs(root), 'outputs': build_outputs(root)}
    if 'index.html' not in manifest['outputs'] or not any(
            name.startswith('assets/') for name in manifest['outputs']):
        raise ValueError('Build chưa có index.html và assets.')
    temporary = dist / '.build-manifest.tmp'
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(dist / 'build-manifest.json')


def _unique_manifest_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Build manifest có khóa trùng.')
        result[key] = value
    return result


def _hash_mapping(value) -> bool:
    if not isinstance(value, dict):
        return False
    return all(isinstance(name, str) and name
               and '\\' not in name and ':' not in name
               and all(part not in {'', '.', '..'} for part in name.split('/'))
               and isinstance(checksum, str)
               and re.fullmatch(r'[0-9a-f]{64}', checksum) is not None
               for name, checksum in value.items())


def frontend_ready(root: Path) -> bool:
    try:
        manifest = json.loads((root / 'frontend/dist/build-manifest.json').read_text('utf-8'),
                              object_pairs_hook=_unique_manifest_object)
        if (not isinstance(manifest, dict)
                or set(manifest) != {'schema_version', 'inputs', 'outputs'}
                or type(manifest['schema_version']) is not int
                or manifest['schema_version'] != 1
                or not _hash_mapping(manifest['inputs'])
                or not _hash_mapping(manifest['outputs'])):
            return False
        return ({'package-lock.json', 'package.json', 'index.html'} <= manifest['inputs'].keys()
                and manifest['inputs'] == build_inputs(root)
                and 'index.html' in manifest['outputs']
                and any(name.startswith('assets/') for name in manifest['outputs'])
                and manifest['outputs'] == build_outputs(root))
    except (OSError, ValueError, KeyError, TypeError, RecursionError):
        return False


def runtime_state(root: Path) -> dict:
    from motionsense_app.data.store import DatasetStore
    from motionsense_app.models.registry import ModelRegistry
    from motionsense_app.settings import Settings

    settings = Settings.default(root)
    info = DatasetStore(settings.data_dir).info()
    return {'dataset': bool(info['ready']), 'models': bool(
        info['ready'] and ModelRegistry(settings.model_dir).has_ready(info['dataset_id']))}


def check_installation(root: Path) -> dict:
    root = root.resolve()
    missing = []
    environment = environment_ready(root)
    if not environment:
        missing.append('environment')
    if not frontend_ready(root):
        missing.append('frontend')
    state = {'dataset': False, 'models': False}
    if environment:
        try:
            result = subprocess.run(
                [str(venv_python(root)), '-X', 'utf8', '-c',
                 ('import json,sys; from pathlib import Path; '
                  'from scripts.preflight import runtime_state; '
                  'print(json.dumps(runtime_state(Path(sys.argv[1]))))'), str(root)],
                cwd=root, env=utf8_env(), capture_output=True, encoding='utf-8',
                timeout=30, check=True,
            )
            state = json.loads(result.stdout)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    missing.extend(name for name in ('dataset', 'models') if not state[name])
    return {'ready': not missing, 'missing': missing}
