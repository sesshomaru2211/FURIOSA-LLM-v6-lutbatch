#!/usr/bin/env python3
"""Build a hybrid control and LUT-batching FFN experiments; no hardware speedup is assumed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
CANDIDATE_FILES = json.loads((ROOT / 'candidates.json').read_text())
CANDIDATES = tuple(CANDIDATE_FILES)
KERNELS = ('sliding_project_qkv', 'sliding_attention_output', 'decoder_feedforward')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for rel, expected in manifest.items():
        path = ROOT / rel
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError('Package changed or incomplete: ' + rel)
    print('Package verified: original tests, fixture and lockfile preserved.', flush=True)


def setup_env():
    env = os.environ.copy()
    home = Path.home()
    compiler = Path(env.get('FURIOSA_COMPILER_DIR', str(home / 'furiosa-tools/compiler-0.6.0/cargo-furiosa-opt-v0.6.0-x86_64-unknown-linux-gnu')))
    env['PATH'] = os.pathsep.join((str(compiler), str(home / '.cargo/bin'), env.get('PATH', '')))
    for key in ('FURIOSA_OPT_OUT_DIR', 'CARGO_TARGET_DIR', 'RUSTC', 'RUSTC_WRAPPER',
                'RUSTC_WORKSPACE_WRAPPER', 'RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS',
                'CARGO_BUILD_TARGET'):
        env.pop(key, None)
    env['RUSTUP_TOOLCHAIN'] = 'nightly-2026-05-01'
    if not shutil.which('cargo', path=env['PATH']) or not shutil.which('rustc', path=env['PATH']):
        raise RuntimeError('Cargo/Rust not found. Run in the Ubuntu environment used for v3.')
    sysroot = subprocess.check_output(['rustc', '--print', 'sysroot'], env=env, text=True).strip()
    env['LD_LIBRARY_PATH'] = str(Path(sysroot) / 'lib') + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    version = subprocess.check_output(['cargo', 'furiosa-opt', '--version'], env=env, text=True).strip()
    if version != 'cargo-furiosa-opt 0.6.0':
        raise RuntimeError('Expected cargo-furiosa-opt 0.6.0; got: ' + version)
    print(version, flush=True)
    return env


def run_logged(cmd, cwd, env, log, quiet=False):
    if quiet:
        # Cargo JSON stdout and diagnostic stderr must remain separate.
        with log.open('w') as output, log.with_suffix('.stderr.log').open('w') as errors:
            result = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=output, stderr=errors)
            while True:
                try:
                    result.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    print('Still building ' + cwd.name + '; logs: ' + str(log.parent), flush=True)
        if result.returncode:
            print(log.with_suffix('.stderr.log').read_text(errors='replace')[-8000:], flush=True)
    else:
        with log.open('w') as output:
            process = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                output.write(line)
                print(line, end='', flush=True)
            result = process
            result.wait()
    if result.returncode:
        raise RuntimeError('Command failed (exit %s); see %s' % (result.returncode, log))


def schedule_stats(path):
    d = json.loads(path.read_text())
    instructions = d['instructions']
    if not instructions:
        raise RuntimeError('Empty schedule: ' + str(path))
    return {'makespan': max(x['lifetime']['end'] for x in instructions),
            'instructions': len(instructions)}


def stage_binary(name, project, log, run_root):
    executables = set()
    for line in log.read_text().splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if (item.get('reason') == 'compiler-artifact' and
                item.get('target', {}).get('name') == 'test_kernels' and item.get('executable')):
            path = Path(item['executable'])
            executables.add(path if path.is_absolute() else project / path)
    if len(executables) != 1:
        raise RuntimeError('Expected exactly one newly built test_kernels executable')
    binary = executables.pop()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError('Built executable is missing or not executable: ' + str(binary))
    stage = run_root / ('upload-' + name)
    stage.mkdir()
    shutil.copy2(binary, stage / 'test_runtime')
    shutil.copy2(ROOT / 'reference/fixtures.safetensors', stage / 'fixtures.safetensors')
    wrapper = (ROOT / 'reference/remote_entrypoint.sh').read_text()
    first, rest = wrapper.split('\n', 1)
    (stage / 'remote_entrypoint.sh').write_text(first + '\n' +
        'echo "Candidate: ' + name + '"\n' +
        'echo "Source-SHA256: ' + source_digest(project) + '"\n' + rest)
    for p in stage.iterdir():
        p.chmod(0o755 if p.name != 'fixtures.safetensors' else 0o644)
    assert sha(stage / 'fixtures.safetensors') == sha(ROOT / 'reference/fixtures.safetensors')
    return stage


def source_digest(project):
    # Hash path + bytes for all source, tests and dependency pins.
    h = hashlib.sha256()
    for rel in sorted(p.relative_to(ROOT / 'source-v3') for p in (ROOT / 'source-v3').rglob('*') if p.is_file()):
        h.update(str(rel).encode() + b'\0')
        h.update((project / rel).read_bytes())
    return h.hexdigest()


def write_report(run_root, rows):
    control = next((r.get('schedules', {}) for r in rows if r['candidate'] == 'hybrid-control'), {})
    lines = ['V6 LUT-batching experiments — STATIC SCHEDULES, not device measurements.',
             'Control: QKV v3 + attention v5 + FFN v3.', '']
    for row in rows:
        lines.append(row['candidate'] + ': ' + row['status'])
        for kernel, stat in row['schedules'].items():
            line = '  %s: %s cycles, %s instructions' % (kernel, stat['makespan'], stat['instructions'])
            if kernel in control:
                delta = (stat['makespan'] / control[kernel]['makespan'] - 1) * 100
                stat['delta_percent_vs_control'] = delta
                line += ' (%+.2f%% vs control)' % delta
            lines.append(line)
        if 'upload_directory' in row:
            lines.append('  Upload 3 files from: ' + row['upload_directory'])
        if row['candidate'].startswith('ffn-batch') and 'decoder_feedforward' in row['schedules']:
            ffn = row['schedules']['decoder_feedforward']['makespan']
            reference = control.get('decoder_feedforward', {}).get('makespan', 504130)
            if ffn >= reference:
                lines.append('  CAUTION: FFN static schedule did not improve; send report before spending RNGD runs.')
        if 'error' in row:
            lines.append('  ERROR: ' + row['error'])
    lines += ['', 'These results do not establish device correctness or speedup.',
              'RNGD: remote_entrypoint.sh | timeout 70 | Args empty | TUC_PROFILE_LEVEL=info',
              'Compare at least 3 complete PASS logs per candidate, against hybrid-control.',
              'If batch8x16 fails, optional fallback: bash run_all.sh --candidate ffn-batch8x12',
              'Keep hybrid-control unless the candidate passes the original tests and improves device medians.']
    (run_root / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    (run_root / 'summary.txt').write_text('\n'.join(lines) + '\n')
    report = run_root / 'report-v6-lutbatch.zip'
    with zipfile.ZipFile(report, 'w', zipfile.ZIP_DEFLATED) as z:
        for filename in ('summary.json', 'summary.txt'):
            z.write(run_root / filename, filename)
        for folder in ('logs', 'schedules', 'sources'):
            for p in (run_root / folder).rglob('*'):
                if p.is_file():
                    z.write(p, p.relative_to(run_root))
    print('\n' + '\n'.join(lines), flush=True)
    print('Send this report back: ' + str(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--schedule-only', action='store_true')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--candidate', choices=CANDIDATES, action='append')
    selection.add_argument('--isolate', action='store_true', help='Also test the up/gate-only batching fallback')
    args = parser.parse_args()
    verify()
    if args.verify_only:
        return 0
    env = setup_env()
    selected = args.candidate or (['hybrid-control', 'ffn-batch8x12', 'ffn-batch8x16'] if args.isolate else ['hybrid-control', 'ffn-batch8x16'])
    output = ROOT / 'runs'
    output.mkdir(exist_ok=True)
    run_root = Path(tempfile.mkdtemp(prefix='run-', dir=output))
    (ROOT / 'LAST_RUN.txt').write_text(str(run_root) + '\n')
    for folder in ('logs', 'schedules', 'sources'):
        (run_root / folder).mkdir()
    rows = []
    for name in selected:
        row = {'candidate': name, 'status': 'STARTED', 'hardware_status': 'NOT_RUN', 'schedules': {}}
        rows.append(row)
        print('\n=== ' + name + ' ===', flush=True)
        try:
            project = run_root / 'projects' / name
            shutil.copytree(ROOT / 'source-v3', project)
            for p in project.rglob('*'):
                p.chmod(0o755 if p.is_dir() else 0o644)
            for rel in CANDIDATE_FILES[name]:
                shutil.copyfile(ROOT / 'variants' / name / rel, project / rel)
            row['source_sha256'] = source_digest(project)
            for rel in ('src/device/sliding/rope.rs', 'src/device/sliding/projection.rs', 'src/device/shared/mlp.rs'):
                dst = run_root / 'sources' / name / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(project / rel, dst)
            local_env = env.copy()
            local_env['CARGO_TARGET_DIR'] = str(project / 'target')
            schedule_dir = run_root / 'schedules' / name
            schedule_dir.mkdir()
            for kernel in KERNELS:
                print('Compiling schedule: ' + kernel, flush=True)
                schedule = schedule_dir / (kernel + '.json')
                run_logged(['cargo', 'furiosa-opt', 'compile', 'ops::' + kernel, '--exact',
                            '--dump-schedule', str(schedule)], project, local_env,
                           run_root / 'logs' / (name + '.' + kernel + '.compile.log'))
                row['schedules'][kernel] = schedule_stats(schedule)
            row['status'] = 'SCHEDULE_OK'
            if not args.schedule_only:
                log = run_root / 'logs' / (name + '.build.jsonl')
                run_logged(['cargo', 'furiosa-opt', 'test', '--locked', '--release', '--test',
                            'test_kernels', '--no-run', '--message-format=json'],
                           project, local_env, log, quiet=True)
                stage = stage_binary(name, project, log, run_root)
                row['upload_directory'] = str(stage)
                row['upload_sha256'] = {p.name: sha(p) for p in stage.iterdir()}
                row['status'] = 'BUILT_NOT_DEVICE_TESTED'
                print('READY FOR RNGD: ' + str(stage), flush=True)
        except (RuntimeError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
            row['status'] = 'FAILED'
            row['error'] = str(exc)
            print('FAILED: ' + str(exc), file=sys.stderr, flush=True)
        (run_root / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    write_report(run_root, rows)
    if shutil.which('wslpath'):
        try:
            windows_path = subprocess.check_output(['wslpath', '-w', str(run_root)], text=True).strip()
            print('Paste this path into Windows File Explorer (Win+E): ' + windows_path, flush=True)
        except (OSError, subprocess.CalledProcessError):
            pass
    return 1 if any(r['status'] == 'FAILED' for r in rows) else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        print('ERROR: ' + str(exc), file=sys.stderr)
        sys.exit(1)
