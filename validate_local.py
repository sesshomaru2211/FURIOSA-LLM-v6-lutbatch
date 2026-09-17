#!/usr/bin/env python3
"""Host-only checks. The fake compiler below does NOT validate Rust, NPU correctness or speed."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
MLP = 'src/device/shared/mlp.rs'
ATTN = 'src/device/sliding/projection.rs'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def block(text, marker):
    begin = text.index(marker)
    opening = text.index('{', begin)
    depth, end = 1, opening + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[opening + 1:end - 1], text[end:]


def normalized_up(body, rows, tail=False):
    b = body
    if rows == 8:
        b = b.replace('.transpose::<m![L % 60 = 8 / 4], m![L % 60 = 8 % 4 # 16]>()',
                      '.transpose::<m![1], m![L % 60 = 8 # 16]>()')
        b = b.replace('.commit_trim::<m![L % 60 = 8 % 4]>()', '.commit_trim::<m![L % 60 = 8]>()')
    b = b.replace(f'L % 60 = {rows}', 'L % 60 = 4')
    b = b.replace(f'tile::<m![L % 60], {rows},', 'tile::<m![L % 60], 4,')
    return b.replace('(56)' if tail else f'({rows} * i)', '(4 * i)')


def normalized_down(body, rows, tail=False):
    b = body.replace(f'H % 120 = {rows}', 'H % 120 = 12')
    b = b.replace(f'tile::<m![H % 120], {rows},', 'tile::<m![H % 120], 12,')
    return b.replace('(112)' if tail else f'({rows} * i)', '(12 * i)')


def main():
    runner = module('runner_v6', ROOT / 'run_all.py')
    runner.verify()
    baseline = ROOT / 'source-v3'
    source_report = ROOT / 'analysis/uploaded-sources'
    for name, expected in (
        ('v3-control', '7a892a3e1ad2b110172be8eae7fda3dbb7f16f7cbdc9ef57ed5c8144b1e18389'),
        ('all3-v5', 'b2212cecb5a59bf998279157226163bea3dcf6f324ce6412b130c862be708389')):
        h = hashlib.sha256()
        for p in sorted(baseline.rglob('*')):
            if p.is_file():
                rel = p.relative_to(baseline)
                from_report = source_report / name / rel
                h.update(str(rel).encode() + b'\0')
                h.update((from_report if from_report.is_file() else p).read_bytes())
        assert h.hexdigest() == expected, 'Uploaded source checksum mismatch'
    assert sha(ROOT / 'reference/fixtures.safetensors') == 'cb288f25808150d8aa9fbf653bf1e65237e84b2a9b7d6674f6ff33ba07cb4461'
    assert (ROOT / 'variants/hybrid-control' / ATTN).read_bytes() == (source_report / 'all3-v5' / ATTN).read_bytes()
    assert not (ROOT / 'variants/hybrid-control' / MLP).exists()
    for name, files in runner.CANDIDATE_FILES.items():
        actual = {str(p.relative_to(ROOT / 'variants' / name)) for p in (ROOT / 'variants' / name).rglob('*.rs')}
        assert actual == set(files)
    fused = (source_report / 'all3-v5' / MLP).read_text()
    unchanged_begin = 'pub(crate) fn feedforward('
    unchanged_end = 'pub(crate) fn project_down('
    base_mlp = (baseline / MLP).read_text()
    unchanged = fused[fused.index(unchanged_begin):fused.index(unchanged_end)]
    assert unchanged == base_mlp[base_mlp.index(unchanged_begin):base_mlp.index(unchanged_end)]
    original_up, after_up = block(fused, '    for i in 0..PASSES {')
    original_gate, _ = block(after_up, '    for i in 0..PASSES {')
    original_down, _ = block(fused, '    for i in 0..10 {')
    for name in ('ffn-batch8x12', 'ffn-batch8x16'):
        text = (ROOT / 'variants' / name / MLP).read_text()
        assert text[text.index(unchanged_begin):text.index(unchanged_end)] == unchanged
        rest = text
        for expected in (original_up, original_gate):
            bulk, rest = block(rest, '    for i in 0..7 {')
            tail, rest = block(rest, '\n    {')
            assert normalized_up(bulk, 8) == expected
            assert normalized_up(tail, 4, tail=True) == expected
        if name.endswith('16'):
            bulk, rest = block(rest, '    for i in 0..7 {')
            tail, _ = block(rest, '\n    {')
            assert normalized_down(bulk, 16) == original_down
            assert normalized_down(tail, 8, tail=True) == original_down
        else:
            body, _ = block(rest, '    for i in 0..10 {')
            assert body == original_down
        assert '.vector_inter_slice_reduce::<DownRows' in text
        assert 'weight_packed: DmTensor<f8e4m3' not in text
    for total, size, last in ((60, 8, 4), (120, 16, 8)):
        visits = [0] * total
        blocks = [(i * size, size) for i in range(7)] + [(7 * size, last)]
        for offset, count in blocks:
            assert offset >= 0 and offset + count <= total
            # Same four-result transpose lane packing used in the existing down12/v4-up8 code.
            assert count % 4 == 0
            assert [(t * 4 + lane) for t in range(count // 4) for lane in range(4)] == list(range(count))
            for row in range(offset, offset + count):
                visits[row] += 1
        assert visits == [1] * total
    assert 8 * 240 * 4 == 16 * 120 * 4 == 7680
    assert 7680 < 8192

    analyzer = module('analyzer_v6', ROOT / 'analysis/analyze_report.py')
    assert analyzer.union_size([(0, 5), (1, 3), (4, 10), (10, 11)]) == 11
    a = analyzer.analyze(ROOT / 'analysis/uploaded-schedules/v3-control/decoder_feedforward.json')
    b = analyzer.analyze(ROOT / 'analysis/uploaded-schedules/all3-v5/decoder_feedforward.json')
    assert (a['makespan'], b['makespan']) == (504130, 523288)
    assert (len(a['lut_setups']), len(b['lut_setups'])) == (3, 40)
    assert b['instruction_count'] - a['instruction_count'] == 2 * (40 - 3) - 3 == 71
    print('PASS host checks: source/fixture identity; exact arithmetic-body retention; tile coverage; uploaded LUT dependencies.')

    # All generated data below are temporary mock fixtures, never real device artifacts.
    with tempfile.TemporaryDirectory(prefix='v6-mock-validation-') as folder:
        tmp = Path(folder)
        shutil.copytree(ROOT, tmp / 'package', ignore=shutil.ignore_patterns('__pycache__', 'runs'))
        runner.ROOT = tmp / 'package'
        runner.setup_env = lambda: dict(os.environ)
        failures = set()
        def fake_compile(cmd, cwd, env, log, quiet=False):
            assert Path(env['CARGO_TARGET_DIR']) == cwd / 'target'
            if cwd.name in failures:
                log.write_text('MOCK deliberate failure')
                raise RuntimeError('MOCK compiler error')
            if cmd[2] == 'compile':
                out = Path(cmd[cmd.index('--dump-schedule') + 1])
                out.write_text(json.dumps({'instructions': [{'lifetime': {'begin': 0, 'end': 100}}]}))
                log.write_text('MOCK schedule, not an NPU result')
            else:
                out = cwd / 'target/mock-test'
                out.parent.mkdir()
                out.write_text('MOCK not a valid executable')
                out.chmod(0o755)
                log.write_text(json.dumps({'reason': 'compiler-artifact', 'target': {'name': 'test_kernels'}, 'executable': str(out)}) + '\n')
        runner.run_logged = fake_compile
        def invoke(args):
            saved = sys.argv
            try:
                sys.argv = ['run_all.py'] + args
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    result = runner.main()
            finally:
                sys.argv = saved
            run = Path((runner.ROOT / 'LAST_RUN.txt').read_text().strip())
            return result, run, json.loads((run / 'summary.json').read_text())
        code, run, rows = invoke([])
        assert code == 0 and [r['candidate'] for r in rows] == ['hybrid-control', 'ffn-batch8x16']
        for row in rows:
            stage = Path(row['upload_directory'])
            assert {p.name for p in stage.iterdir()} == {'test_runtime', 'fixtures.safetensors', 'remote_entrypoint.sh'}
            assert row['status'] == 'BUILT_NOT_DEVICE_TESTED' and row['hardware_status'] == 'NOT_RUN'
            assert row['source_sha256'] in (stage / 'remote_entrypoint.sh').read_text()
            assert sha(stage / 'fixtures.safetensors') == sha(ROOT / 'reference/fixtures.safetensors')
        assert rows[0]['source_sha256'] != rows[1]['source_sha256']
        assert (run / 'report-v6-lutbatch.zip').is_file()
        code, run, rows = invoke(['--schedule-only'])
        assert code == 0 and not list(run.glob('upload-*'))
        failures.add('ffn-batch8x16')
        code, run, rows = invoke(['--isolate'])
        assert code == 1 and [r['status'] for r in rows] == ['BUILT_NOT_DEVICE_TESTED', 'BUILT_NOT_DEVICE_TESTED', 'FAILED']
        assert not (run / 'upload-ffn-batch8x16').exists()
        missing = run / 'logs/no-artifact.log'
        missing.write_text('{}\n')
        try:
            runner.stage_binary('bad-artifact', run, missing, run)
        except RuntimeError:
            pass
        else:
            raise AssertionError('No executable should be rejected')
        assert not (run / 'upload-bad-artifact').exists()

        compare = module('compare_v6', ROOT / 'compare_runs.py')
        logs = tmp / 'logs'
        logs.mkdir()
        text = 'Candidate: ffn-batch8x16\nSource-SHA256: ' + 'a' * 64 + '\n'
        for kernel in runner.KERNELS:
            text += '==> ' + kernel + '\n'
            text += '[MOCK check] -> PASS\n' * (3 if kernel == 'sliding_project_qkv' else 1)
            text += '    cycles=12345\n\n'
        text += 'all 3 tests passed\nremote_entrypoint.sh: all kernel tests passed\n'
        for i in range(3):
            (logs / f'run{i}.log').write_text(text)
        name, values = compare.read_group(logs)
        assert name == 'ffn-batch8x16' and all(len(v) == 3 for v in values.values())
        for bad in (text.replace('-> PASS', '-> FAIL', 1), text.replace('a' * 64, 'b' * 64),
                    text.replace('cycles=12345', '', 1), text.replace('ffn-batch8x16', 'hybrid-control'),
                    text.replace('remote_entrypoint.sh: all kernel tests passed', '')):
            (logs / 'run2.log').write_text(bad)
            try:
                compare.read_group(logs)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid/mixed logs accepted')
    print('PASS MOCK checks: clean builds, isolated failures, schedule-only, report/staging and strict log parsing.')
    print('Rust/Furiosa compilation: NOT RUN. RNGD correctness/performance: NOT RUN.')


if __name__ == '__main__':
    main()
