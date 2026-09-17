#!/usr/bin/env python3
"""Compare complete independent RNGD logs against hybrid-control or v3-control."""
import argparse
import json
from pathlib import Path
import re
import statistics
import sys

KERNELS = ('sliding_project_qkv', 'sliding_attention_output', 'decoder_feedforward')
ALLOWED = set(json.loads((Path(__file__).resolve().parent / 'candidates.json').read_text()))


def read_group(folder):
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ('.log', '.txt'))
    if len(files) < 3:
        raise ValueError(str(folder) + ': need at least 3 independent log files')
    names, sources = set(), set()
    values = {k: [] for k in KERNELS}
    for path in files:
        text = path.read_text(errors='replace')
        marker = re.findall(r'^Candidate: (\S+)\s*$', text, re.M)
        digest = re.findall(r'^Source-SHA256: ([0-9a-f]{64})\s*$', text, re.M)
        if len(marker) != 1 or marker[0] not in ALLOWED or len(digest) != 1:
            raise ValueError(str(path) + ': missing or ambiguous candidate/source marker')
        if ('-> FAIL' in text or len(re.findall(r'^all 3 tests passed\s*$', text, re.M)) != 1 or
                'remote_entrypoint.sh: all kernel tests passed' not in text or
                len(re.findall(r'-> PASS\s*$', text, re.M)) != 5):
            raise ValueError(str(path) + ': failed or incomplete original three-kernel run')
        names.add(marker[0]); sources.add(digest[0])
        for kernel in KERNELS:
            sections = re.findall(r'^==> ' + re.escape(kernel) + r'\s*\n(.*?)(?=^==>|\Z)', text, re.M | re.S)
            counts = [int(v) for section in sections for v in re.findall(r'^\s*cycles=(\d+)\s*$', section, re.M)]
            if len(sections) != 1 or len(counts) != 1 or counts[0] <= 0:
                raise ValueError(str(path) + ': require exactly one device cycle count per kernel')
            values[kernel].append(counts[0])
    if len(names) != 1 or len(sources) != 1:
        raise ValueError(str(folder) + ': mixed candidates or source versions')
    return names.pop(), values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    args = p.parse_args()
    if args.control.resolve() == args.candidate.resolve():
        raise ValueError('Control and candidate folders must differ')
    control_name, control = read_group(args.control)
    name, candidate = read_group(args.candidate)
    if control_name not in {'hybrid-control', 'v3-control'} or name == control_name:
        raise ValueError('Use hybrid-control or v3-control against a different candidate')
    print('Device cycles: ' + name + ' vs ' + control_name + '. Lower is better.')
    for kernel in KERNELS:
        a, b = statistics.median(control[kernel]), statistics.median(candidate[kernel])
        delta = (b / a - 1) * 100
        print('%s: median %g -> %g (%+.2f%%), n=%d/%d; ranges %d..%d / %d..%d' %
              (kernel, a, b, delta, len(control[kernel]), len(candidate[kernel]),
               min(control[kernel]), max(control[kernel]), min(candidate[kernel]), max(candidate[kernel])))
    print('These are per-kernel samples, not end-to-end model latency or competition score.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as exc:
        print('REJECTED: ' + str(exc), file=sys.stderr)
        sys.exit(1)
