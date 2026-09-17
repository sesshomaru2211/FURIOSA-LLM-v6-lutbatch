#!/usr/bin/env python3
"""Analyze uploaded schedules; all durations are compiler estimates, not RNGD times."""
import argparse
import collections
import json
from pathlib import Path


def union_size(intervals):
    end = None
    total = 0
    for a, b in sorted(intervals):
        if b <= a:
            continue
        total += b - max(a, end if end is not None else a) if end is None or b > end else 0
        end = max(b, end if end is not None else b)
    return total


def analyze(path):
    data = json.loads(path.read_text())
    instructions = data['instructions']
    tensors = {int(t['index']): t for t in data['tensors']}
    active = [i for i in instructions if i['lifetime']['end'] > i['lifetime']['begin']]
    contexts = collections.defaultdict(list)
    groups = collections.defaultdict(list)
    for i in active:
        for c in i['contexts']:
            contexts[c].append((i['lifetime']['begin'], i['lifetime']['end']))
        groups[(i['description'].split('\n')[0], i['tpe'])].append(i)
    # Compiler-added table setup: anonymous HBM load (4096 B), then Sub expansion,
    # whose output is an additional input to a source-level fetch_table_lookup.
    consumers = collections.defaultdict(list)
    for i in instructions:
        for tid in i['input_tensors']:
            consumers[tid].append(i)
    setups = []
    for i in active:
        if i['tpe'] != 'DmaLoad' or i['description'] or len(i['output_tensors']) != 1:
            continue
        tid = i['output_tensors'][0]
        if tensors[tid]['size'] != 4096:
            continue
        for sub in consumers[tid]:
            if sub['tpe'] != 'Sub' or sub['description']:
                continue
            for out in sub['output_tensors']:
                uses = [u for u in consumers[out] if '.fetch_table_lookup' in u['description']]
                if uses:
                    setups.append({'load_index': i['index'], 'expand_index': sub['index'],
                                   'load_cycles': i['lifetime']['end'] - i['lifetime']['begin'],
                                   'expand_cycles': sub['lifetime']['end'] - sub['lifetime']['begin'],
                                   'table_bytes': tensors[out]['size'],
                                   'consumer_indices': [u['index'] for u in uses]})
    sram = [t for t in data['tensors'] if t['buffer_type'] == 'Sram']
    peak, peak_cycle = 0, 0
    for cycle in sorted({t['lifetime']['begin'] for t in sram}):
        size = union_size((t['address'], t['address'] + t['size']) for t in sram
                          if t['lifetime']['begin'] <= cycle < t['lifetime']['end'])
        if size > peak:
            peak, peak_cycle = size, cycle
    return {
        'makespan': max(i['lifetime']['end'] for i in instructions),
        'instruction_count': len(instructions), 'tensor_count': len(tensors),
        'instruction_types': dict(collections.Counter(i['tpe'] for i in instructions)),
        'context_busy_union_cycles': {c: union_size(spans) for c, spans in contexts.items()},
        'sram_live_address_union_peak_bytes': peak,
        'sram_live_address_union_peak_cycle': peak_cycle,
        'lut_setups': setups,
        'operator_groups': [
            {'source': loc, 'type': typ, 'count': len(items),
             'duration_sum': sum(i['lifetime']['end'] - i['lifetime']['begin'] for i in items),
             'durations': sorted({i['lifetime']['end'] - i['lifetime']['begin'] for i in items}),
             'begin': min(i['lifetime']['begin'] for i in items),
             'end': max(i['lifetime']['end'] for i in items)}
            for (loc, typ), items in groups.items()],
        'largest_sram_tensors': sorted(sram, key=lambda t: t['size'], reverse=True)[:8],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('report', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result = {str(path.relative_to(args.report / 'schedules')): analyze(path)
              for path in sorted((args.report / 'schedules').glob('*/*.json'))}
    encoded = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end='')


if __name__ == '__main__':
    main()
