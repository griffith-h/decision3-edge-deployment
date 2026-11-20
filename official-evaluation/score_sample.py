"""Use upstream native scorers with the frozen sample as their entire denominator."""
import collections
import copy
import json
import math
from pathlib import Path
import statistics

from decision_index import editions
from decision_index.pipeline import score_run
from decision_index.scoring import added, index02
from decision_index.scoring.metrics import prediction
from decision_index.scoring.report import benchmark_summary, load_results
from decision_index.suite.io import read_jsonl


class SampleSuite:
    def __init__(self, path):
        self.sample = list(read_jsonl(path))
        ids = [r['_evaluation']['run_id'] for r in self.sample]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate sampled request id')
        extra_ids = {int(n) for n in index02.spec('0.3')['added']}
        extra = sum(r['_evaluation']['catalog_id'] in extra_ids for r in self.sample)
        self.edition = {**editions.get('0.3'), 'requests': len(ids) - extra,
                        'scoreable': len(ids) - extra, 'excluded': 0,
                        'added_requests': extra,
                        'benchmarks': len({r['_evaluation']['catalog_id'] for r in self.sample})}

    def rows(self, apply_exclusions=True):
        return iter(self.sample)

    def excluded(self):
        return set()

    def manifest(self):
        return {'scope': 'frozen stratified sample, not full suite', 'requests': len(self.sample)}


def latency(values):
    if not values:
        return {'count': 0}
    values = sorted(values)
    assert all(math.isfinite(v) and v >= 0 for v in values)
    return {'count': len(values), 'p50_ms': statistics.median(values),
            'p95_ms': values[math.ceil(.95 * len(values)) - 1],
            'mean_ms': statistics.mean(values), 'min_ms': values[0], 'max_ms': values[-1]}


def counter_hits(records):
    paths = ('calls', 'graphs/optimized', 'mlp/optimized', 'mlp/calls',
             'native_mlp_calls', 'attention_npu_calls', 'projections/optimized')
    def value(data, path):
        for part in path.split('/'):
            if not isinstance(data, dict) or part not in data:
                return None
            data = data[part]
        return data if type(data) in (int, float) else None
    report = {}
    for path in paths:
        measured = hits = 0
        for rec in records:
            raw = rec['raw_output']
            before, after = value(raw.get('counts_before', {}), path), value(raw.get('counts', {}), path)
            if before is not None and after is not None:
                measured += 1
                hits += after > before
        if measured:
            report[path] = {'requests_observed': measured, 'requests_with_positive_delta': hits}
    return report


def labelled_fields(rows, results):
    """Extra, explicitly single-label metric; excludes ranking/preference/soft targets."""
    correct = total = 0
    per_group = {}
    counted_panel = {n for area in index02.spec('0.3')['areas'] for n in area['benchmarks']}
    for row in rows:
        if row['_evaluation']['catalog_id'] not in counted_panel:
            continue
        if row.get('scoring', {}).get('type') in (
                'retrieval_ranking', 'forecast_probability', 'routing_outcome',
                'chess_reference_value', 'human_group_rank'):
            continue
        e = row['_evaluation']
        actual = results.get(e['run_id'], {})
        answers = actual.get('response', {}).get('answers', {}) if actual.get('status') == 'ok' else {}
        group = (e['catalog_id'], e['group_id'])
        for key, q in row['questions'].items():
            gold = row.get('expected', {}).get(key)
            if not (q['type'] == 'choice' and isinstance(gold, str)
                    or q['type'] == 'noul' and type(gold) is bool):
                continue
            hit = key in answers and prediction(q, answers[key]) == gold
            total += 1
            correct += bool(hit)
            item = per_group.setdefault(str(group), [0, 0])
            item[0] += bool(hit)
            item[1] += 1
    return {'correct': correct, 'denominator': total, 'accuracy': correct / total if total else None,
            'scope': 'single-label choice and boolean noul fields on the official counted panel; errors/unsupported count wrong; '
                     'ranking, preference and soft targets excluded', 'groups': per_group}


def score(sample_path, result_path, out_dir, stage):
    suite = SampleSuite(sample_path)
    out_dir = Path(out_dir)
    results = load_results(result_path)
    observed = {r['_evaluation']['catalog_id'] for r in suite.sample}
    spec = index02.spec('0.3')
    required = {n for a in spec['areas'] for n in a['benchmarks']}
    if required <= observed:
        official = score_run(suite, result_path, stage, out_dir)
        # Preserve upstream files for audit, but never present this as a full-board result.
        native = official['benchmarks']
        sampled_index = official['decision_index']
    else:
        # A restricted corpus may not cover every panel member. Do not invent weights.
        extra_ids = {int(n) for n in spec['added']}
        base = [r for r in suite.sample if r['_evaluation']['catalog_id'] not in extra_ids]
        metrics = {int(n): (m['name'], m['key']) for n, m in spec.get('metrics', {}).items()}
        summary = benchmark_summary(suite, results, stage, rows=base, metrics=metrics)
        native = {str(b['catalog_id']): b for b in summary['benchmarks']}
        for n in observed & extra_ids:
            native[str(n)] = added.report(n, [r for r in suite.sample if r['_evaluation']['catalog_id'] == n], results)
        (out_dir / 'benchmark-summary.json').write_text(json.dumps(native, ensure_ascii=False, indent=2) + '\n')
        sampled_index = None
    # Last record wins exactly as upstream resume semantics; keep raw attempts on disk.
    raw = {r['run_id']: r for r in read_jsonl(result_path, complete_lines_only=True)}
    ids = {r['_evaluation']['run_id'] for r in suite.sample}
    if set(raw) - ids:
        raise ValueError('Results contain requests outside the frozen sample')
    successful = [r for r in raw.values() if r['status'] == 'ok']
    method = [r['raw_output']['board_method_ms'] for r in successful]
    bins = collections.defaultdict(list)
    for r in successful:
        tokens = r['response'].get('usage', {}).get('input_tokens', 0)
        label = '0-128' if tokens <= 128 else '129-512' if tokens <= 512 else '513-1024' if tokens <= 1024 else '1025+'
        bins[label].append(r['raw_output']['board_method_ms'])
    report = {'stage': stage, 'scope': 'frozen official-source stratified sample only; not full benchmark',
              'expected_requests': len(ids), 'completed_requests': len(raw),
              'sample_complete': set(raw) == ids,
              'counts': dict(collections.Counter(r['status'] for r in raw.values())),
              'labelled_field_accuracy': labelled_fields(suite.sample, results),
              'sampled_public_index': sampled_index,
              'missing_index_benchmarks': sorted(required - observed),
              'board_method': latency(method),
              'transport_request': latency([r['model_request_wall_ms'] for r in successful]),
              'method_by_input_tokens': {k: latency(v) for k, v in bins.items()},
              'acceleration_counter_hits': counter_hits(successful),
              'native_benchmark_metrics': native}
    (out_dir / 'sample-summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report
