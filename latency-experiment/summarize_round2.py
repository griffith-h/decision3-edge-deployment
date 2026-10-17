"""Summarize retained raw evidence. No benchmark samples are filtered out."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--selected', choices=('default', 'three'), required=True)
a = p.parse_args()
def read(name):
    return json.loads((ROOT/'logs'/name).read_text())
def timing(name):
    source = read(name)
    records = source['records']
    times = sorted(r['seconds'] for r in records)
    return dict(source='logs/'+name, count=len(times), median=statistics.median(times),
                p95_nearest_rank=times[math.ceil(.95*len(times))-1], maximum=max(times),
                below_one_second=sum(t<1 for t in times),
                median_minor_faults=statistics.median(r['minor_faults'] for r in records),
                median_voluntary_switches=(statistics.median(r['voluntary_switches'] for r in records)
                                           if 'voluntary_switches' in records[0] else None),
                settings=source.get('settings'), native_stats=source.get('native_stats'),
                backend=source.get('backend'))

result = dict(scope='RK3576, CPU cores 4-7, Torch 4 threads, 115 input tokens, single question, concurrency 1. '
                    'Each fresh process gets exactly one warmup; initialization excluded. No response memoization.',
              model='vllm-sr/d3-edge@20c745653400d191ecacd99c4eafc0c5900dc15e',
              selected=a.selected, original_weights_and_12_layers_preserved=True,
              noncausal_full_attention_and_causal_delta_scan_preserved=True,
              experiments={}, quality=None, http=None, production=None)
for key, filename in [('previous_candidate', 'tail-default-allocator.json'),
                      ('early_native_trial', 'tail-native-default.json'),
                      ('norm_only_passive', 'tail-native-passive.json'),
                      ('norm_only_active', 'tail-native-active.json'),
                      ('final_four_cores', 'tail-final.json'),
                      ('reserved_three_cores', 'tail-reserved.json')]:
    if (ROOT/'logs'/filename).exists():
        result['experiments'][key] = timing(filename)
if (ROOT/'logs/validation-final.json').exists():
    v = read('validation-final.json')
    result['operator_verification'] = dict(passed=v['passed'], principal_cases=164,
                                          source='logs/validation-final-run.log')
    result['quality'] = dict(passed=v['passed'], cases=len(v['cases']),
        questions=sum(len(c['candidate']['response']['answers']) for c in v['cases'].values()),
        choices_unchanged=all(c['candidate']['choices_unchanged'] for c in v['cases'].values()),
        max_delta_vs_original_rknn=max(c['candidate']['max_delta_vs_original_rknn'] for c in v['cases'].values()),
        cpu_reference_cases=sum('max_delta_vs_historical_optimized_cpu' in c['candidate'] for c in v['cases'].values()),
        max_delta_vs_historical_optimized_cpu=max(c['candidate'].get('max_delta_vs_historical_optimized_cpu',0)
                                                 for c in v['cases'].values()),
        native_stats=v.get('native_stats'), source='logs/validation-final.json')
if a.selected == 'three' and (ROOT/'logs/tail-reserved.json').exists():
    checks = read('tail-reserved.json')['case_checks']
    c = checks['cases']
    result['quality'] = dict(passed=checks['passed'], cases=len(c),
        questions=sum(len(x['response']['answers']) for x in c.values()),
        choices_unchanged=all(x['choices_unchanged'] for x in c.values()),
        max_delta_vs_original_rknn=max(x['max_delta_vs_original_rknn'] for x in c.values()),
        cpu_reference_cases=sum('max_delta_vs_cpu' in x for x in c.values()),
        max_delta_vs_historical_optimized_cpu=max(x.get('max_delta_vs_cpu',0) for x in c.values()),
        source='logs/tail-reserved.json', reference='logs/validation-final.json')
http_name = 'http-reserved.json' if a.selected == 'three' else 'http-final.json'
if (ROOT/'logs'/http_name).exists():
    result['http'] = read(http_name)
if (ROOT/'logs/service-preservation-round2.json').exists():
    result['production'] = read('service-preservation-round2.json')
result['source_sha256'] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                          for name in ('cpu_attention_optimized.py','native_norm.c','native_norm.py')}
result['exploratory_control_note'] = ('The early ACTIVE/PASSIVE trials had a conservative conv fallback, '
    'so they are not equivalent controls of the final native-conv implementation. Final runs assert '
    'and record norm_calls and conv_calls separately. A PASSIVE wrapper exited 2 after completing '
    'all 100 samples because the launch script was overwritten while running; its measurements '
    'are retained as exploratory evidence. Subsequent launch-script updates used atomic rename.')
result['serving_teardown_note'] = ('The temporary HTTP serving sessions were explicitly stopped after '
    'all saved numerical checks and 100 HTTP requests passed. Their docker-run sessions exited 137 '
    'during teardown; this is not the status of the completed numerical/HTTP checks. '
    'No production container was stopped or restarted.')
if (ROOT/'logs/interference-analysis.json').exists():
    result['health_interference'] = read('interference-analysis.json')
if (ROOT/'logs/interference-reserved.json').exists():
    result['health_under_three_threads'] = read('interference-reserved.json')
if (ROOT/'logs/health-client-cost.json').exists():
    result['health_client_cost'] = read('health-client-cost.json')
selected = result['experiments'].get('reserved_three_cores' if a.selected == 'three' else 'final_four_cores')
result['selected_fresh_p95_under_one_second'] = selected['p95_nearest_rank'] < 1
result['selected_fresh_all_samples_under_one_second'] = selected['below_one_second'] == selected['count']
(ROOT/'result-round2.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
