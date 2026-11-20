"""RK-host final HTTP parity check against 34 frozen native-Owner cases."""
import argparse
import concurrent.futures
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request

from launch_release import OLD, RELEASE, api_environment, inspect, package
from stages import HISTORY


def numeric(value, path='', include_confidence=True):
    if isinstance(value, dict):
        return {k: v for key, item in value.items() if key != 'legend'
                and (include_confidence or key != 'confidence')
                for k, v in numeric(item, path + '/' + key, include_confidence).items()}
    if type(value) in (float, int):
        if not math.isfinite(value):
            raise AssertionError('Non-finite response: ' + path)
        return {path: value}
    return {}


def compare(actual, expected, include_confidence=True):
    x = numeric(actual['answers'], include_confidence=include_confidence)
    y = numeric(expected['answers'], include_confidence=include_confidence)
    if x.keys() != y.keys() or actual['answers'].keys() != expected['answers'].keys():
        raise AssertionError('Response schema changed')
    same = all(v.get('choice') == expected['answers'][key].get('choice')
               for key, v in actual['answers'].items())
    same = same and all((v['noul'] >= .5) == (expected['answers'][key]['noul'] >= .5)
                        for key, v in actual['answers'].items() if 'noul' in v)
    return {'max_numeric_delta': max((abs(x[k] - y[k]) for k in x), default=0),
            'choices_unchanged': same}


def request_payload(case):
    # Preserve multimodal regression inputs; expected answers stay in the checker.
    return {key: case['request'][key] for key in ('state', 'questions', 'images', 'videos')
            if key in case['request']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--port', type=int, choices=[8900, 8901], default=8901)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.root).resolve()
    manifest, marker = package(root)
    env = api_environment(inspect(OLD))
    headers = {'Content-Type': 'application/json'}
    if env['DECISION_API_KEY']:
        headers['Authorization'] = 'Bearer ' + env['DECISION_API_KEY']
    def call(path, body=None):
        request = urllib.request.Request(f'http://192.168.1.164:{a.port}' + path,
                                        data=None if body is None else json.dumps(body).encode(), headers=headers)
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response), time.perf_counter() - start
    health, _ = call('/health')
    d = health['deployment']
    if not (health['status'] == 'ready' and d['release'] == RELEASE
            and d['sample_rows_sha256'] == manifest['sample_rows_sha256']
            and d['torch_threads'] == 3 and d['OMP_THREAD_LIMIT'] == 4 and d['affinity'] == [4, 5, 6, 7]):
        raise AssertionError('Wrong final endpoint configuration')
    cases = json.loads((Path(HISTORY['native']) / 'logs/regression-native.json').read_text())['cases']
    original = json.loads((Path(HISTORY['fusion']) / 'logs/references.json').read_text())['cases']
    if len(cases) != 34 or set(cases) != set(original):
        raise AssertionError('Wrong frozen regression set')
    proof = {'passed': False, 'port': a.port, 'manifest_sha256': marker, 'cases': {}}
    def save():
        Path(a.output).write_text(json.dumps(proof, indent=2) + '\n')
    def check(key, result):
        expected = cases[key]['output']
        same = compare(result, expected)
        baseline = compare(result, original[key]['output'], False)
        confidence = compare(result, original[key]['output'], True)
        if not (same['max_numeric_delta'] <= 1e-6 and same['choices_unchanged']
                and baseline['max_numeric_delta'] <= .01 and baseline['choices_unchanged']
                and result['model'] == expected['model'] and result['usage'] == expected['usage']):
            raise AssertionError('HTTP output parity failed: ' + key)
        return {'vs_native': same, 'vs_original_excluding_confidence': baseline,
                'vs_original_including_confidence': confidence}
    save()
    for key, case in cases.items():
        result, seconds = call('/v1/systemone', request_payload(case))
        proof['cases'][key] = {'seconds': seconds, **check(key, result)}
        save()
        print('HTTP_CASE', key, 'passed', flush=True)
    try:
        call('/v1/systemone', {'questions': {}})
    except urllib.error.HTTPError as exc:
        if exc.code != 422:
            raise
        proof['invalid_request_422'] = True
    else:
        raise AssertionError('Invalid request accepted')
    models, _ = call('/v1/models')
    if models['models'][0]['name'] != 'vllm-sr/d3-edge':
        raise AssertionError('Wrong model listing')
    keys = list(cases)[:2]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(call, '/v1/systemone', request_payload(cases[key])) for key in keys]
        proof['concurrent_distinct_requests'] = {key: check(key, job.result()[0]) for key, job in zip(keys, futures)}
    proof['final_health'], _ = call('/health')
    proof['regression_cases'] = len(proof['cases'])
    proof['passed'] = True
    save()
    print('HTTP_PASSED', a.port, len(cases), flush=True)


if __name__ == '__main__':
    main()
