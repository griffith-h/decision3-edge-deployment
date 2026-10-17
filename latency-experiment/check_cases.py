"""Reuse the retained paired reference for a CPU scheduling configuration check."""
import copy
import json
from pathlib import Path
import time

def numeric(x, path=''):
    if isinstance(x, dict):
        result = {}
        for k,v in x.items():
            if k not in ('confidence', 'legend'):
                result.update(numeric(v, path+'/'+k))
        return result
    return {path:x} if isinstance(x, (int, float)) else {}

def compare(actual, expected):
    a, b = numeric(actual['answers']), numeric(expected['answers'])
    assert a.keys() == b.keys()
    delta = max(abs(a[k]-b[k]) for k in a)
    same = all(v.get('choice') == expected['answers'][k].get('choice') for k,v in actual['answers'].items())
    assert delta <= .01 and same, (delta, same)
    return delta

def validate_cases(model, reference_path='/work/logs/validation-final.json'):
    reference = json.loads(Path(reference_path).read_text())
    cpu = json.loads(Path('/base/logs/benchmark-all.json').read_text())['cases']
    cases = json.loads(Path('/base/request-cases.json').read_text())
    image = json.loads(Path('/base/golden-reference.json').read_text())['image']
    image['state'] = 'Describe the attached image.'
    cases.append({'id':'image', **image})
    single = next(c for c in cases if c['id'] == 'single')
    for name, text in [('novel-math', '请计算 37 乘以 19，并检查结果。'),
                       ('novel-chat', '今天下班后想和朋友散步聊天，有什么轻松的话题？')]:
        item = copy.deepcopy(single)
        item['id'], item['state'] = name, text
        cases.append(item)
    results = {}
    for case in cases:
        t = time.perf_counter()
        output = model.system_one(**{k:case[k] for k in ('state','questions','images','videos') if k in case})
        results[case['id']] = dict(seconds=time.perf_counter()-t,
            max_delta_vs_original_rknn=compare(output, reference['cases'][case['id']]['original_rknn']['response']),
            max_delta_vs_final_four_cores=compare(output, reference['cases'][case['id']]['candidate']['response']),
            choices_unchanged=True, response=output)
        if case['id'] in cpu:
            results[case['id']]['max_delta_vs_cpu'] = compare(output, cpu[case['id']]['cpu']['response'])
        print('CORE_CONFIG_VALIDATED', case['id'], flush=True)
    return dict(passed=True, cases=results)
