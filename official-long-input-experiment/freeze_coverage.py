"""Freeze complete-group development coverage inside the original1000 rows."""
import collections,hashlib,json
from pathlib import Path
from decision_index.suite.io import read_jsonl,sha256_file

ROOT=Path('output/decision-index-audit')
EXPECTED='57d0e3fdd655741beacc5cf492131aa061106d63a613a3f9530c1356bc57c35d'

def main():
    target=ROOT/'development-coverage'
    if target.exists():raise FileExistsError('Coverage view already frozen')
    sample=ROOT/'sample-1000.jsonl.gz'
    assert sha256_file(sample,gunzip=True)==EXPECTED
    rows=list(read_jsonl(sample));assert len(rows)==1000
    known={r.get('run_id') for r in json.loads((ROOT/'iterations/r001/cases-manifest.json').read_text())['cases']}
    groups=collections.defaultdict(list);catalogs={}
    for index,row in enumerate(rows):
        e=row['_evaluation'];key=(e['catalog_id'],e['group_id'])
        groups[key].append((index,row));catalogs.setdefault(e['catalog_id'],[])
        if key not in catalogs[e['catalog_id']]:catalogs[e['catalog_id']].append(key)
    selected=[]
    for catalog,keys in catalogs.items():
        key=next((k for k in keys if all(r['_evaluation']['run_id'] not in known for _,r in groups[k])),keys[0])
        selected.extend(groups[key])
    selected.sort(key=lambda item:item[0]);assert len(catalogs)==43
    inputs=[];manifest=[]
    for number,(index,row) in enumerate(selected):
        e=row['_evaluation'];key=f'coverage-{number:04d}'
        payload={k:row[k] for k in ('state','questions')}
        digest=hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        inputs.append({'id':key,'payload':payload})
        manifest.append({'id':key,'original_index':index,'run_id':e['run_id'],'catalog_id':e['catalog_id'],
            'group_id':e['group_id'],'development_payload_sha256':digest,'questions':len(payload['questions']),
            'previously_exposed':e['run_id'] in known})
    target.mkdir()
    data=json.dumps(inputs,ensure_ascii=False)+'\n';(target/'coverage-inputs.json').write_text(data)
    record={'scope':'development complete-group coverage view, no replacement or resampling of original1000',
        'source_sample_rows_sha256':EXPECTED,'selection':'First complete group with no previously exposed request, per catalog, using frozen metadata/order only; fallback first group',
        'catalogs':len(catalogs),'groups':len(catalogs),'requests':len(inputs),
        'questions':sum(r['questions'] for r in manifest),'input_file_sha256':hashlib.sha256(data.encode()).hexdigest(),'cases':manifest}
    (target/'coverage-manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='cases'}))

if __name__=='__main__':main()
