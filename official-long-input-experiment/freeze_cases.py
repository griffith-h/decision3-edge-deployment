"""Freeze private input-only development cases; references never reach the RK worker."""
import hashlib,json
from pathlib import Path
from decision_index.suite.io import read_jsonl

ROOT=Path('output/decision-index-audit')
def sha(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
 target=ROOT/'iterations/r001';target.mkdir(parents=True,exist_ok=True)
 if (target/'inputs.json').exists():raise FileExistsError('Development cases already frozen')
 sample={r['_evaluation']['run_id']:r for r in read_jsonl(ROOT/'sample-1000.jsonl.gz')}
 results=list(read_jsonl(ROOT/'runs/native-bf16/results.jsonl',complete_lines_only=True))
 names=['short314','tools32','long7232','medium997','medium1659','multi4605','multi4597']
 assert len(results)==len(names)
 inputs=[];references={};manifest=[]
 for name,result in zip(names,results):
  row=sample[result['run_id']];payload={k:row[k] for k in ('state','questions')}
  inputs.append({'id':name,'payload':payload})
  manifest.append({'id':name,**row['_evaluation'],'development_payload_sha256':sha(payload),'questions':len(payload['questions'])})
  if result['status']=='ok':references[name]={'response':result['response'],'method_ms':result['raw_output']['board_method_ms']}
 cases=json.loads(Path('cpu-tail-deployment/http-cases.json').read_text())
 short=next(c for c in cases if c['id']=='single');body={k:short[k] for k in ('state','questions')}
 for name,payload in [('protect115',body),('protect615',dict(body,state=' '.join(['alpha']*512)))]:
  inputs.append({'id':name,'payload':payload});manifest.append({'id':name,'development_payload_sha256':sha(payload),'source':'existing self-built protection case'})
 (target/'inputs.json').write_text(json.dumps(inputs,ensure_ascii=False)+'\n')
 (target/'references.json').write_text(json.dumps(references,ensure_ascii=False)+'\n')
 (target/'cases-manifest.json').write_text(json.dumps({'scope':'development, not full benchmark','source_rows_sha256':json.loads((ROOT/'sample-1000.jsonl.manifest.json').read_text())['sample_rows_sha256'],'cases':manifest},ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'cases':len(inputs),'references':len(references),'directory':str(target)}))
if __name__=='__main__':main()
