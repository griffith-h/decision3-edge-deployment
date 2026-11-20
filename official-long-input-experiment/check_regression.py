"""Evaluate existing frozen numerical gates, confidence reported separately."""
import argparse,json
from pathlib import Path
from summarize import compare

def check_pair(response,reference):
 native=compare(response,reference['native'],False);native_all=compare(response,reference['native'])
 original=compare(response,reference['current'],False);all_fields=compare(response,reference['current'],True)
 # The new medium domain intentionally changes attention arithmetic. Keep
 # both original and previous non-confidence bounds, plus all decisions.
 for label,parity in (('previous native',native),('original numerical gate',original)):
  assert parity['max_delta']<=.01 and all(parity[k] for k in ('choices_unchanged','usage_equal','model_equal')),(label,parity)
 # The existing fast shapes are outside the new domain and must stay equal.
 if response['usage'].get('input_tokens') in (115,615):assert native_all['max_delta']<=1e-6,('protected native identity',native_all)
 return native,native_all,original,all_fields

def check(folder):
 execution=json.loads((folder/'execution.json').read_text())
 assert execution['exit_code']==0 and execution['retained_healthy'] and not execution.get('cleanup_errors')
 data=json.loads((folder/'regression.json').read_text());refs=json.loads((folder.parent/'regression-references.json').read_text())
 assert data['completed'] and len(data['records'])==len(refs)==34
 rows={};maximum=0.;maximum_confidence=0.
 for record in data['records']:
  key=record['id'];native,native_all,original,all_fields=check_pair(record['response'],refs[key])
  maximum=max(maximum,original['max_delta']);maximum_confidence=max(maximum_confidence,all_fields['max_delta'])
  rows[key]={'vs_previous_native':native,'vs_previous_native_all_fields':native_all,'vs_original_without_confidence':original,'vs_original_all_fields':all_fields}
 result={'passed':True,'cases':len(rows),'gate':'Original and previous probability/score/noul<=0.01; choice and binary decisions equal;115/615 native identity<=1e-6; confidence separate','max_original_delta_excluding_confidence':maximum,'max_original_delta_including_confidence':maximum_confidence,'rows':rows}
 (folder/'regression-check.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='rows'}));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();check(Path(a.folder))
