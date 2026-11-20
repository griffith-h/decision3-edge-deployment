"""Offline development evidence; never score incomplete diagnostics as full-suite results."""
import argparse,collections,json,math
from pathlib import Path

def numbers(value,path='',confidence=True):
 if isinstance(value,dict):return {k:v for name,item in value.items() if name!='legend' and (confidence or name!='confidence') for k,v in numbers(item,path+'/'+name,confidence).items()}
 if type(value) in (float,int):
  if not math.isfinite(value):raise ValueError('Nonfinite output '+path)
  return {path:value}
 return {}
def compare(actual,expected,confidence=True):
 a,b=numbers(actual['answers'],confidence=confidence),numbers(expected['answers'],confidence=confidence)
 if a.keys()!=b.keys() or actual['answers'].keys()!=expected['answers'].keys():raise ValueError('Answer schema changed')
 noul_unchanged=all((v['noul']>=.5)==(expected['answers'][k]['noul']>=.5)
                    for k,v in actual['answers'].items() if 'noul' in v)
 return {'max_delta':max((abs(a[k]-b[k]) for k in a),default=0),'choices_unchanged':noul_unchanged and all(v.get('choice')==expected['answers'][k].get('choice') for k,v in actual['answers'].items()),'noul_predictions_unchanged':noul_unchanged,'usage_equal':actual['usage']==expected['usage'],'model_equal':actual['model']==expected['model']}
def summarize(folder):
 execution=json.loads((folder/'execution.json').read_text());totals=collections.defaultdict(lambda:{'completed_calls':0,'inclusive':0.,'exclusive':0.});stack=[];batches=[]
 path=folder/'trace.jsonl'
 if path.exists():
  for line in path.read_text().splitlines():
   event=json.loads(line)
   if event['event']=='batch':batches.append({k:v for k,v in event.items() if k not in ('event','at')})
   elif event['event']=='begin':stack.append({**event,'child':0.})
   elif event['event']=='end':
    frame=stack.pop();assert frame['tag']==event['tag']
    row=totals[event['kind']];row['completed_calls']+=1;row['inclusive']+=event['seconds'];row['exclusive']+=max(0,event['seconds']-frame['child'])
    if stack:stack[-1]['child']+=event['seconds']
 report={'run':folder.name,'case':execution['case'],'completed':execution.get('request_completed',False),'budget_exceeded':execution.get('budget_exceeded',False),'service_healthy':execution.get('retained_healthy'),'batches':batches,'completed_spans_by_kind':dict(totals),'incomplete_stack':[x['tag'] for x in stack],'note':'Instrumented spans, inclusive entries overlap; incomplete stacks do not have a complete duration.'}
 result=folder/'result.json'
 if result.exists():
  r=json.loads(result.read_text());report.update(seconds=r['seconds'],rss_kib=r['rss_kib'],input_tokens=r['response']['usage']['input_tokens'])
  references=json.loads((folder.parent/'references.json').read_text())
  if r['case'] in references:
   report['vs_previous']=compare(r['response'],references[r['case']]['response'])
 (folder/'summary.json').write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();print(json.dumps(summarize(Path(a.folder)),indent=2))
