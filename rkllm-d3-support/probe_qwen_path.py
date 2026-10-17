"""Inspect how the public loader selects the Qwen3.5 conversion mapping."""
import json
from pathlib import Path
from rkllm.api import RKLLM

llm=RKLLM()
ret=llm.load_huggingface('/work/source-model',device='cpu',dtype='float32',load_weight=False)
report={'return_code':ret}
for key in ['modelname','modelpath','custom_config','state','v']:
    value=getattr(llm.base,key,None)
    try: json.dumps(value); report[key]=value
    except TypeError: report[key]=repr(value)
report['loaded_class']=type(getattr(llm.base,'model',None)).__name__
(Path('/support/logs')/'qwen-path.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
assert ret==0
