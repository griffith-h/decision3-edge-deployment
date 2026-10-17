"""Inspect published SDK interfaces only; do not infer support from metadata alone."""
import importlib
import inspect
import json
from pathlib import Path
import subprocess
from contextlib import redirect_stdout
import io

import rkllm.base.converter as converter
import rkllm.base.loader as loader
import rkllm.base.common as common

root = Path('/support')
def describe(obj):
    result = {'type':str(type(obj)), 'doc':inspect.getdoc(obj)}
    try: result['signature']=str(inspect.signature(obj))
    except (TypeError,ValueError): pass
    return result

report = {'core_modules':{},'configs':{},'converter_classes':{},'help':{}}
for module in [converter,loader,common]:
    report['core_modules'][module.__name__]=module.__file__
    if hasattr(module,'custom_help'):
        stream=io.StringIO()
        try:
            with redirect_stdout(stream):
                value=module.custom_help()
            report['help'][module.__name__]={'stdout':stream.getvalue(),'return':repr(value)}
        except Exception as exc: report['help'][module.__name__]={'exception':repr(exc)}
for key in ['qwen3_5_config','qwen3_config','allow_model','model_type_map']:
    value=getattr(common,key,getattr(loader,key,None))
    report['configs'][key]=repr(value)
for key in ['Model','CustomModel','Qwen2Model','BertModel']:
    cls=getattr(converter,key)
    report['converter_classes'][key]={name:describe(getattr(cls,name)) for name in dir(cls) if not name.startswith('_')}
report['functions']={name:describe(getattr(converter,name)) for name in ['convert_model','get_llm_config','get_key_opts']}
report['model_archs']={str(k):str(v) for k,v in converter.MODEL_ARCH_NAMES.items()}
exports=subprocess.check_output(['nm','-D','--defined-only','/work/runtime/librkllmrt.so'],text=True)
report['runtime_exports']=exports.splitlines()
(root/'logs/sdk-public-api.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'configs':report['configs'],'help':report['help'],'functions':report['functions'],
                  'custom_class':report['converter_classes']['CustomModel'],'model_archs':report['model_archs']},ensure_ascii=False,indent=2))
