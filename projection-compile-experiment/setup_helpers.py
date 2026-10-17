import ctypes,copy,json,sys
from pathlib import Path
sys.path[:0]=['/remainder','/base','/combined','/scan','/release','/work']
import torch
from common import compare,stats,state
def configure(engine):
    old=engine.lib;lib=ctypes.CDLL('/remainder/libsafe_half_bridge.so')
    for name in ('mm_error','mm_create_core','mm_run','mm_destroy','mm_prepare_bf16','mm_copy_prepared','mm_run_prepared_bf16','mm_run_bf16'):
        src=getattr(old,name);dst=getattr(lib,name);dst.argtypes=src.argtypes;dst.restype=src.restype
    engine.lib=lib
def cases():
    out=json.loads(Path('/references/request-cases.json').read_text())
    image=json.loads(Path('/references/golden-reference.json').read_text())['image'];image['state']='Describe the attached image.';out.append({'id':'image',**image})
    single=next(c for c in out if c['id']=='single')
    for name,text in [('novel-math','请计算 37 乘以 19，并检查结果。'),('novel-chat','今天下班后想和朋友散步聊天，有什么轻松的话题？')]:
        item=copy.deepcopy(single);item.update(id=name,state=text);out.append(item)
    return out
