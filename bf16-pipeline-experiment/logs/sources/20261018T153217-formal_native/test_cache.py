"""Ownership retirement test without touching model weights or the real SDK."""
from cache_owner import Owner
from types import SimpleNamespace
from pathlib import Path
import json,torch
app=object.__new__(Owner);model=torch.nn.Module();model.a=torch.nn.Linear(1,1);model.b=torch.nn.Linear(1,1);model.c=torch.nn.Linear(1,1)
model.a._bp_handle=1;model.b._bp_handle=None;model.c._bp_handle=None;private=SimpleNamespace(_bp_handle=2);destroyed=[]
app.model=SimpleNamespace(backbone=model);app.engine=SimpleNamespace(handles=[1,2,91,92,93],packed={1:([private],[1])});app.float_handles={id(model.b):91,id(model.c):92,999:93};app.mlp_ids={id(model.b),id(model.c)};app.native_mlp=SimpleNamespace(handles={100:105});app.mixed=SimpleNamespace(mm_destroy=destroyed.append);app.cache_retired={'half':0,'normal_fp32_mlp':0};app.cache_last_events=[]
app.retire('half');assert set(destroyed)=={1,2} and app.engine.handles==[91,92,93] and model.a._bp_handle is None and private._bp_handle is None and len(app.float_handles)==3 and app.native_mlp.handles=={100:105}
app.retire('normal_fp32_mlp');assert set(destroyed)=={1,2,91,92} and app.engine.handles==[93] and app.float_handles=={999:93} and app.native_mlp.handles=={100:105}
before=list(destroyed);app.retire('half');app.retire('normal_fp32_mlp');assert destroyed==before
r={'completed':True,'half_registry_includes_private_packed_modules':True,'float_bank_protected':True,'only_normal_fp32_mlp_retired':True,'native_bank_untouched':True,'repeated_retirement_idempotent':True,'retired':app.cache_retired};Path('/experiment/logs/cache-unit.json').write_text(json.dumps(r,indent=2));print('CACHE UNIT',r,flush=True)
