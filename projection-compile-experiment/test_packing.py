import setup_helpers
import json,time
from pathlib import Path
import torch
from merged_adapter import PackedEngine
from adapter import Engine
from setup_helpers import configure
torch.set_num_threads(3)
def run():
    torch.manual_seed(7);records=[]
    for mode in ('merge','balanced'):
        e=PackedEngine(mode);configure(e)
        modules=[]
        column_offset=0
        for i,n in enumerate((96,16,16)):
            m=torch.nn.Linear(1024,n,bias=True,dtype=torch.bfloat16)
            with torch.no_grad():
                values=torch.arange(column_offset+1,column_offset+n+1).bfloat16()/1024
                m.weight.copy_(values[:,None].expand(n,1024));m.bias.fill_(i/8)
            column_offset+=n
            m._bp_handle=None;m._bp_original=lambda x,m=m:torch.nn.functional.linear(x.float(),m.weight.float(),m.bias.float()).bfloat16()
            e.core[id(m)]=1 if i==0 else 2;e.bucket[id(m)]=0 if i==0 else 1
            modules.append(m)
        try:
            with torch.inference_mode():
                for rows in (1,115,128,129,512,1):
                    # Strided input; distinct output columns and nonzero bias.
                    x=torch.zeros((rows,2048),dtype=torch.bfloat16)[:,::2]
                    x[:,0]=((torch.arange(rows)%4+1).float()*.125).bfloat16()
                    outputs=e.group(modules,x)
                    assert all(torch.equal(outputs[id(m)],m._bp_original(x)) for m in modules)
                    records.append({'mode':mode,'rows':rows,'noncontiguous':True,'exact':True})
                for i,m in enumerate(modules):
                    with torch.no_grad():m.weight.fill_(1 if i==0 else 0)
                # Repack after changing synthetic test weights, never production weights.
                e.close();e=PackedEngine(mode);configure(e)
                x=torch.full((1,1024),1024,dtype=torch.bfloat16)
                outputs=e.group(modules,x)
                assert all(torch.equal(outputs[id(m)],m._bp_original(x)) for m in modules)
                assert e.stats['packed_range_fallbacks']==1
                normal=x/8192;outputs=e.group(modules,normal)
                assert all(torch.equal(outputs[id(m)],m._bp_original(normal)) for m in modules)
                bad=normal.clone();bad[0,0]=float('nan')
                try:e.group(modules,bad)
                except ValueError:pass
                else:raise AssertionError('invalid inputs must raise')
                # A worker failure must join the other job before propagating.
                native=hasattr(e,'run_piece');old=e.run_piece if native else Engine.execute;finished=[]
                def faulty(self,m,x,*args,**kw):
                    if self.core[id(m)]==1:raise RuntimeError('injected')
                    time.sleep(.03);finished.append(True);return old(m,x,*args,**kw) if native else old(self,m,x,*args,**kw)
                if native:
                    from types import MethodType
                    e.run_piece=MethodType(faulty,e)
                else:Engine.execute=faulty
                try:
                    try:e.group(modules,normal)
                    except RuntimeError as exc:assert 'injected' in str(exc)
                    else:raise AssertionError('missing injected error')
                    assert finished==[True]
                finally:
                    if native:e.run_piece=old
                    else:Engine.execute=old
                assert all(torch.equal(e.group(modules,normal)[id(m)],m._bp_original(normal)) for m in modules)
                e.close();e=PackedEngine(mode);configure(e)
                for m in modules:
                    m.bias=None
                    m._bp_original=lambda x,m=m:torch.nn.functional.linear(x.float(),m.weight.float()).bfloat16()
                outputs=e.group(modules,normal)
                assert all(torch.equal(outputs[id(m)],m._bp_original(normal)) for m in modules)
                assert all(p.bias is None for p in e.pack(modules)[0])
        finally:e.close()
    result={'passed':True,'records':records,'no_bias_fast_path':True,'whole_group_output_range_fallback':True,'input_validation':True,'all_workers_joined_on_error':True,'recovery':True}
    Path('/work/logs/packing-unit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PACKING_UNITS_PASSED',flush=True)
if __name__=='__main__':run()
