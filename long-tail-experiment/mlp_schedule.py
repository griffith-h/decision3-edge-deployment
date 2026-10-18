"""Two existing independent graph handles; join all workers before any fallback."""
from concurrent.futures import ThreadPoolExecutor,wait
import torch

class Scheduler:
    def __init__(self,bank,set_core):
        self.bank=bank;self.set_core=set_core;self.original=bank.run_rows
        self.pool=ThreadPoolExecutor(max_workers=2,thread_name_prefix='d3-mlp')
        self.masks={id(h):0 for h in bank.extra};self.disabled=set();self.enabled=False;self.closed=False;self.reset()
    def reset(self):self.counts={'parallel_layers':0,'serial_layers':0,'errors':[],'chunks':0,'padded_rows':0}
    def core(self,h,mask):
        if self.masks[id(h)]!=mask:
            if self.set_core(h,mask):raise RuntimeError('MLP graph core mask failed')
            self.masks[id(h)]=mask
    def run(self,g,x):
        n=x.shape[0]*x.shape[1] if x.ndim==3 else 0;graphs=self.bank.by_id[id(g)]
        if not self.enabled or tuple(x.shape)!=(1,615,1024) or tuple(self.bank.plans.get(615,()))!=(128,128,384) or id(g) in self.disabled:
            for h in graphs.values():self.core(h,0)
            self.counts['serial_layers']+=1;return self.original(g,x)
        # The measured baseline uses exactly this 128+128+384 plan at 615 rows.
        # Use that same row assignment; only scheduling and core placement differ.
        flat=x.reshape(-1,1024);tasks={128:[(0,128),(128,128)],384:[(256,359)]}
        self.core(graphs[128],2);self.core(graphs[384],1)
        def worker(width):
            values=[]
            with torch.inference_mode():
                for start,used in tasks[width]:
                    if used==width:part=flat[start:start+used].reshape(1,width,1024)
                    else:
                        part=torch.zeros((1,width,1024),dtype=x.dtype);part[0,:used].copy_(flat[start:start+used])
                    value=graphs[width].run_bf16(part);values.append((start,used,value))
            return values
        futures=[self.pool.submit(worker,width) for width in (128,384)]
        wait(futures)
        try:values=[value for f in futures for value in f.result()]
        except (ValueError,RuntimeError) as exc:
            self.counts['errors'].append(str(exc));self.disabled.add(id(g))
            for h in graphs.values():self.core(h,0)
            return self.original(g,x)
        out=torch.empty_like(flat)
        for start,used,value in values:out[start:start+used].copy_(value.reshape(-1,1024)[:used])
        self.bank.counts['graph_calls']+=3;self.bank.counts['padded_rows']+=25
        self.bank.counts.setdefault('plans',{});self.bank.counts['plans']['128,128,384']=self.bank.counts['plans'].get('128,128,384',0)+1
        self.counts['parallel_layers']+=1;self.counts['chunks']+=3;self.counts['padded_rows']+=25
        return out.reshape(x.shape)
    def close(self):
        if self.closed:return
        self.closed=True;self.pool.shutdown(wait=True)
        for h in self.bank.extra:self.core(h,0)
