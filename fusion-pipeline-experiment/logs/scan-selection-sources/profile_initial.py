import sys
sys.path.insert(0,'/tail')
from runner import *
import ctypes,statistics,resource
p=Path('/experiment');app=Owner();case=json.loads(Path('/tail/logs/formal.json').read_text())['cases']['words-512']['request'];r={'settings':state(),'scope':'Independent profile; native stage times are sums of worker times, not elapsed critical path','model':[],'attention':[]}
lib=ctypes.CDLL(str(p/'libattention.so'));lib.d3_set_sgemm.argtypes=[ctypes.c_void_p];lib.d3_set_sgemm(ctypes.cast(lk.blas.scipy_cblas_sgemm64_,ctypes.c_void_p));lib.d3_attention.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*5+[ctypes.c_float,ctypes.c_int,ctypes.c_void_p];lib.d3_attention.restype=ctypes.c_int
try:
 for mode in ('current','validate_mlp'):
  app.use(mode);out=app.invoke(case);r[mode+'_reference']=out
 app.use('validate_mlp');app.invoke(case)
 for i in range(3):
  app.use('validate_mlp');t=time.perf_counter();out=app.invoke(case);r['model'].append({'seconds':time.perf_counter()-t,'output':out,**app.counts()});print('BASE',r['model'][-1]['seconds'],flush=True)
 captured=torch.load('/tail/captured.pt',weights_only=False)
 with torch.inference_mode():
  for rep in range(3):
   for layer,v in enumerate(captured['attention']):
    q,k,value,mask,scale=v['args'];b,h,n,d=q.shape;out=torch.empty((b,n,h,d),dtype=q.dtype);s=(ctypes.c_int64*9)(*[z for x in (q,k,value) for z in x.stride()[:3]]);st=(ctypes.c_double*(b*h*5))()
    t=time.perf_counter();rc=lib.d3_attention(q.data_ptr(),k.data_ptr(),value.data_ptr(),mask.data_ptr(),out.data_ptr(),ctypes.addressof(s),b,h,k.shape[1],n,d,scale,1,ctypes.addressof(st));dt=time.perf_counter()-t
    assert rc==0 and torch.equal(out,v['reference'])
    rec={'rep':rep,'layer':layer,'shape':list(q.shape),'kv_shape':list(k.shape),'seconds':dt,'worker_seconds':{key:sum(st[j*5+i] for j in range(b*h)) for i,key in enumerate(('pack_check','qk','softmax','pv','writeback'))}}
    r['attention'].append(rec);print('ATT',layer,dt,rec['worker_seconds'],flush=True)
 r['handles']=len(app.engine.handles);r['rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;r['completed']=True;r['final_settings']=state();(p/'logs/profile-initial.json').write_text(json.dumps(r,indent=2)+'\n')
finally:app.close()
