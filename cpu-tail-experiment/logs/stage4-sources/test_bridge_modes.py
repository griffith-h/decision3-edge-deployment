import ctypes,torch,time,json
from pathlib import Path
from baseline import stats
P=ctypes.c_void_p;I=ctypes.c_int
libs={}
for name,path in [('baseline','/previous/libconversion_bridge.so'),('lut','/work/liblut_bridge.so'),('native','/work/libnative_bridge.so'),('m115','/work/libm115_bridge.so')]:
 lib=ctypes.CDLL(path);lib.mm_create_core.argtypes=[I,I,P,I];lib.mm_create_core.restype=P;lib.mm_error.restype=ctypes.c_char_p
 lib.mm_destroy.argtypes=[P];lib.mm_destroy.restype=None
 for fn,args in [('mm_run_bf16',[P,I,P,P,P]),('mm_prepare_bf16',[P,I,P]),('mm_copy_prepared',[P,P,I]),('mm_run_prepared_bf16',[P,I,P,P]),('mm_run_prepared_scatter_bf16',[P,I,I]+[ctypes.POINTER(I)]*3+[ctypes.POINTER(P),P])]:
  f=getattr(lib,fn);f.argtypes=args;f.restype=I
 libs[name]=lib
r={'variants':{}};torch.set_num_threads(3);torch.manual_seed(199)
with torch.inference_mode():
 for name in ('lut','native','m115'):
  row={'cases':[]};r['variants'][name]=row;lib=libs[name]
  try:
   for k,n in ((32,32),(1024,1024),(3584,1024)):
    weight=(torch.randn(k,n)*.03).half().contiguous();base=libs['baseline'];h=base.mm_create_core(k,n,weight.data_ptr(),1);g=lib.mm_create_core(k,n,weight.data_ptr(),1)
    if not g:
     if h:base.mm_destroy(h)
     raise RuntimeError(lib.mm_error().decode())
    assert h
    try:
     for m in (1,3,115,128,129,512,115):
      x=(torch.randn(m,k)*.4).bfloat16();ref=torch.empty(m,n,dtype=torch.bfloat16);out=torch.empty_like(ref);before=x.clone()
      assert base.mm_run_bf16(h,m,x.data_ptr(),None,ref.data_ptr())==0
      code=lib.mm_run_bf16(g,m,x.data_ptr(),None,out.data_ptr())
      if code:raise RuntimeError(lib.mm_error().decode())
      delta=(out.float()-ref.float()).abs().max().item();torch.testing.assert_close(out,ref,atol=.02,rtol=.02);assert torch.equal(x,before)
      # Two aligned segments also support destination row gaps/column offsets.
      a=torch.full((m,n//2+4),-7,dtype=torch.bfloat16);b=torch.full_like(a,-7)
      st=(I*2)(0,n//2);wi=(I*2)(n//2,n//2);sr=(I*2)(n//2+4,n//2+4);tar=(P*2)(a.data_ptr()+4,b.data_ptr()+4)
      code=lib.mm_run_prepared_scatter_bf16(g,m,2,st,wi,sr,tar,None)
      if code:raise RuntimeError(lib.mm_error().decode())
      assert torch.equal(a[:,2:2+n//2],out[:,:n//2]) and torch.equal(b[:,2:2+n//2],out[:,n//2:]);assert bool((a[:,:2]==-7).all() and (a[:,-2:]==-7).all())
      row['cases'].append({'k':k,'n':n,'m':m,'delta':delta})
     vals=[];x=(torch.randn(115,k)*.4).bfloat16();out=torch.empty(115,n,dtype=torch.bfloat16)
     for _ in range(18):t=time.perf_counter();assert lib.mm_run_bf16(g,115,x.data_ptr(),None,out.data_ptr())==0;vals.append(time.perf_counter()-t)
     row.setdefault('timing',[]).append({'k':k,'n':n,**stats(vals)})
     huge=torch.full((3,k),float('inf'),dtype=torch.bfloat16);assert lib.mm_run_bf16(g,3,huge.data_ptr(),None,out.data_ptr())==-2
    finally:base.mm_destroy(h);lib.mm_destroy(g)
   row['passed']=True
  except Exception as e:row['passed']=False;row['error']=repr(e)
  print('BRIDGE',name,row['passed'],row.get('error'),flush=True)
  Path('/work/logs/bridge-modes-unit.json').write_text(json.dumps(r,indent=2)+'\n')
