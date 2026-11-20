"""Request-scoped, bounded BF16 row-layout cache and exact original-order tails."""
import collections,ctypes,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import torch
from enabled_sequential_repair_candidate import original_backend

class LayoutTail:
    def __init__(self,mode=16,pad=True,max_bytes=64*1024*1024,trusted_inference=None):
        if mode not in (4,16) or max_bytes<0:raise ValueError('Invalid fixed layout policy')
        self.mode=mode;self.pad=pad;self.max_bytes=max_bytes;self.trusted=trusted_inference or (lambda module:False)
        self.cache=collections.OrderedDict();self.active=False;self.closed=False;self.live_bytes=0
        self.counts={'calls':0,'selected':0,'builds':0,'hits':0,'evictions':0,'invalidations':0,'oversize_bypasses':0,'peak_bytes':0,'allocated_bytes':0,'prepare_seconds':0.,'release_seconds':0.,'requests':0}
        root=Path(__file__).parent;manifest=json.loads((root/'selected-layout-tail-build.json').read_text())
        for name,digest in manifest['sha256'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Layout-tail build mismatch')
        self.lib=ctypes.CDLL(str(root/'libselected_layout_tail.so'));self.fn=self.lib.d3_layout_tail
        self.fn.argtypes=[ctypes.c_void_p,ctypes.c_int64,ctypes.c_void_p,ctypes.c_int64,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p];self.fn.restype=ctypes.c_int
    def clear(self):
        start=time.perf_counter();self.cache.clear();self.live_bytes=0;self.counts['release_seconds']+=time.perf_counter()-start
    def begin(self):
        if self.active or self.closed:raise RuntimeError('Invalid layout-cache request boundary')
        self.clear();self.active=True;self.counts['requests']+=1
    def end(self):self.clear();self.active=False
    def close(self):self.end();self.closed=True
    def layout(self,module):
        weight=module.weight
        if not self.pad:return weight
        if weight.is_inference():
            if not self.trusted(module):raise ValueError('Versionless layout weight lacks original packed guard')
            version=None
        else:version=weight._version
        key=id(module);previous=self.cache.get(key)
        signature=(tuple(weight.shape),tuple(weight.stride()),weight.data_ptr(),version)
        if previous is not None:
            if previous[0] is weight and previous[1]==signature:
                self.cache.move_to_end(key);self.counts['hits']+=1;return previous[2]
            self.live_bytes-=previous[3];del self.cache[key];del previous;self.counts['invalidations']+=1
        n,k=weight.shape;size=n*(k+32)*2
        if size>self.max_bytes:self.counts['oversize_bypasses']+=1;return weight
        start=time.perf_counter()
        while self.live_bytes+size>self.max_bytes:
            _,entry=self.cache.popitem(last=False);self.live_bytes-=entry[3];self.counts['evictions']+=1;del entry
        storage=torch.empty((n,k+32),dtype=torch.bfloat16);value=storage[:,:k];value.copy_(weight)
        if value.data_ptr()%64:raise ValueError('Unexpected BF16 allocator alignment')
        self.cache[key]=(weight,signature,value,size);self.live_bytes+=size
        self.counts['builds']+=1;self.counts['allocated_bytes']+=size;self.counts['peak_bytes']=max(self.live_bytes,self.counts['peak_bytes'])
        self.counts['prepare_seconds']+=time.perf_counter()-start
        return value
    def run(self,module,inputs,positions,output,debug=None):
        if not self.active or self.closed:raise RuntimeError('Layout repair requires active request')
        if not original_backend():raise ValueError('Original enabled CPU backend changed')
        w=module.weight
        if (inputs.device.type!='cpu' or inputs.dtype!=torch.bfloat16 or inputs.ndim!=2 or inputs.stride(1)!=1 or w.device.type!='cpu' or w.dtype!=torch.bfloat16 or w.ndim!=2 or w.stride(1)!=1 or inputs.shape[1]!=w.shape[1] or positions.device.type!='cpu' or positions.dtype!=torch.long or positions.ndim!=2 or positions.shape[1]!=2 or output.device.type!='cpu' or output.dtype!=torch.bfloat16 or output.shape!=(inputs.shape[0],w.shape[0]) or not output.is_contiguous()):raise ValueError('Unsupported layout repair tensors')
        if debug is not None and (debug.device.type!='cpu' or debug.dtype!=torch.float32 or debug.shape!=(len(positions),) or not debug.is_contiguous()):raise ValueError('Unsupported debug buffer')
        # Invalid dimensions/indices are rejected by the C guard before output writes.
        weight=self.layout(module);pairs=positions.contiguous()
        rc=self.fn(inputs.data_ptr(),inputs.stride(0),weight.data_ptr(),weight.stride(0),inputs.shape[0],inputs.shape[1],weight.shape[0],pairs.data_ptr(),len(pairs),output.data_ptr(),self.mode,debug.data_ptr() if debug is not None else None)
        if rc:raise ValueError('Layout-tail guard failed: '+str(rc))
        self.counts['calls']+=1;self.counts['selected']+=len(pairs)

class ScopedLayout:
    def __init__(self,owner):
        self.owner=owner;self.state=owner.medium_sparse.repair;self.old_kernel=self.state.blas_repair;self.old_request=owner.model.system_one;self.enabled=False;self.closed=False
        def trusted(module):return any(module is item for modules,_ in owner.engine.packed.values() for item in modules)
        self.kernel=LayoutTail(trusted_inference=trusted)
        def request(*args,**kwargs):
            if not self.enabled:return self.old_request(*args,**kwargs)
            self.kernel.begin()
            try:return self.old_request(*args,**kwargs)
            finally:self.kernel.end()
        self.wrapper=request;owner.model.system_one=request
    def use(self,enabled):
        if self.closed or self.kernel.active:raise RuntimeError('Cannot change active layout policy')
        self.enabled=bool(enabled);self.state.blas_repair=self.kernel if enabled else self.old_kernel
    def close(self):
        if self.closed:return
        self.use(False);self.kernel.close();self.closed=True
        if self.owner.model.system_one is self.wrapper:self.owner.model.system_one=self.old_request

def install(owner):
    owner.selected_layout=ScopedLayout(owner);owner.selected_layout.use(True)
    return owner.selected_layout.close
