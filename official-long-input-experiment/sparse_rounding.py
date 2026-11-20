"""Experimental original-CPU GEMM repair near BF16 rounding boundaries.

The uncertainty band combines measured FP32 accumulation scale with a Cauchy
bound on input/weight FP16 conversion. This is an empirical development guard,
not a mathematical bound on arbitrary SDK implementations. Full-model numerical
and task gates remain mandatory. No labels or answers enter this module.
"""
import math,time
import torch
import torch.nn.functional as f

class SparseRepair:
    def __init__(self,noise_factor=1e-7,tile=4,max_fraction=.1,max_entries=65536,backend='gather',profile=False,native_preparation=False):
        if not math.isfinite(noise_factor) or noise_factor<=0 or tile!=4 or not 0<max_fraction<=1 or max_entries<1:raise ValueError('Invalid precision repair limits')
        if backend not in ('torch','gather','cblas'):raise ValueError('Unknown CPU repair backend')
        self.preparation=None
        if native_preparation:
            from sparse_cpu_backend import Preparation
            self.preparation=Preparation()
        self.profile=profile
        self.backend=backend;self.row_gather=None;self.blas_repair=None
        if backend=='gather':
            from sparse_cpu_backend import RowGather
            self.row_gather=RowGather()
        if backend=='cblas':
            from sparse_cpu_backend import BlasRepair
            self.blas_repair=BlasRepair()
        self.noise_factor=noise_factor;self.tile=tile;self.max_fraction=max_fraction;self.max_entries=max_entries;self.cache={}
        self.counts={'backend':backend,'calls':0,'entries':0,'selected':0,'changed':0,'seconds':0.,'density_rejections':0,'weight_norm_builds':0,'maximum_fraction':0.,'maximum_gather_float32_bytes':0,'phases':{'norms':0.,'mask':0.,'gather':0.,'bmm':0.,'scatter':0.}}
    @staticmethod
    def norm(x):return torch.linalg.vector_norm(x,dim=-1,dtype=torch.float64).float()
    def weight_norms(self,module,allow_inference_weight=False):
        weight=module.weight;key=id(module)
        if weight.is_inference():
            if not allow_inference_weight:raise ValueError('Versionless weight requires original-parameter guards')
            # Only private packed weights derived from RowProjection's guarded,
            # unchanged original parameters may use identity-only norm caching.
            version=None
        else:version=weight._version
        previous=self.cache.get(key)
        if previous is not None and previous[0] is weight and previous[1]==version:return previous[2:]
        if self.preparation is not None:norms,loss=self.preparation.norms(weight.detach())
        else:
            value=weight.detach().float();half=value.half().float()
            if not bool(torch.isfinite(half).all()):raise ValueError('Weight conversion overflow')
            norms=self.norm(value);loss=self.norm(value-half)
        self.cache[key]=(weight,version,norms,loss);self.counts['weight_norm_builds']+=1
        return norms,loss
    def repair(self,module,inputs,raw,allow_inference_weight=False):
        start=time.perf_counter()
        if (inputs.device.type!='cpu' or inputs.dtype!=torch.bfloat16 or inputs.ndim!=2
            or raw.device.type!='cpu' or raw.dtype!=torch.float32 or raw.shape!=(inputs.shape[0],module.weight.shape[0])
            or module.weight.device.type!='cpu' or module.weight.dtype!=torch.bfloat16
            or inputs.shape[1]!=module.weight.shape[1] or module.bias is not None
            or not 0<inputs.shape[0]<=615 or inputs.shape[1]>10240):raise ValueError('Unsupported precision repair tensors')
        if self.preparation is not None:
            wnorm,wloss=self.weight_norms(module,allow_inference_weight)
            xn,xloss,xhn=self.preparation.norms(inputs,True)
        else:
            if not bool(torch.isfinite(raw).all()):raise ValueError('Nonfinite FP32 output')
            wnorm,wloss=self.weight_norms(module,allow_inference_weight);x=inputs.float();xh=x.half().float()
            if not bool(torch.isfinite(xh).all()):raise ValueError('Input conversion overflow')
            xn=self.norm(x);xloss=self.norm(x-xh);xhn=self.norm(xh)
        if self.profile:
            phase=time.perf_counter();self.counts['phases']['norms']+=phase-start
        if self.preparation is not None:
            try:positions=self.preparation.mask(raw,xn,xloss,xhn,wnorm,wloss,self.noise_factor,self.max_fraction,self.max_entries)
            except ValueError as error:
                if 'density budget exceeded' in str(error):self.counts['density_rejections']+=1
                raise
        else:
            band=(self.noise_factor*xn[:,None]*wnorm[None,:]
                  +xloss[:,None]*wnorm[None,:]+xhn[:,None]*wloss[None,:])
            bits=raw.abs().contiguous().view(torch.int32)
            lower=(bits & -65536).view(torch.float32);upper=((bits & -65536)+65536).view(torch.float32)
            midpoint=(lower+upper)*.5
            positions=((raw.abs()-midpoint).abs()<=band).nonzero()
        fraction=len(positions)/raw.numel()
        if fraction>self.max_fraction or len(positions)>self.max_entries:
            self.counts['density_rejections']+=1;raise ValueError('Sparse repair density budget exceeded')
        output=raw.bfloat16();before=output[positions[:,0],positions[:,1]].clone()
        if self.profile:
            now=time.perf_counter();self.counts['phases']['mask']+=now-phase
        if self.blas_repair is not None:
            phase=time.perf_counter();self.blas_repair.run(module,inputs,positions,output)
            if self.profile:self.counts['phases']['bmm']+=time.perf_counter()-phase
        else:
            # At most4096 pairs and64MiB FP32 gathered operands per batch.
            batch_pairs=min(4096,(64*1024*1024//(8*inputs.shape[1])//self.tile)*self.tile)
            for offset in range(0,len(positions),batch_pairs):
                if self.profile:phase=time.perf_counter()
                selected=positions[offset:offset+batch_pairs];used=len(selected)
                rr,cc=selected[:,0],selected[:,1]
                if self.row_gather is not None:xx,ww=self.row_gather.run(module,inputs,selected)
                else:
                    xx=inputs[rr].float();ww=module.weight[cc].float();padded=(-used)%self.tile
                    if padded:
                        xx=torch.cat((xx,xx[-1:].expand(padded,-1)),0)
                        ww=torch.cat((ww,ww[-1:].expand(padded,-1)),0)
                self.counts['maximum_gather_float32_bytes']=max(self.counts['maximum_gather_float32_bytes'],2*xx.numel()*4)
                xx=xx.reshape(-1,self.tile,inputs.shape[1]);ww=ww.reshape(-1,self.tile,inputs.shape[1])
                if self.profile:
                    now=time.perf_counter();self.counts['phases']['gather']+=now-phase;phase=now
                exact=torch.bmm(xx,ww.transpose(1,2)).diagonal(dim1=-2,dim2=-1).reshape(-1)[:used].bfloat16()
                if self.profile:
                    now=time.perf_counter();self.counts['phases']['bmm']+=now-phase;phase=now
                output[rr,cc]=exact
                if self.profile:self.counts['phases']['scatter']+=time.perf_counter()-phase
        self.counts['calls']+=1;self.counts['entries']+=raw.numel();self.counts['selected']+=len(positions)
        self.counts['changed']+=int((before!=output[positions[:,0],positions[:,1]]).sum())
        self.counts['maximum_fraction']=max(self.counts['maximum_fraction'],fraction)
        self.counts['seconds']+=time.perf_counter()-start
        return output
