import ctypes,torch
from pathlib import Path
lib=ctypes.CDLL(str(Path(__file__).with_name('libconv_btc.so')))
fn=lib.d3_conv_btc;fn.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*3;fn.restype=ctypes.c_int

def install():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    original=qwen.causal_conv1d_fn;counts={'optimized':0,'fallback':0}
    def forward(x,weight,bias=None,activation=None,**kwargs):
        good=(not torch.is_grad_enabled() and x.device.type=='cpu' and x.dtype==torch.bfloat16 and x.ndim==3 and x.shape[0]>0 and x.shape[2]>0 and x.shape[1]%4==0 and
              weight.device==x.device and weight.dtype==x.dtype and weight.shape==(x.shape[1],4) and
              x.stride()==(x.shape[1]*x.shape[2],1,x.shape[1]) and activation in (None,'silu','swish') and all(v is None or (k=="use_cache" and v is False) for k,v in kwargs.items()) and
              (bias is None or (bias.device==x.device and bias.dtype==x.dtype and bias.shape==(x.shape[1],))))
        if not good:
            counts['fallback']+=1;return original(x,weight,bias,activation=activation,**kwargs)
        batch,channels,length=x.shape;out=torch.empty((batch,length,channels),dtype=x.dtype)
        w=weight.contiguous();b=bias.contiguous() if bias is not None else None
        rc=fn(x.data_ptr(),w.data_ptr(),b.data_ptr() if b is not None else None,out.data_ptr(),batch,length,channels)
        if rc:raise RuntimeError('validated BTC conv rejected '+str(rc))
        counts['optimized']+=1
        out=out.transpose(1,2)
        return torch.nn.functional.silu(out) if activation else out
    qwen.causal_conv1d_fn=forward
    def restore():qwen.causal_conv1d_fn=original
    return restore,counts
