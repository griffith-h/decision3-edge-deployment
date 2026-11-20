"""Independent-row FP32-dense reference preserving original batch left padding.

The steps mirror the pinned D3.logits encoder/readout. This is a correctness
oracle, not a new performance baseline or a change to the candidate engine.
"""
from types import MethodType
import torch
from d3_runtime import sdpa_backends


def install(model):
    original=model.logits
    state={'width':None,'records':[]}
    assert model.encoder is not None and model.readout_dtype=='float32' and not model.permutation_average
    def logits(model,sequences,counts):
        assert len(sequences)==len(counts)==1
        width=state['width'];sequence=sequences[0]
        assert width>=len(sequence)>0
        ids=torch.full((1,width),model.pad_id,dtype=torch.long)
        present=torch.zeros((1,width),dtype=torch.long)
        ids[0,width-len(sequence):]=torch.as_tensor(sequence,dtype=torch.long)
        present[0,width-len(sequence):]=1
        ids=ids.to(model.device);present=present.to(model.device)
        with torch.inference_mode(),sdpa_backends(model.device):
            hidden=model.encoder(model.backbone.get_input_embeddings()(ids),present)[:,-1]
            values=hidden.float()@model.readout.T
            invalid=torch.arange(values.shape[1],device=model.device)[None]>=torch.as_tensor(counts,device=model.device)[:,None]
            result=values.masked_fill(invalid,float('-inf'))
        state['records'].append({'effective_tokens':len(sequence),'original_batch_width':width})
        return result
    wrapper=MethodType(logits,model);model.logits=wrapper
    def close():
        if model.logits==wrapper:model.logits=original
    return state,close
