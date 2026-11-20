"""Component screening on fixed synthetic equivalents of the two slow shapes."""
import json,resource,time
import torch
import torch.nn.functional as F
from board import operating_state
from long_npu_attention_candidate import LongNpuAttention

def run(owner,emit,out):
    # Joined owner-thread boundary; avoid retaining idle half banks during a
    # memory-bounded component sweep. No model request or gold is changed.
    owner.retire('half');records=[]
    for batch,length,tiles in ((2,2311,(128,512,1024,2048)),(1,7232,(2048,1024,512,128))):
        generator=torch.Generator().manual_seed(8008+length)
        def tensor(heads):return (torch.randn((batch,length,heads,256),generator=generator)*.25).bfloat16().transpose(1,2)
        q,k,v=tensor(8),tensor(2),tensor(2)
        mask=torch.ones((batch,1,1,length),dtype=torch.bool);mask[...,:7]=False
        emit({'event':'request_begin','phase':'CPU reference','batch':batch,'length':length})
        with torch.inference_mode():
            start=time.perf_counter()
            reference=F.scaled_dot_product_attention(q.float(),k.float().repeat_interleave(4,1),v.float().repeat_interleave(4,1),attn_mask=mask,dropout_p=0.,is_causal=False,scale=1/16).transpose(1,2).bfloat16()
            seconds=time.perf_counter()-start
        record={'kind':'CPU component reference','batch':batch,'length':length,'seconds':seconds}
        records.append(record);emit({'event':'case_complete',**record})
        for tile in tiles:
            algorithm=LongNpuAttention(owner,tile=tile)
            try:
                for repeat in range(3):
                    condition=operating_state()
                    emit({'event':'request_begin','tile':tile,'repeat':repeat,'batch':batch,'length':length})
                    with torch.inference_mode():
                        start=time.perf_counter();actual=algorithm.kernel(q,k,v,mask,1/16);seconds=time.perf_counter()-start
                        delta=actual.float()-reference.float()
                        maximum=delta.abs().max().item()
                        relative=(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(reference.float())).item()
                    record={'kind':'NPU component','batch':batch,'length':length,'tile':tile,'repeat':repeat,
                        'cold_context':repeat==0,'seconds':seconds,'max_delta':maximum,'relative_l2':relative,
                        'passed':maximum<=.002 and relative<=.01,'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        'operating_state':condition}
                    records.append(record)
                    (out/'tile-sweep.json').write_text(json.dumps({'completed':False,'records':records})+'\n')
                    emit({'event':'case_complete',**{k:record[k] for k in ('batch','length','tile','repeat','seconds','passed')}})
                    if not record['passed']:raise RuntimeError('Component numerical gate failed')
            finally:algorithm.close()
            assert not algorithm.handles
    (out/'tile-sweep.json').write_text(json.dumps({'completed':True,'records':records})+'\n')
    emit({'event':'complete','component_records':len(records),'scope':'synthetic shape screening; not model speed'})
