"""CPU-only replay of preserved first differing projection, no model owner/NPU."""
import json,math
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from safetensors import safe_open
from stages import MODEL

def run(emit,out,owner=None):
    torch.set_num_threads(3)
    from sparse_cpu_backend import BlasRepair,DiagonalRepair
    item=torch.load('/inputs/first-projection-difference.pt',map_location='cpu',weights_only=True)
    x=item['input'];got=item['candidate'];expected=item['expected'];name=item['module'];context={}
    with safe_open(str(Path(MODEL)/'model-00001-of-00001.safetensors'),framework='pt',device='cpu') as f:
        weight=f.get_tensor(name+'.weight')
        qkv=f.get_tensor(name.replace('in_proj_z','in_proj_qkv')+'.weight')
    flat=x.reshape(-1,x.shape[-1]);diff=(got.reshape(-1,got.shape[-1])!=expected.reshape(-1,expected.shape[-1])).nonzero()
    if len(diff)!=1:raise ValueError('Expected the preserved single projection mismatch')
    row,col=map(int,diff[0]);module=SimpleNamespace(weight=weight,bias=None);records=[]
    if owner is not None:
        import inspect
        actual_module=owner.projections.modules[name]
        context={'original_function_module':actual_module._bp_original.__module__,
                 'original_function_source':inspect.getsource(actual_module._bp_original),
                 'weight_dtype':str(actual_module.weight.dtype),'weight_stride':list(actual_module.weight.stride()),
                 'checkpoint_weight_matches':bool(torch.equal(actual_module.weight,weight)),
                 'cached_weight_matches':bool(torch.equal(actual_module._cpu_fp32_weight,weight.float())),
                 'cached_weight_stride':list(actual_module._cpu_fp32_weight.stride())}
        module=actual_module
    context.update({'torch_threads':torch.get_num_threads(),'mkldnn':torch.backends.mkldnn.enabled,
                    'float32_precision':torch.get_float32_matmul_precision(),'input_stride':list(x.stride()),
                    'weight_source_dtype':str(weight.dtype)})
    def val(kind,value):
        v=value.detach().reshape(()).float();records.append({'kind':kind,'fp32':float(v),'fp32_bits':int(v.view(torch.int32)),
            'bf16':float(v.bfloat16())})
    def save(completed=False):
        (out/'cpu-point-comparison.json').write_text(json.dumps({'completed':completed,'full_request_completed':False,
            'scope':'Preserved layer4 projection at one differing position; no inference NPU path used','runtime_context':context,
            'module':name,'input_shape':list(x.shape),'position':[row,col],'candidate_bf16':float(got.reshape(-1,got.shape[-1])[row,col]),
            'saved_cpu_bf16':float(expected.reshape(-1,expected.shape[-1])[row,col]),'records':records})+'\n')
    emit({'event':'ready','case':'projection-single-point','variant':'cpu-only'})
    with torch.inference_mode():
        emit({'event':'request_begin','phase':'original_full_cpu_projection'})
        full=F.linear(x.float(),weight.float());rounded=full.bfloat16()
        mismatches=int((rounded!=expected).sum())
        context['direct_cpu_mismatches_vs_preserved']=mismatches
        if owner is not None:
            runtime=module._bp_original(x)
            context['runtime_cpu_mismatches_vs_preserved']=int((runtime!=expected).sum())
            context['runtime_cpu_value']=float(runtime.reshape(-1,runtime.shape[-1])[row,col]);del runtime
        context['blas_configuration']=BlasRepair().configuration
        context['blas_threads']=BlasRepair().threads()
        val('original_full_batch_cpu',full.reshape(-1,full.shape[-1])[row,col]);del full,rounded;save()
        emit({'event':'request_begin','phase':'cpu_geometry_and_repair'})
        for rows in (1,4,16,128,512,615):
            start=row//rows*rows;part=flat[start:start+rows]
            result=F.linear(part.float(),weight.float());val('cpu_rows_'+str(len(part)),result[row-start,col])
        exact=math.fsum((flat[row].double()*weight[col].double()).tolist())
        records.append({'kind':'fp64_fsum','value':exact,'bf16':float(torch.tensor(exact,dtype=torch.float64).bfloat16())})
        one=flat[row:row+1];positions=torch.tensor([[0,col]],dtype=torch.long)
        for kind,repair in (('public_cblas4x4',BlasRepair()),('diagonal_q352',DiagonalRepair())):
            output=torch.zeros((1,weight.shape[0]),dtype=torch.bfloat16);repair.run(module,one,positions,output)
            records.append({'kind':kind,'bf16':float(output[0,col])})
        packed=torch.cat((qkv,weight),0)
        for rows in (512,9448):
            start=row//rows*rows;part=flat[start:start+rows]
            result=F.linear(part.float(),packed.float());val('cpu_packed8192_rows_'+str(len(part)),result[row-start,qkv.shape[0]+col]);del result
        save(True)
    emit({'event':'complete','diagnostic_positions':1,'full_request_completed':False,'npu_owner_loaded':owner is not None})
