"""One real original-B8 projection against the actual current CPU forward.

Intentionally stop at the first projection group; not a complete request or
accuracy result. All inputs and original padding come from the frozen case.
"""
import json,time
import torch

class ProbeComplete(Exception):pass

def run(model,owner,request,emit,out,cpu_geometry=False):
    old=owner.projections.run;records=[]
    def inspect(members,x,key):
        if tuple(x.shape)!=(8,1181,1024):raise ValueError('Unexpected frozen first-batch geometry')
        start=time.perf_counter();actual=old(members,x,key);accelerated=time.perf_counter()-start
        for module in members:
            name=next(n for n,m in owner.projections.modules.items() if m is module)
            begin=time.perf_counter();expected=module._bp_original(x);seconds=time.perf_counter()-begin
            got=actual[id(module)];different=got!=expected
            records.append({'module':name,'input_shape':list(x.shape),'output_shape':list(got.shape),
                'cpu_seconds_correctness_only':seconds,'accelerated_group_seconds':accelerated,
                'bf16_mismatches':int(different.sum()),'meaningful_mismatches':int((different & (expected.abs()>1e-20)).sum()),
                'maximum_delta':float((got.float()-expected.float()).abs().max())})
            if cpu_geometry:
                # Diagnose the actual enabled-backend CPU reduction. Never
                # install these partial calculations as a candidate repair.
                import torch.nn.functional as f
                if not torch.backends.mkldnn.enabled:raise AssertionError('Original CPU backend changed')
                flat=x.reshape(-1,x.shape[-1]);target=expected.reshape(-1,module.out_features)
                positions=(got.reshape_as(target)!=target).nonzero()
                selected=positions[torch.linspace(0,len(positions)-1,min(512,len(positions))).long()]
                if not len(selected):raise AssertionError('Expected preserved mismatch sample')
                rr,cc=selected.unbind(1);xx=flat[rr].float();ww=module.weight[cc].float()
                expected_points=target[rr,cc]
                def measured(name,fn):
                    begin=time.perf_counter();value=fn().bfloat16()
                    return {'method':name,'points':len(value),'bf16_mismatches':int((value!=expected_points).sum()),
                            'max_delta':float((value.float()-expected_points.float()).abs().max()),
                            'seconds_correctness_only':time.perf_counter()-begin}
                methods=[measured('fp64_sum',lambda:(xx.double()*ww.double()).sum(-1)),
                         measured('fp32_sum',lambda:(xx*ww).sum(-1))]
                for tile in (1,4,16):
                    methods.append(measured('torch_bmm_'+str(tile),lambda tile=tile:torch.bmm(
                        xx.reshape(-1,tile,xx.shape[-1]),ww.reshape(-1,tile,ww.shape[-1]).transpose(1,2)
                    ).diagonal(dim1=-2,dim2=-1).reshape(-1)))
                methods.append(measured('torch_linear_selected_square',lambda:f.linear(xx,ww).diagonal()))
                from types import SimpleNamespace
                from sparse_cpu_backend import DiagonalRepair
                native_module=SimpleNamespace(weight=ww.bfloat16())
                diagonal=torch.arange(len(selected));native_positions=torch.stack((diagonal,diagonal),1)
                for block in (352,1024):
                    native=DiagonalRepair(block)
                    def native_points(native=native):
                        result=torch.full((len(selected),len(selected)),77.,dtype=torch.bfloat16)
                        native.run(native_module,xx.bfloat16(),native_positions,result)
                        return result.diagonal()
                    methods.append(measured('native_diagonal_q'+str(block),native_points))
                torch.save({'inputs':xx.bfloat16(),'weights':ww.bfloat16(),'expected':expected_points,
                            'scope':'Selected real intermediate CPU reduction diagnostic, never production inputs'},
                           out/'cpu-reduction-sample.pt')
                begin=time.perf_counter();flat_result=f.linear(flat.float(),module._cpu_fp32_weight,module._cpu_fp32_bias).bfloat16()
                flat_check={'bf16_mismatches':int((flat_result!=target).sum()),
                            'seconds_correctness_only':time.perf_counter()-begin}
                del flat_result
                (out/'cpu-reduction-geometry.json').write_text(json.dumps({'completed':True,'full_request_completed':False,
                    'mkldnn':torch.backends.mkldnn.enabled,'module':name,'original_shape':list(x.shape),
                    'sample_source':'Evenly spaced mismatching intermediate positions; diagnostic only, no labels',
                    'methods':methods,'flat_full_projection_vs_original_3d':flat_check})+'\n')
                break
            del expected,different
        (out/'first-projection.json').write_text(json.dumps({'completed':True,'full_request_completed':False,
            'scope':'One real original8x1181 first projection group; CPU forward at identical B/T, partial model only','records':records})+'\n')
        raise ProbeComplete
    owner.projections.run=inspect
    emit({'event':'request_begin','phase':'first_original_B8_projection'})
    try:
        try:model.system_one(**request)
        except ProbeComplete:pass
        else:raise AssertionError('Diagnostic did not reach expected projection')
    finally:owner.projections.run=old
    assert len(records)==(1 if cpu_geometry else 2)
    emit({'event':'complete','primitive_modules':len(records),'full_request_completed':False})
