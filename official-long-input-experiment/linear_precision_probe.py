"""Real first-layer FP32 outputs: isolate conversion from accumulation/rounding."""
import ctypes,hashlib,json,time
from pathlib import Path
import torch
import torch.nn.functional as f

class ProbeComplete(Exception):pass

def run(model,owner,cases,emit,out):
    import unittest,test_sparse_rounding
    tests=unittest.defaultTestLoader.loadTestsFromModule(test_sparse_rounding)
    if not unittest.TextTestRunner(verbosity=2).run(tests).wasSuccessful():raise AssertionError('Sparse repair lifecycle checks failed')
    root=Path(__file__).parent;manifest=json.loads((root/'linear-precision-build.json').read_text())
    for name,digest in manifest['sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,'Precision probe source/library mismatch'
    lib=ctypes.CDLL(str(root/'liblinear_precision_probe.so'))
    lib.nm_create.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int];lib.nm_create.restype=ctypes.c_void_p
    lib.nm_destroy.argtypes=[ctypes.c_void_p];lib.nm_error.restype=ctypes.c_char_p
    for name in ('nm_prepare','d3_probe_copy_fp32'):
        getattr(lib,name).argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p];getattr(lib,name).restype=ctypes.c_int
    lib.nm_raw.argtypes=[ctypes.c_void_p,ctypes.c_int];lib.nm_raw.restype=ctypes.c_int
    records=[];handles=0
    def status(rc):
        if rc:raise AssertionError('Precision probe SDK failure: '+repr(lib.nm_error())+' rc='+str(rc))
    def save(completed=False):
        (out/'linear-precision.json').write_text(json.dumps({'completed':completed,'live_handles':handles,
          'scope':'Real first-layer primitive, first512 rows. Partial model intentionally stopped; not a full request or benchmark score.',
          'records':records})+'\n')
    # Exercise the native-layout FP32 copy with exact, non-square tiny matrices
    # before reading any real model output. Every value is a power-of-two fraction.
    for rows in (3,129):
        source=(torch.arange(rows*64).reshape(rows,64)%7).bfloat16()/8
        weights=(torch.arange(32*64).reshape(32,64)%5).half()/16
        target=torch.empty((rows,32),dtype=torch.float32)
        packed_weights=weights.t().contiguous()
        handle=lib.nm_create(64,32,packed_weights.data_ptr(),1)
        assert handle,'Synthetic create failed';handles+=1
        try:
            status(lib.nm_prepare(handle,rows,source.contiguous().data_ptr()));status(lib.nm_raw(handle,rows))
            status(lib.d3_probe_copy_fp32(handle,rows,target.data_ptr()))
            assert torch.equal(target,f.linear(source.float(),weights.float())),'FP32 copy/layout test failed'
        finally:lib.nm_destroy(handle);handles-=1
    (out/'linear-precision-unit.json').write_text(json.dumps({'passed':True,'cases':2,'live_handles':handles})+'\n')
    original=owner.projections.run
    def inspect(members,x,key):
        nonlocal handles
        assert x.dtype==torch.bfloat16 and x.shape==(2,1193,1024)
        part=x.reshape(-1,x.shape[-1])[:512].contiguous()
        for module in members:
            name=next(n for n,v in owner.projections.modules.items() if v is module)
            assert module.bias is None
            expected=f.linear(part.float(),module.weight.float());rounded=expected.bfloat16()
            start_cpu=time.perf_counter();cpu_rounded=module._bp_original(part);cpu_seconds=time.perf_counter()-start_cpu
            assert torch.equal(rounded,cpu_rounded)
            for factor in (1.,16.,256.):
                emit({'event':'request_begin','module':name,'factor':factor})
                scaled=(part*factor).contiguous();weights=(module.weight.float()*factor).half().t().contiguous()
                assert bool(torch.isfinite(scaled.half()).all()) and bool(torch.isfinite(weights).all())
                quantized=f.linear(scaled.half().float(),weights.t().float())/(factor*factor)
                handle=lib.nm_create(part.shape[1],module.out_features,weights.data_ptr(),1)
                assert handle,'Precision probe create failed';handles+=1
                try:
                    status(lib.nm_prepare(handle,512,scaled.data_ptr()));status(lib.nm_raw(handle,512))
                    actual=torch.empty_like(expected);status(lib.d3_probe_copy_fp32(handle,512,actual.data_ptr()))
                    actual/=factor*factor
                finally:lib.nm_destroy(handle);handles-=1
                got=actual.bfloat16();different=(got!=rounded)&(rounded!=0)
                indices=different.nonzero()[:20];examples=[]
                for row,column in indices.tolist():
                    exact=float((part[row].double()*module.weight[column].double()).sum())
                    exact_bf16=float(torch.tensor(exact,dtype=torch.float64).bfloat16())
                    examples.append({'row':row,'column':column,'cpu_fp32':float(expected[row,column]),'sdk_fp32':float(actual[row,column]),
                      'cpu_bf16':float(rounded[row,column]),'sdk_bf16':float(got[row,column]),'exact_fp64':exact,'exact_bf16':exact_bf16})
                bits=actual.contiguous().view(torch.int32)&65535
                row={'module':name,'factor_input_and_weight':factor,'input_shape':list(part.shape),'output_shape':list(actual.shape),
                  'sdk_fp32_max_delta':float((actual-expected).abs().max()),
                  'sdk_bf16_mismatch_fraction':float((got!=rounded).float().mean()),
                  'sdk_bf16_nonzero_mismatches':int(different.sum()),
                  'conversion_cpu_fp32_max_delta':float((quantized-expected).abs().max()),
                  'conversion_cpu_bf16_mismatches':int((quantized.bfloat16()!=rounded).sum()),
                  'near_bf16_midpoint_counts':{str(window):int(((bits-32768).abs()<=window).sum()) for window in (32,64,256,1024)},
                  'first20_nonzero_mismatch_double_checks':examples}
                row['cpu_zero_sdk_nonzero']=int(((rounded==0)&(got!=0)).sum())
                if factor==256.:
                    repairs=[]
                    for window in (64,256):
                        positions=((bits-32768).abs()<=window).nonzero();corrected=got.clone()
                        for selected in positions.split(512):
                            rr,cc=selected[:,0],selected[:,1]
                            exact=(part[rr].double()*module.weight[cc].double()).sum(-1).bfloat16()
                            corrected[rr,cc]=exact
                        repairs.append({'window_fp32_ulps':window,'recomputed_entries':len(positions),
                          'bf16_mismatches_vs_cpu':int((corrected!=rounded).sum()),
                          'nonzero_mismatches_vs_cpu':int(((corrected!=rounded)&(rounded!=0)).sum()),
                          'max_bf16_delta_vs_cpu':float((corrected.float()-rounded.float()).abs().max())})
                    row['diagnostic_sparse_fp64_repairs']=repairs
                    # Test whether small CPU GEMMs reproduce the canonical CPU
                    # accumulation better than a mathematically exact double dot.
                    sample=((got!=rounded)&(expected.abs()>1e-20)).nonzero()[:256]
                    strategies=[]
                    for tile in (1,4,8,16,32):
                        values=[]
                        for selected in sample.split(tile):
                            rr,cc=selected[:,0],selected[:,1]
                            xx=part[rr].float();ww=module.weight[cc].float()
                            if len(selected)<tile:
                                xx=torch.cat((xx,xx[-1:].expand(tile-len(selected),-1)),0)
                                ww=torch.cat((ww,ww[-1:].expand(tile-len(selected),-1)),0)
                            values.append(f.linear(xx,ww).diagonal()[:len(selected)])
                        if values:
                            value=torch.cat(values);target=expected[sample[:,0],sample[:,1]]
                            strategies.append({'square_cpu_gemm_tile':tile,'entries':len(sample),
                               'exact_fp32_matches':int((value==target).sum()),
                               'bf16_mismatches':int((value.bfloat16()!=target.bfloat16()).sum()),
                               'fp32_max_delta':float((value-target).abs().max())})
                    row['selective_cpu_gemm_strategies']=strategies
                if factor==1.:
                    from sparse_rounding import SparseRepair
                    repair=SparseRepair();fixed=repair.repair(module,part,actual)
                    row['sparse_cpu_repair']={'original_cpu_dense_seconds_diagnostic_only':cpu_seconds,'counts':repair.counts,'bf16_mismatches_vs_cpu':int((fixed!=rounded).sum()),
                      'meaningful_mismatches_vs_cpu':int(((fixed!=rounded)&(expected.abs()>1e-20)).sum()),
                      'max_bf16_delta_vs_cpu':float((fixed.float()-rounded.float()).abs().max())}
                records.append(row);save();emit({'event':'case_complete','module':name,'factor':factor})
        save(True);raise ProbeComplete
    owner.projections.run=inspect
    try:
        try:model.system_one(**cases['diagnostic-home-pair'])
        except ProbeComplete:pass
        else:raise AssertionError('Probe did not reach intended first projection')
    finally:owner.projections.run=original
    assert len(records)==6 and handles==0
    emit({'event':'complete','primitive_modes':len(records),'full_request_completed':False})
