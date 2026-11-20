"""Paired real-SDK reader timing; no model or benchmark labels loaded."""
import hashlib,json,statistics,time,unittest
from pathlib import Path
import torch
from test_native_copy_layout import library
import test_native_copy_layout

def run(emit,out):
    emit({'event':'ready','case':'native-copy-preflight','model_loaded':False})
    emit({'event':'request_begin','phase':'native_reader_tests_and_pairs','budget_seconds':120})
    root=Path(__file__).parent
    manifest=json.loads((root/'linear-precision-build.json').read_text())
    for name,digest in manifest['sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Reader build identity mismatch')
    baseline=json.loads((root/'native-reader-baseline.json').read_text())
    if hashlib.sha256((root/'liblinear_precision_before_layout.so').read_bytes()).hexdigest()!=baseline['library_sha256']:
        raise ValueError('Previous reader identity mismatch')
    (out/'native-reader-build-identity.json').write_text(json.dumps({'current':manifest,'previous':baseline},indent=2)+'\n')
    tests=unittest.defaultTestLoader.loadTestsFromModule(test_native_copy_layout)
    checked=unittest.TextTestRunner(verbosity=2).run(tests)
    if not checked.wasSuccessful():raise AssertionError('Native copy layout/rejection checks failed')
    libs={'previous':library(root/'liblinear_precision_before_layout.so'),
          'blocked':library(root/'liblinear_precision_probe.so')}
    records=[]
    for rows,n in ((615,3584),(615,1024),(513,3584)):
        generator=torch.Generator().manual_seed(74)
        x=torch.randn(rows,64,generator=generator,dtype=torch.bfloat16)
        weight=torch.randn(n,64,generator=generator,dtype=torch.float16);packed=weight.t().contiguous()
        handles={};outputs={};times={'previous':[],'blocked':[]}
        try:
            for name,lib in libs.items():
                handle=lib.nm_create(64,n,packed.data_ptr(),1)
                if not handle:raise AssertionError('Native reader benchmark context create failed')
                handles[name]=handle
                if lib.nm_prepare(handle,rows,x.data_ptr()) or lib.nm_raw(handle,rows):raise AssertionError('Native reader fixture failed')
                outputs[name]=torch.empty(rows,n,dtype=torch.float32)
                if lib.d3_probe_copy_fp32(handle,rows,outputs[name].data_ptr()):raise AssertionError('Native reader warm copy failed')
            if not torch.equal(outputs['previous'].view(torch.int32),outputs['blocked'].view(torch.int32)):
                raise AssertionError('Reader outputs differ bitwise')
            for pair in range(7):
                for name in (('previous','blocked') if pair%2==0 else ('blocked','previous')):
                    start=time.perf_counter()
                    for _ in range(5):
                        if libs[name].d3_probe_copy_fp32(handles[name],rows,outputs[name].data_ptr()):raise AssertionError('Native copy pair failed')
                    times[name].append((time.perf_counter()-start)/5)
            if not torch.equal(outputs['previous'].view(torch.int32),outputs['blocked'].view(torch.int32)):
                raise AssertionError('Repeated reader outputs differ bitwise')
            ratio=statistics.median(b/a for a,b in zip(times['previous'],times['blocked']))
            records.append({'shape':[rows,n],'times':times,'median_pair_ratio':ratio,'bitwise_equal':True})
        finally:
            for name,handle in handles.items():libs[name].nm_destroy(handle)
    data={'completed':True,'scope':'Real SDK reader component only, no model speed/accuracy result',
          'tests_run':checked.testsRun,'records':records,
          'qualified_for_model_test':all(r['median_pair_ratio']<=.8 for r in records)}
    (out/'native-copy-preflight.json').write_text(json.dumps(data,indent=2)+'\n')
    emit({'event':'complete','qualified_for_model_test':data['qualified_for_model_test'],'model_loaded':False})
