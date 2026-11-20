"""CPU-only native row-copy verification; matrix arithmetic remains TorchBMM."""
import json,time,unittest
import torch
import torch.nn.functional as f
from sparse_rounding import SparseRepair
import test_sparse_cpu_backend

def run():
    torch.set_num_threads(3);torch.manual_seed(19)
    checked=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_sparse_cpu_backend))
    assert checked.wasSuccessful()
    rows=[]
    for columns in (2048,6144):
        module=torch.nn.Linear(1024,columns,bias=False,dtype=torch.bfloat16);x=torch.randn(512,1024,dtype=torch.bfloat16)
        raw=f.linear(x.float(),module.weight.float())
        original=SparseRepair(backend='torch');expected=original.repair(module,x,raw)
        repair=SparseRepair();measurements=[]
        for repeat in range(2):
            start=time.perf_counter();result=repair.repair(module,x,raw);seconds=time.perf_counter()-start
            assert torch.equal(result,expected),'Native gather changed identical TorchBMM arithmetic'
            measurements.append(seconds)
        rows.append({'columns':columns,'seconds':measurements,'torch_gather_total_seconds':original.counts['seconds'],'counts':repair.counts})
    return {'passed':True,'scope':'Synthetic row-copy/bounded-memory preflight only, no model or NPU; all arithmetic retained in TorchBMM',
      'unit_checks':checked.testsRun,'records':rows}

if __name__=='__main__':print(json.dumps(run()))
