import time,types,unittest
from concurrent.futures import ThreadPoolExecutor
from native_output_elision_candidate import install

class Input:
    shape=(1,615,64)
    def numel(self):return 615*64
    def reshape(self,*shape):return self
    def contiguous(self):return self
    def data_ptr(self):return 123
class Function:
    def __init__(self,fn):self.fn=fn
    def __call__(self,*args):return self.fn(*args)

class OutputPolicyTests(unittest.TestCase):
    def fixture(self,pool,fail=False):
        members=[object(),object()];done=[];prepared=[]
        def compute(handle,rows):
            if handle==2:time.sleep(.01)
            done.append(handle);return -3 if fail and handle==1 else 0
        state=types.SimpleNamespace(counts={'fused_sparse_mlp_chunks':0},raw_native=types.SimpleNamespace(d3_native_run_checked=Function(compute)))
        lib=types.SimpleNamespace(nm_prepare=lambda *args:prepared.append(('prepare',args)) or 0,nm_copy=lambda *args:prepared.append(('copy',args)) or 0)
        def status(rc):
            if rc:raise ValueError('range guard')
        native=types.SimpleNamespace(ensure=lambda m:members.index(m)+1,lib=lib,status=status)
        owner=types.SimpleNamespace(medium_sparse=state,native_mlp=native,engine=types.SimpleNamespace(pool=pool,bucket={id(m):i for i,m in enumerate(members)}))
        return owner,members,done,prepared
    def test_same_preparation_buckets_join_and_scope_restore(self):
        with ThreadPoolExecutor(2) as pool:
            owner,members,done,prepared=self.fixture(pool);close=install(owner)
            owner.medium_sparse.native_group_no_output(members,Input())
            self.assertEqual(sorted(done),[1,2]);self.assertEqual([x[0] for x in prepared],['prepare','copy'])
            self.assertEqual(owner.medium_sparse.counts['elided_native_outputs'],2)
            close();close();self.assertFalse(hasattr(owner.medium_sparse,'native_compute_no_output'))
    def test_error_still_joins_other_worker_before_propagating(self):
        with ThreadPoolExecutor(2) as pool:
            owner,members,done,_=self.fixture(pool,True);close=install(owner)
            try:
                with self.assertRaisesRegex(ValueError,'range guard'):owner.medium_sparse.native_group_no_output(members,Input())
                self.assertEqual(sorted(done),[1,2]);self.assertEqual(owner.medium_sparse.counts['elided_native_outputs'],0)
            finally:close()

if __name__=='__main__':unittest.main()
