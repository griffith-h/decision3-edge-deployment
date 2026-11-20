import concurrent.futures,types,unittest
from request_cost_ledger import Recorder,partition,union

def event(i,parent,start,end,category,thread=1):
    return dict(id=i,parent=parent,start_ns=start,end_ns=end,category=category,thread=thread,layer=0,tag=category)

class LedgerTests(unittest.TestCase):
    def test_nested_and_concurrent_sdk_intervals_close_without_double_counting(self):
        root=event(0,None,0,100,'request_other')
        events=[root,event(1,0,10,90,'projection_other'),event(2,1,20,80,'worker_join'),event(3,None,21,78,'worker_task',2)]
        sdk=[dict(start_ns=30,end_ns=60),dict(start_ns=40,end_ns=70)]
        result=partition(events,root,sdk);rows=result['exclusive_seconds']
        self.assertEqual(result['closure_error_ns'],0);self.assertEqual(result['sum_seconds'],1e-7)
        self.assertAlmostEqual(rows['sdk_runtime_sync'],40e-9)
        self.assertAlmostEqual(rows['worker_join'],20e-9)
        self.assertEqual(union([(1,5),(2,7),(9,10)]),[[1,7],[9,10]])
    def test_real_future_return_exception_and_descriptor_are_restored(self):
        old=concurrent.futures.Future.result;obj=types.SimpleNamespace(call=lambda:7);original=obj.call
        rec=Recorder();rec.wrap(obj,'call','cpu');rec.wrap(concurrent.futures.Future,'result','worker_join');rec.active=True
        try:
            with rec.span('request','request_other'):
                self.assertEqual(obj.call(),7)
                future=concurrent.futures.Future();future.set_exception(ValueError('retained'))
                with self.assertRaises(ValueError):future.result()
            self.assertTrue(any(r.get('error')=='ValueError' for r in rec.events))
        finally:rec.close()
        self.assertIs(obj.call,original);self.assertIs(concurrent.futures.Future.result,old)
    def test_overlap_and_missing_root_are_rejected(self):
        root=event(0,None,0,100,'root')
        with self.assertRaises(ValueError):partition([root,event(1,0,10,60,'a'),event(2,0,50,70,'b')],root,[])
        with self.assertRaises(ValueError):partition([event(1,None,10,20,'a')],root,[])

if __name__=='__main__':unittest.main()
