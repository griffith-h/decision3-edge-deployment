"""Bank ownership and precision-mode restoration, using no native resources."""
import types,unittest
from long_projection_fp32_candidate import PreciseProjection

def fixture(fail=False):
    mixed=types.SimpleNamespace(mode=0)
    def precision(mode):old=mixed.mode;mixed.mode=mode;return old
    mixed.mm_set_output_precision=precision
    existing=types.SimpleNamespace(weight='existing');new=types.SimpleNamespace(weight='new')
    engine=types.SimpleNamespace(ensure=lambda m:'old-'+m.weight,core={},handles=['half','float-existing'])
    def create(weight,core):
        assert mixed.mode==1
        if fail:raise RuntimeError('injected create failure')
        engine.handles.append('float-'+weight);return engine.handles[-1]
    engine.create=create
    model=types.SimpleNamespace(logits=lambda seq,counts:[engine.ensure(existing),engine.ensure(new)])
    owner=types.SimpleNamespace(model=model,engine=engine,mixed=mixed,fused=True,
        float_handles={id(existing):'float-existing'},cache_retired={'half':0},
        projections=types.SimpleNamespace(good=lambda members,x:True,max_rows=2048,width=512,mode='chunked'))
    def retire(kind):
        assert kind=='half'
        if 'half' in engine.handles:engine.handles.remove('half');owner.cache_retired['half']+=1
    owner.retire=retire
    return owner

class PrecisionTests(unittest.TestCase):
    def test_long_scope_reuses_float_and_retires_only_inactive_half_bank(self):
        owner=fixture();old=owner.engine.ensure;scope=PreciseProjection(owner)
        actual=owner.model.logits([range(7232)],[2])
        self.assertEqual(actual,['float-existing','float-new'])
        self.assertEqual(owner.engine.handles,['float-existing','float-new'])
        self.assertEqual(scope.counts['fp32_created'],1);self.assertEqual(scope.counts['retired_half'],1)
        self.assertEqual(owner.mixed.mode,0);self.assertFalse(scope.active)
        scope.close();scope.close();self.assertIs(owner.engine.ensure,old)
    def test_short_scope_does_not_touch_bank_or_precision(self):
        owner=fixture();scope=PreciseProjection(owner)
        self.assertEqual(owner.model.logits([range(615)],[2]),['old-existing','old-new'])
        self.assertEqual(owner.engine.handles,['half','float-existing']);self.assertEqual(scope.counts['scopes'],0)
        scope.close()
    def test_create_failure_restores_precision_and_scope_then_owner_keeps_handles(self):
        owner=fixture(True);old=owner.engine.ensure;scope=PreciseProjection(owner)
        with self.assertRaisesRegex(RuntimeError,'create failure'):owner.model.logits([range(7232)],[2])
        self.assertEqual(owner.mixed.mode,0);self.assertEqual(owner.projections.max_rows,2048)
        self.assertTrue(owner.fused);self.assertFalse(scope.active)
        self.assertEqual(owner.engine.handles,['float-existing']);scope.close();self.assertIs(owner.engine.ensure,old)

if __name__=='__main__':unittest.main()
