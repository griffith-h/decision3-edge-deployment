"""Composition ownership and failure paths, without loading a second model."""
import types,unittest
from long_input_owner import Owner

class FactoryTests(unittest.TestCase):
    def fixture(self):
        events=[]
        base=types.SimpleNamespace(model=object(),use=lambda mode:events.append(mode),
            close=lambda:events.append('base_close'),counts=lambda:{'base':1},
            long_mlp=types.SimpleNamespace(counts={'optimized':1}))
        def install(name,fail=False,close_fail=False):
            def call(owner):
                self.assertIs(owner,base);events.append(name)
                if fail:raise RuntimeError('install failed')
                def close():
                    events.append(name+'_close')
                    if close_fail:raise RuntimeError('close failed')
                return close
            return call
        return events,base,install
    def test_order_counts_snapshot_and_idempotent_close(self):
        events,base,install=self.fixture()
        owner=Owner(lambda:base,[install('a'),install('b')]);self.assertIs(owner.model,base.model)
        counts=owner.counts();base.long_mlp.counts['optimized']=2
        self.assertEqual(counts['long_input']['long_mlp']['optimized'],1)
        owner.use('candidate');owner.close();owner.close()
        self.assertEqual(events,['candidate','a','b','b_close','a_close','base_close'])
        with self.assertRaises(ValueError):owner.use('candidate')
    def test_partial_install_restores_successful_adapters_and_base(self):
        for close_fail in (False,True):
            events,base,install=self.fixture()
            with self.assertRaisesRegex(RuntimeError,'install failed'):Owner(lambda:base,[install('a',close_fail=close_fail),install('b',fail=True)])
            self.assertEqual(events,['candidate','a','b','a_close','base_close'])
    def test_restore_error_does_not_skip_other_owners(self):
        events,base,install=self.fixture();owner=Owner(lambda:base,[install('a'),install('b',close_fail=True)])
        with self.assertRaisesRegex(RuntimeError,'close failed'):owner.close()
        self.assertEqual(events[-3:],['b_close','a_close','base_close']);owner.close()

if __name__=='__main__':unittest.main()
