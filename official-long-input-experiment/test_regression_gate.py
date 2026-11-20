"""Changed medium-domain arithmetic cannot weaken the original0.01 gate."""
import unittest
from check_regression import check_pair

def response(value,tokens=1000):
    return {'model':'fixed','usage':{'input_tokens':tokens},'answers':{'q':{'type':'score','score':value}}}

class RegressionGateTests(unittest.TestCase):
    def test_original_limit_still_rejects_even_if_previous_native_is_close(self):
        with self.assertRaisesRegex(AssertionError,'original numerical gate'):
            check_pair(response(.511),{'native':response(.502),'current':response(.5)})
    def test_changed_arithmetic_reports_both_references(self):
        previous,_,original,_=check_pair(response(.505),{'native':response(.501),'current':response(.5)})
        self.assertAlmostEqual(previous['max_delta'],.004);self.assertAlmostEqual(original['max_delta'],.005)
    def test_protected_fast_shapes_still_require_identity(self):
        for tokens in (115,615):
            with self.assertRaisesRegex(AssertionError,'protected native identity'):
                check_pair(response(.501,tokens),{'native':response(.5,tokens),'current':response(.5005,tokens)})

if __name__=='__main__':unittest.main()
