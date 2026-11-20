"""A small probability difference can still change a binary decision."""
import unittest
from summarize import compare

class ComparisonTests(unittest.TestCase):
    def response(self,noul):return {'model':'fixed','usage':{'input_tokens':1},'answers':{'q':{'type':'noul','noul':noul}}}
    def test_threshold_crossing_fails_discrete_gate_even_below_numeric_limit(self):
        result=compare(self.response(.5001),self.response(.4999))
        self.assertLess(result['max_delta'],.01)
        self.assertFalse(result['choices_unchanged']);self.assertFalse(result['noul_predictions_unchanged'])
    def test_same_binary_decision_stays_valid(self):
        self.assertTrue(compare(self.response(.49),self.response(.48))['choices_unchanged'])

if __name__=='__main__':unittest.main()
