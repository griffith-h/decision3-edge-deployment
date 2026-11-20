"""HTTP verification must send the same inputs and enforce binary decisions."""
import unittest
from verify_final_http import compare,request_payload

class HttpVerificationTests(unittest.TestCase):
    def test_multimodal_inputs_preserved_without_expected_output(self):
        request={'state':'synthetic','questions':{'q':{}},'images':['synthetic-image'],
                 'videos':['synthetic-video'],'id':'local-case'}
        payload=request_payload({'request':request,'output':{'private':'reference'}})
        self.assertEqual(set(payload),{'state','questions','images','videos'})
        self.assertEqual(payload['images'],request['images']);self.assertEqual(payload['videos'],request['videos'])
    def test_binary_threshold_change_cannot_pass_on_small_numeric_delta(self):
        actual={'answers':{'q':{'noul':.5001}}};expected={'answers':{'q':{'noul':.4999}}}
        parity=compare(actual,expected)
        self.assertLess(parity['max_numeric_delta'],.01);self.assertFalse(parity['choices_unchanged'])

if __name__=='__main__':unittest.main()
