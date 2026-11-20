import struct,unittest
from dot_pmu_monitor import decode_group

class PmuReadTests(unittest.TestCase):
    def test_group_counts_and_multiplex_fraction_are_preserved(self):
        r=decode_group(struct.pack('5Q',2,100,80,1234,567),('cpu_cycles','inst_retired'))
        self.assertEqual(r['counts'],{'cpu_cycles':1234,'inst_retired':567});self.assertEqual(r['running_fraction'],.8)
        self.assertEqual(decode_group(struct.pack('4Q',1,0,0,0),('cpu_cycles',))['running_fraction'],1.)
    def test_incomplete_and_wrong_shape_are_rejected(self):
        with self.assertRaises(ValueError):decode_group(b'',('cpu_cycles',))
        with self.assertRaises(ValueError):decode_group(struct.pack('4Q',2,1,1,9),('cpu_cycles',))

if __name__=='__main__':unittest.main()
