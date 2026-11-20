import json,unittest
from unittest.mock import patch
from dot_pmu_monitor import owned_peer

class PeerOwnershipTests(unittest.TestCase):
    def inspect(self,label):
        return json.dumps([{'Config':{'Labels':{'decision3.opt.run':label}},'State':{'Pid':100}}])
    def test_owned_descendant_accepted(self):
        with patch('dot_pmu_monitor.subprocess.check_output',return_value=self.inspect('stamp-selected-dot-diagnosis-plain')),patch('dot_pmu_monitor.Path.read_text',return_value='200 (python worker) S 100 0 0'):
            self.assertEqual(owned_peer(200),'stamp-selected-dot-diagnosis-plain')
    def test_unrelated_process_rejected(self):
        with patch('dot_pmu_monitor.subprocess.check_output',return_value=self.inspect('stamp-selected-dot-diagnosis-plain')),patch('dot_pmu_monitor.Path.read_text',return_value='200 (python) S 1 0 0'):
            with self.assertRaisesRegex(RuntimeError,'outside owned'):owned_peer(200)
    def test_other_case_rejected_before_process_lookup(self):
        with patch('dot_pmu_monitor.subprocess.check_output',return_value=self.inspect('stamp-ordinary-plain')),patch('dot_pmu_monitor.Path.read_text') as read:
            with self.assertRaisesRegex(RuntimeError,'owned diagnostic'):owned_peer(200)
            read.assert_not_called()

if __name__=='__main__':unittest.main()
