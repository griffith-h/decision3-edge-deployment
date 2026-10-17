"""Focused contract tests, independent of the closed native Runtime."""
import json
import shutil
from pathlib import Path
import tempfile
import unittest
from patches.d3_export_adapter import D3Contract, UnsupportedNativeRuntime, require_native_runtime, write_contract


class Writer:
    def __init__(self): self.fields={}
    def add_causal_attention(self,value): self.fields['custom.attention.causal']=value
    def add_string(self,key,value): self.fields[key]=value
    def add_array(self,key,value): self.fields[key]=value
    def add_bool(self,key,value): self.fields[key]=value
    def add_uint32(self,key,value): self.fields[key]=value


class AdapterTest(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parent.parent/'rkllm-experiment/source-model'
        self.contract=D3Contract.from_model(root)

    def test_original_12_layer_layout_retained(self):
        self.assertEqual(len(self.contract.layer_types),12)
        self.assertEqual(self.contract.layer_types[9:12],('full_attention',)*3)
        self.assertEqual(self.contract.layer_types.count('linear_attention'),6)

    def test_distinct_full_and_linear_attention_flags(self):
        writer=Writer(); write_contract(writer,self.contract)
        self.assertIs(writer.fields['custom.attention.causal'],False)
        self.assertIs(writer.fields['d3.contract.linear_attention_causal'],True)
        self.assertEqual(writer.fields['d3.contract.layer_types'],[int(k=='full_attention') for k in self.contract.layer_types])
        self.assertEqual(writer.fields['d3.contract.prefill_order'],1)

    def test_current_native_runtime_rejected(self):
        with self.assertRaisesRegex(UnsupportedNativeRuntime,'noncausal_full_attention'):
            require_native_runtime({'last_hidden_layer':True})

    def test_partial_capabilities_do_not_enable_native_execution(self):
        with self.assertRaisesRegex(UnsupportedNativeRuntime,'layer_major_full_sequence_prefill'):
            require_native_runtime({'last_hidden_layer':True,'noncausal_full_attention':True,'exact_per_layer_layout':True})

    def test_requires_native_implementation_marker(self):
        self.assertIs(self.contract.as_dict()['requires_native_runtime_implementation'],True)

    def test_modified_checkpoint_config_rejected(self):
        source=Path(__file__).resolve().parent.parent/'rkllm-experiment/source-model'
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)
            for name in ['config.json','decision_config.json','MODEL_MANIFEST.json']:
                shutil.copyfile(source/name,target/name)
            config=json.loads((target/'config.json').read_text())
            config['text_config']['num_hidden_layers']=11
            (target/'config.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'integrity'):
                D3Contract.from_model(target)


if __name__=='__main__': unittest.main(verbosity=2)
