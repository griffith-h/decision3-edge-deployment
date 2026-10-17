"""Verify our export hook with the actual official writer, without model inference.

The generated file contains metadata only, not an executable model.
"""
import inspect
import json
from pathlib import Path
import sys
sys.path.insert(0,'/support')
from patches.d3_export_adapter import D3Contract, install_writer_hook
import rkllm.base.converter as converter
from rkllm.base.gguf import GGUFReader

root=Path('/support')
contract=D3Contract.from_model('/work/source-model')
target=root/'output/d3-contract-metadata-only.rkllm'
report={'artifact_is_executable_model':False,'contract':contract.as_dict(),'method_signatures':{}}
for key in ['add_array','write_header_to_file','write_kv_data_to_file','write_ti_data_to_file','close']:
    report['method_signatures'][key]=str(inspect.signature(getattr(converter.LLMWriter,key)))
original=install_writer_hook(contract)
writer=None
stream=None
try:
    stream=target.open('wb')
    writer=converter.LLMWriter(stream,'custom',use_temp_file=False)
    writer.add_block_count(len(contract.layer_types))
    writer.add_embedding_length(contract.hidden_size)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    writer.close(); writer=None
    reader=GGUFReader(str(target),crypt=True)
    readback={}
    for key in ['custom.block_count','custom.attention.causal','d3.contract.layer_types',
                'd3.contract.linear_attention_causal','d3.contract.prefill_order']:
        field=reader.fields[key]
        values=[]
        for index in field.data:
            part=field.parts[index]
            if str(field.types[-1])=='8': values.append(bytes(part).decode())
            else: values.append(part.tolist())
        readback[key]=values
    report['readback']=readback
    assert readback['custom.attention.causal']==[[False]]
    assert readback['custom.block_count']==[[12]]
    assert readback['d3.contract.layer_types']==[[int(kind=='full_attention')] for kind in contract.layer_types]
    assert readback['d3.contract.linear_attention_causal']==[[True]]
    assert readback['d3.contract.prefill_order']==[[1]]
    report['serialization_passed']=True
finally:
    converter.LLMWriter.write_header_to_file=original
    if writer is not None: writer.close()
    if stream is not None: stream.close()
    (root/'logs/writer-patch-readback.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
