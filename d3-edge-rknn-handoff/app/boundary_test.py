"""Focused checks for real native buffers, dynamic shape switches and CPU fallback."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from cpu_optimized_server import optimize
from rknn_linear import Engine, install, uninstall

torch.set_num_threads(4)
torch.manual_seed(9)
backbone=torch.nn.Module()
backbone.language_model=torch.nn.Module()
backbone.language_model.layers=torch.nn.ModuleList([torch.nn.Module()])
layer=torch.nn.Linear(1024,512,bias=True).bfloat16().eval().requires_grad_(False)
backbone.language_model.layers[0].proj=layer
model=SimpleNamespace(backbone=backbone)
optimize(model)
cpu_forward=layer.forward
engine=Engine()
result={}
try:
    assert len(install(model,engine))==1
    for rows in [127,128,129,256,257,512,513]:
        x=torch.randn(1,rows,1024).bfloat16()
        reference=cpu_forward(x)
        before=engine.calls
        output=layer(x)
        assert torch.isfinite(output).all()
        if rows==513:
            assert torch.equal(output,reference)
            assert engine.calls==before and engine.cpu_fallbacks==1
        else:
            assert engine.calls==before+1
            assert (output.float()-reference.float()).norm()/reference.float().norm()<0.001
        result[str(rows)]={'npu_used':engine.calls>before,'max_absolute_delta':(output.float()-reference.float()).abs().max().item()}
    x=torch.randn(1,17,1024,dtype=torch.float32)
    assert torch.equal(layer(x),cpu_forward(x))
    assert engine.cpu_fallbacks==2
    try:
        engine.lib.mm_create(7,8,x.data_ptr())
        assert 'alignment' in engine.lib.mm_error().decode()
        bad=torch.randn(513,1024).half()
        engine.run(layer._rknn_handle,bad,512)
        raise AssertionError('Native oversized request accepted')
    except RuntimeError as exc:
        assert 'Invalid run argument' in str(exc)
    uninstall(model)
    engine.close()
    assert torch.equal(layer(x),cpu_forward(x))
    result.update(non_bf16_cpu_fallback=True,native_oversize_rejected=True,cpu_forward_restored_after_release=True,passed=True)
    Path('/work/logs/boundary-test.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
finally:
    uninstall(model)
    engine.close()
