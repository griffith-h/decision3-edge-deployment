"""Locate primitive divergence in one original B8 decoder layer; diagnostic only."""
import json
import torch

class LayerComplete(Exception):pass

def run(model,owner,request,emit,out,replay_path=None,layer_index=4):
    if type(layer_index) is not int or not 0<=layer_index<12:
        raise ValueError('Invalid original decoder layer index')
    replay=None
    if replay_path is not None:
        replay=torch.load(replay_path,map_location='cpu',weights_only=True);layer_index=replay['layer_index']
        if layer_index!=11 or tuple(replay['input'].shape)!=(8,1181,1024) or replay['input'].dtype!=torch.bfloat16:
            raise ValueError('Unexpected original last-layer activation replay')
    records=[];active=False;mlp_active=False;collected={};products=[];changed=[]
    def save():
        (out/'layer-component-comparison.json').write_text(json.dumps({'completed':False,'full_request_completed':False,
            'scope':f'Original B8/T1181 layer{layer_index} primitives; actual CPU on identical inputs; activation replay={replay is not None}; partial diagnostic only','records':records})+'\n')
    def compare(kind,a,b,**meta):
        if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('Component tensor contract changed')
        aa=a.reshape(-1,a.shape[-1]);bb=b.reshape_as(aa);count=0;maximum=0.;examples=[]
        for start in range(0,len(aa),512):
            x,y=aa[start:start+512],bb[start:start+512];diff=x!=y;count+=int(diff.sum())
            maximum=max(maximum,float((x.float()-y.float()).abs().max()))
            if len(examples)<8:
                for r,c in diff.nonzero()[:8-len(examples)].tolist():
                    examples.append({'row':start+r,'column':c,'candidate':float(x[r,c]),'cpu':float(y[r,c])})
        row={'kind':kind,'shape':list(a.shape),'bf16_mismatches':count,'maximum_delta':maximum,'examples':examples,**meta}
        records.append(row);save();emit({'event':'primitive_compared','kind':kind,'bf16_mismatches':count,'maximum_delta':maximum})
        return row
    def replace(obj,name,value):
        old=getattr(obj,name);setattr(obj,name,value);changed.append((obj,name,old));return old
    old_projection=owner.projections.run
    def projection(members,x,key):
        result=old_projection(members,x,key)
        if active:
            for module in members:
                name=next(n for n,m in owner.projections.modules.items() if m is module)
                expected=module._bp_original(x);got=result[id(module)]
                row=compare('projection',got,expected,module=name,input_shape=list(x.shape))
                if row['bf16_mismatches']:
                    torch.save({'input':x.detach(),'candidate':got.detach(),'expected':expected.detach(),'module':name},out/('first-projection-difference.pt'))
        return result
    replace(owner.projections,'run',projection)
    sparse=owner.medium_sparse;old_native=sparse.correct_native
    def native(module,x,handle=None):
        result=old_native(module,x,handle=handle)
        if mlp_active:
            collected.setdefault(id(module),[]).append(result.detach().reshape(-1,result.shape[-1]).clone())
            if module is down_module:products.append(x.detach().reshape(-1,x.shape[-1]).clone())
        return result
    replace(sparse,'correct_native',native)
    old_mlp=owner.long_mlp.forward;down_module=None
    def mlp(layer,old,members,params,activation,x):
        nonlocal mlp_active,down_module
        if not active:return old_mlp(layer,old,members,params,activation,x)
        gate,up,down=members;down_module=down;mlp_active=True
        try:result=old_mlp(layer,old,members,params,activation,x)
        finally:mlp_active=False
        rows=x.numel()//x.shape[-1]
        def joined(module):
            values=collected.pop(id(module),None)
            if not values:raise ValueError('Expected repaired native MLP not observed')
            return torch.cat(values,0)[:rows].reshape(*x.shape[:-1],module.out_features)
        gv=joined(gate);uv=joined(up);dv=joined(down)
        expected_gate=gate._bp_original(x);compare('mlp_gate',gv,expected_gate)
        expected_up=up._bp_original(x);compare('mlp_up',uv,expected_up)
        expected_product=activation(expected_gate)*expected_up
        product=torch.cat(products,0)[:rows].reshape(expected_product.shape);products.clear()
        compare('mlp_product',product,expected_product)
        del expected_gate,expected_up,gv,uv
        expected_down=down._bp_original(expected_product);compare('mlp_down',dv,expected_down)
        compare('mlp_result',result,expected_down)
        if records[-1]['bf16_mismatches']:
            torch.save({'input':x.detach(),'product':product.detach(),'cpu_product':expected_product.detach(),
                        'candidate':dv.detach(),'cpu':expected_down.detach()},out/'first-mlp-difference.pt')
        return result
    replace(owner.long_mlp,'forward',mlp)
    layer=model.encoder.stack[layer_index][0];old_layer=layer.forward
    def observe(*args,**kwargs):
        nonlocal active
        # Prefix traversal and same-input primitive comparisons are separate
        # bounded diagnostic phases, never a full-request speed measurement.
        emit({'event':'request_begin','phase':'same_input_layer'+str(layer_index)+'_primitives',
              'budget_seconds':120,'full_request_completed':False})
        active=True
        try:value=old_layer(*args,**kwargs)
        finally:active=False
        if replay is not None:
            compare('decoder_vs_captured_candidate',value,replay['candidate_output'],reference_kind='Previously captured same candidate decoder output')
            compare('decoder_vs_captured_cpu',value,replay['cpu_output'],reference_kind='Previously captured same-input original CPU decoder output')
        raise LayerComplete
    replace(layer,'forward',observe)
    if replay is not None:
        def encoder(embeds,present,mrope=None):
            if mrope is not None or tuple(embeds.shape)!=(8,1181,1024) or tuple(present.shape)!=(8,1181):
                raise ValueError('Activation replay changed original frozen batch geometry')
            expected_mask=present.bool()[:,None,None,:]
            if not torch.equal(expected_mask,replay['kwargs']['attention_mask']):raise ValueError('Activation replay changed original padding mask')
            # Diagnostic only: this activation came from the unchanged first11
            # layers of this exact frozen original B8. Never a deployed cache.
            value=replay['input'].clone()
            return layer(value,*replay['tail'],**replay['kwargs']) if replay['positional'] else layer(hidden_states=value,**replay['kwargs'])
        replace(model,'encoder',encoder)
    try:
        emit({'event':'request_begin','phase':'original_B8_layer'+str(layer_index)+'_primitives','activation_replay':replay is not None})
        try:model.system_one(**request)
        except LayerComplete:pass
        else:raise AssertionError('Layer diagnostic failed to stop')
        data=json.loads((out/'layer-component-comparison.json').read_text());data['completed']=True
        (out/'layer-component-comparison.json').write_text(json.dumps(data)+'\n')
        emit({'event':'complete','primitive_records':len(records),'full_request_completed':False})
    finally:
        for obj,name,old in reversed(changed):setattr(obj,name,old)
