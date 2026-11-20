"""Observe real sparse repair at captured original-B8 projection differences.

Diagnostic intermediates only. CPU reference is rechecked at the complete
original geometry; row1 and public BLAS alternatives never redefine it.
"""
import ctypes,json,re
from pathlib import Path
import torch
from sparse_cpu_backend import BlasRepair

class PointComplete(Exception):pass

def run(model,owner,request,emit,out):
    item=torch.load('/inputs/first-projection-difference.pt',map_location='cpu',weights_only=True)
    shape=tuple(item['input'].shape)
    if shape[:2]!=(8,1181) or item['candidate'].shape!=item['expected'].shape:
        raise ValueError('Unexpected original B8 projection artifact')
    name=item['module'];match=re.fullmatch(r'language_model\.layers\.([0-9]|1[01])\..+',name)
    if match is None:raise ValueError('Invalid original decoder projection identity')
    layer_index=int(match.group(1));input_width=shape[-1];sources=[]
    component_path=Path('/inputs/layer-component-comparison.json')
    if component_path.exists():
        for record in json.loads(component_path.read_text())['records']:
            if (record['kind']=='projection' and record['bf16_mismatches']
                and record['module'].rsplit('.',1)[0]==name.rsplit('.',1)[0]
                and tuple(record['input_shape'])==shape):
                # These qkv/z members share the same real projection input;
                # verify every listed expected point with actual full CPU below.
                if len(record['examples'])!=record['bf16_mismatches']:
                    raise ValueError('Captured examples do not cover all differing points')
                sources.append((record['module'],record['examples']))
    if not sources:
        width=item['candidate'].shape[-1]
        r,c=map(int,(item['candidate'].reshape(-1,width)!=item['expected'].reshape(-1,width)).nonzero()[0])
        sources=[(name,[{'row':r,'column':c,'cpu':float(item['expected'].reshape(-1,width)[r,c])}])]
    emit({'event':'request_begin','phase':'verify_original_B8_CPU_points','budget_seconds':120})
    points=[];flat=item['input'].reshape(-1,input_width)
    for module_name,examples in sources:
        original=owner.projections.modules[module_name]
        with torch.inference_mode():expected=original._bp_original(item['input']).reshape(-1,original.out_features)
        for example in examples:
            r,c=example['row'],example['column']
            if float(expected[r,c])!=example['cpu']:raise AssertionError('Full CPU point reference did not reproduce')
            points.append({'name':module_name,'r':r,'c':c,'expected':example['cpu'],
                           'target':flat[r].clone(),'weight':original.weight[c].detach(),'module':original})
        del expected
    del flat,item
    state=owner.medium_sparse.repair;old_repair=state.repair
    layer=model.encoder.stack[layer_index][0];old_layer=layer.forward;active=False;records=[]
    libm=ctypes.CDLL(None);libm.fegetround.restype=ctypes.c_int
    def repair(module,inputs,raw,allow_inference_weight=False):
        matches=[]
        if active and inputs.shape[1]==input_width:
            for point in points:
                target,weight=point['target'],point['weight']
                rows=((inputs[:,:4]==target[:4]).all(dim=1)).nonzero().reshape(-1).tolist()
                cols=((module.weight[:,:4]==weight[:4]).all(dim=1)).nonzero().reshape(-1).tolist()
                rows=[i for i in rows if torch.equal(inputs[i],target)]
                cols=[i for i in cols if torch.equal(module.weight[i],weight)]
                if rows and cols:
                    if len(rows)!=1 or len(cols)!=1:raise ValueError('Ambiguous diagnostic point')
                    matches.append((point,rows[0],cols[0]))
        observations=[]
        if matches:
            wn,wl=state.weight_norms(module,allow_inference_weight);xn,xl,xhn=state.preparation.norms(inputs,True)
            positions=state.preparation.mask(raw,xn,xl,xhn,wn,wl,state.noise_factor,state.max_fraction,state.max_entries)
            for point,rr,cc in matches:
                band=((state.noise_factor*xn[rr])*wn[cc]+xl[rr]*wn[cc])+xhn[rr]*wl[cc]
                rv=raw[rr,cc];bits=rv.abs().view(torch.int32)
                lo=(bits & -65536).view(torch.float32);hi=((bits & -65536)+65536).view(torch.float32);mid=(lo+hi)*.5
                row={'shape':list(inputs.shape),'weight_shape':list(module.weight.shape),'position':[rr,cc],
                     'layer_index':layer_index,'original_module':point['name'],'original_full_batch_position':[point['r'],point['c']],
                     'raw_fp32':float(rv),'midpoint_magnitude':float(mid),'midpoint_distance':float((rv.abs()-mid).abs()),
                     'noise_factor':state.noise_factor,'band':float(band),'input_norm':float(xn[rr]),'weight_norm':float(wn[cc]),
                     'selected':bool(((positions[:,0]==rr)&(positions[:,1]==cc)).any()),'selected_count':len(positions),
                     'main_fegetround':libm.fegetround(),'original_cpu_bf16':point['expected'],'full_CPU_reference_reproduced':True}
                conversion=float(xl[rr])*float(wn[cc])+float(xhn[rr])*float(wl[cc])
                row['minimum_noise_estimate_selection_only']=max(0.,(row['midpoint_distance']-conversion)/(float(xn[rr])*float(wn[cc])))
                observations.append((point,rr,cc,row))
        result=old_repair(module,inputs,raw,allow_inference_weight=allow_inference_weight)
        for point,rr,cc,row in observations:
            row['actual_repaired_bf16']=float(result[rr,cc]);pairs=torch.tensor([[rr,cc]],dtype=torch.long)
            for kind,backend in (('forced_native_single_position',state.blas_repair),('public_cblas_alternative',BlasRepair())):
                private=result.clone();backend.run(module,inputs,pairs,private);row[kind]=float(private[rr,cc]);del private
            row['CPU_row1_not_canonical']=float(point['module']._bp_original(inputs[rr:rr+1])[0,point['c']])
            records.append(row)
            (out/'projection-repair-point.json').write_text(json.dumps({'completed':False,'full_request_completed':False,'records':records})+'\n')
            emit({'event':'repair_point_compared',**row})
        return result
    state.repair=repair
    def observed(*args,**kwargs):
        nonlocal active
        active=True
        try:old_layer(*args,**kwargs)
        finally:active=False
        raise PointComplete
    layer.forward=observed
    try:
        emit({'event':'request_begin','phase':'actual_layer'+str(layer_index)+'_sparse_points','budget_seconds':120})
        try:model.system_one(**request)
        except PointComplete:pass
        else:raise AssertionError('Point diagnostic failed to stop')
        if len(records)!=len(points):raise ValueError('Expected each preserved point exactly once')
        (out/'projection-repair-point.json').write_text(json.dumps({'completed':True,'full_request_completed':False,'records':records})+'\n')
        emit({'event':'complete','diagnostic_positions':len(records),'full_request_completed':False})
    finally:state.repair=old_repair;layer.forward=old_layer
