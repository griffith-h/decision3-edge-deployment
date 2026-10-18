from pathlib import Path
import json,statistics
p=Path(__file__).resolve().parent
screen=json.loads((p/'logs/screen-domains.json').read_text());old=json.loads((p/'logs/screen-float.json').read_text());ablate=json.loads((p/'logs/ablate-domains.json').read_text())
assert screen['completed'] and old['completed'] and ablate['completed'] and len(screen['cases'])==len(old['cases'])==13
rank={}
for mode in ('float','float_readout','mlponly_readout','projection_readout'):
 rows=[r for r in ablate['records'] if r['mode']==mode];assert len(rows)==12
 source=old if mode=='float' else screen;proof=[r for c in source['cases'].values() for r in c['records'] if r['mode']==mode];assert len(proof)==13
 passed=all(r['vs_current']['passed'] and not r['precise_errors'] and not r.get('last_readout_errors',[]) and not r['precise_range_fallbacks'] for r in rows+proof)
 rank[mode]={'median':statistics.median(r['seconds'] for r in rows),'passed':passed,'n':len(rows)}
assert rank['float']['passed'],'No qualified full FP32 path'
winner=min((k for k in rank if rank[k]['passed']),key=lambda k:rank[k]['median'])
if rank[winner]['median']>rank['float']['median']*.98:winner='float'
if winner in ('mlponly_readout','projection_readout') and rank['float_readout']['passed'] and rank[winner]['median']>rank['float_readout']['median']*.98:winner='float_readout'
domain={'float':'all','float_readout':'all','mlponly_readout':'mlp','projection_readout':'projections'}[winner]
e=json.loads((p/'execution.json').read_text());e.update(fp32=True,fp32_domain=domain,precise615=domain!='projections',precise_width=640,readout=winner.endswith('_readout'),full640=False,final_mode=winner,domain_selection=rank,domain_rule='All13 original-current gates plus12 rotating samples/mode. Require2% median gain to add readout/domain restriction. Final34-case regression mandatory.',scope='Frozen B1/T615 logits only for FP32/precise615/readout; original qualified paths for other shapes, short multi-batch precision repair retained.')
(p/'execution.json').write_text(json.dumps(e,indent=2)+'\n');print('DOMAIN SELECTION',e)
