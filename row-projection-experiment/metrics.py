from experiment import compare,numeric

def detailed(out,ref):
 result=compare(out,ref);a,b=numeric(out['answers']),numeric(ref['answers']);delta={k:abs(v-b[k]) for k,v in a.items()}
 result['max_prediction_field']=max(delta,key=delta.get)
 result['probability_max_delta']=max((v for k,v in delta.items() if '/probabilities/' in k),default=0)
 result['score_max_delta']=max((v for k,v in delta.items() if k.endswith('/score')),default=0)
 result['noul_max_delta']=max((v for k,v in delta.items() if k.endswith('/noul')),default=0)
 return result
