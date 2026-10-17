"""Extra numerical tests for the exact graph shape, never counted in latency."""
import copy
from pathlib import Path
import json

def build(model,single):
    out=[]
    for i,text in enumerate(['你好。','1+1=?','Explain photosynthesis.','Write a Python loop.','我今天想出去散步。','计算 37 乘以 19。','Translate hello to French.','How to sort a list?','Tell me a joke.','Explain gravity.','推荐三个轻松聊天话题。','检查这段代码是否有错误。']):
        found=None
        for n in range(80):
            c=copy.deepcopy(single);c.update(id='shape115-'+str(i),state=text+' x'*n)
            prepared=model.prepare(state=c['state'],questions=c['questions']);length=len(next(iter(prepared.sequences.values())))
            if length==115:found=c;break
            if length>115:break
        if found is not None:out.append(found)
    if len(out)<8:raise ValueError('insufficient target shape cases '+str(len(out)))
    return out
