"""Split oversized padded batches, preserve every sequence and restore row order.

All sequences and option counts are retained; model weights are unchanged.
Padding and kernel selection can change finite-precision results; parity is measured.
"""
from types import MethodType
import torch


def plan(lengths,row_budget=2048,max_batch=8):
    if row_budget<=0 or max_batch<=0:raise ValueError('Positive batch limits required')
    if any(n<=0 for n in lengths):raise ValueError('Empty sequence is not supported')
    groups=[];current=[]
    for index in sorted(range(len(lengths)),key=lambda i:(lengths[i],i)):
        if current and (len(current)>=max_batch or (len(current)+1)*lengths[index]>row_budget):
            groups.append(current);current=[]
        current.append(index)
    if current:groups.append(current)
    return groups


class BatchPlanner:
    def __init__(self,model,row_budget=2048,max_subbatch=1):
        self.model=model;self.original=model.logits;self.row_budget=row_budget;self.closed=False;self.enabled=True
        self.counts={'max_subbatch':max_subbatch,'split_batches':0,'subbatches':0,'original_padded_tokens':0,'planned_padded_tokens':0,'plans':[]}
        def logits(model,sequences,counts):
            if (self.closed or not self.enabled or len(sequences)<=1 or len(sequences)!=len(counts)
                    or any(not s for s in sequences) or max(map(len,sequences))>row_budget
                    or len(sequences)*max(map(len,sequences))<=row_budget):
                return self.original(sequences,counts)
            groups=plan([len(s) for s in sequences],row_budget,max_batch=min(max_subbatch,len(sequences)))
            if len(groups)<=1:return self.original(sequences,counts)
            self.counts['split_batches']+=1
            self.counts['original_padded_tokens']+=len(sequences)*max(map(len,sequences))
            result=None
            with torch.inference_mode():
                for indices in groups:
                    values=self.original([sequences[i] for i in indices],[counts[i] for i in indices])
                    if values.ndim!=2 or values.shape[0]!=len(indices):raise RuntimeError('Unexpected logits batch shape')
                    if result is None:result=values.new_empty((len(sequences),values.shape[1]))
                    if result.shape[1:]!=values.shape[1:]:raise RuntimeError('Inconsistent logits width')
                    result.index_copy_(0,torch.as_tensor(indices,dtype=torch.long,device=values.device),values)
                    self.counts['subbatches']+=1;self.counts['planned_padded_tokens']+=len(indices)*max(len(sequences[i]) for i in indices)
            self.counts['plans'].append({'lengths':[len(s) for s in sequences],'groups':groups})
            return result
        self.wrapper=MethodType(logits,model);model.logits=self.wrapper
    def close(self):
        if self.closed:return
        self.closed=True
        if self.model.logits==self.wrapper:self.model.logits=self.original


def install(owner):
    owner.batch_planner=BatchPlanner(owner.model)
    return owner.batch_planner.close
