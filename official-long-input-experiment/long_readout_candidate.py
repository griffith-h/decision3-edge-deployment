"""Private last-position consumer for long B1 text logits; public encoder restored.

The first11 layers process every position. The twelfth layer keeps every K/V
position and computes the consumed query, residual, MLP and norm row. This is
the already validated615-row consumer extended behind the same structural guards.
"""
from types import MethodType

class LongReadout:
    def __init__(self,owner):
        self.owner=owner;self.model=owner.model;self.reference=owner.last_readout
        self.original=self.model.logits;self.closed=False;self.active=False;self.disabled=False;self.width=None
        self.counts={'optimized':0,'bypass':0,'errors':[],'last_metadata':{}}
        def scope(model,sequences,counts):return self.logits(sequences,counts)
        self.wrapper=MethodType(scope,self.model);self.model.logits=self.wrapper
    def good(self,sequences,counts):
        if self.closed or self.disabled or self.active or len(sequences)!=1 or len(counts)!=1 or not 2048<len(sequences[0])<=8192:return False
        g=self.reference;m=self.model
        if not g.supported or g.disabled or g.closed or m.encoder is not g.encoder:return False
        if getattr(type(m),'logits',None) is not g.original.__func__:return False
        import torch
        if str(m.device)!='cpu' or m.attention_mode!='noncausal_full_attention' or m.readout_dtype!='float32' or m.permutation_average:return False
        if g.last.block_type!='full_attention' or g.att.head_dim!=256 or (g.att.q_proj.out_features,g.att.k_proj.out_features,g.att.v_proj.out_features,g.att.o_proj.in_features,g.att.o_proj.out_features)!=(4096,512,512,2048,1024):return False
        if torch.is_autocast_enabled('cpu') or m.readout is not g.readout or m.readout._version!=g.readout_version:return False
        if type(g.encoder).__call__ is not g.encoder_call or g.encoder.recurrent_mask!=g.recurrent or g.language.rotary_emb.forward!=g.rotary_forward:return False
        if m.backbone.training or g.encoder.language_model is not g.language or len(g.encoder.stack)!=12 or any(a is not b or x!=y for (a,x),(b,y) in zip(g.encoder.stack,g.stack)):return False
        if any(v.training or v._forward_hooks or v._forward_pre_hooks or v._backward_hooks for v in m.backbone.modules()):return False
        live=list(g.last.modules())+[g.language.norm]
        if len(live)!=len(g.guarded_modules) or any(a is not b for a,b in zip(live,g.guarded_modules)):return False
        if any(getattr(v,'_bp_original',None)!=fn for v,fn in g.cpu_methods):return False
        if any(v.forward!=fn for v,fn in g.methods) or any(getattr(v,name) is not p or p._version!=version or p.device.type!='cpu' for v,name,p,version in g.parameters):return False
        return all(v.dtype==torch.bfloat16 for v in g.last.parameters()) and m.readout.dtype==torch.float32
    def logits(self,sequences,counts):
        if not self.good(sequences,counts):self.counts['bypass']+=1;return self.original(sequences,counts)
        old=self.model.encoder;self.active=True;self.width=len(sequences[0]);failed=False
        try:
            # D3.logits immediately consumes encoder(... )[:,-1]. The private
            # replacement returns exactly that one final-norm row. No public
            # encoder method is modified, and the attribute is restored first.
            self.model.encoder=self.encode
            result=self.original(sequences,counts)
            self.counts['optimized']+=1;return result
        except (ValueError,RuntimeError) as error:
            self.counts['errors'].append(str(error));self.disabled=True;failed=True
        finally:
            self.model.encoder=old;self.active=False;self.width=None
        if failed:return self.original(sequences,counts)
    def encode(self,embeds,present,mrope=None):
        import torch
        g=self.reference;model=g.language;width=self.width
        if mrope is not None or tuple(embeds.shape)!=(1,width,1024) or tuple(present.shape)!=(1,width) or embeds.dtype!=torch.bfloat16 or embeds.device.type!='cpu':raise ValueError('Unsupported private long encoder signature')
        ramp=torch.arange(width,device=embeds.device).view(1,1,-1).expand(4,1,-1);positions,mrope=ramp[0],ramp[1:]
        rope=model.rotary_emb(embeds,mrope);keys=present.bool()[:,None,None,:]
        scan=g.encoder.recurrent_mask(config=model.config,inputs_embeds=embeds,attention_mask=present,past_key_values=None,position_ids=positions)
        hidden=embeds
        for layer,softmax in g.stack[:-1]:hidden=layer(hidden,position_embeddings=rope,attention_mask=keys if softmax else scan,position_ids=positions,past_key_values=None,use_cache=False)
        return model.norm(self.last_layer(hidden,rope,keys))
    def last_layer(self,hidden,rope,mask):
        import torch,attention_prepare as ap,residual,tail_native as native
        g=self.reference;width=self.width
        if tuple(hidden.shape)!=(1,width,1024) or hidden.dtype!=torch.bfloat16 or mask.dtype!=torch.bool or tuple(mask.shape)!=(1,1,1,width):raise ValueError('Unsupported last-layer signature')
        cosine,sine=rope;R=cosine.shape[-1]
        if cosine.shape!=sine.shape or cosine.ndim!=3 or cosine.shape[:2]!=(1,width) or cosine.dtype!=torch.bfloat16 or not 0<R<=256 or R%8:raise ValueError('Unsupported long readout RoPE')
        residual_row=hidden[:,-1:].contiguous();x=g.last.input_layernorm(hidden)
        members=[g.att.k_proj,g.att.v_proj]
        if not self.owner.projections.good(members,x) or any(id(v) in self.owner.projections.disabled for v in members):raise ValueError('Long final K/V projection guard')
        values=self.owner.projections.run(members,x,tuple(id(v) for v in members));kr=values[id(members[0])].contiguous();vr=values[id(members[1])]
        qr=g.q_project(x[:,-1:].contiguous())
        if not all(native.finite(v.contiguous()) for v in (qr,kr,vr)):raise ValueError('Nonfinite final projections')
        qw=g.att.q_norm.weight.contiguous();kw=g.att.k_norm.weight.contiguous()
        q=torch.empty((1,8,1,256),dtype=hidden.dtype);k=torch.empty((1,2,width,256),dtype=hidden.dtype)
        c=cosine.contiguous();s=sine.contiguous();cl=c[:,-1:].contiguous();sl=s[:,-1:].contiguous()
        g.lib.d3_attention_prepare(qr.data_ptr(),qr.data_ptr(),qw.data_ptr(),kw.data_ptr(),cl.data_ptr(),sl.data_ptr(),q.data_ptr(),k.data_ptr(),1,1,8,0,256,R,g.att.q_norm.eps,g.att.k_norm.eps)
        g.lib.d3_attention_prepare(qr.data_ptr(),kr.data_ptr(),qw.data_ptr(),kw.data_ptr(),c.data_ptr(),s.data_ptr(),q.data_ptr(),k.data_ptr(),1,width,0,2,256,R,g.att.q_norm.eps,g.att.k_norm.eps)
        v=vr.view(1,width,2,256).transpose(1,2)
        y=g.attention(q,k,v,mask,g.att.scaling);y=ap.gated(y,qr,self.owner.candidate.table);projected=g.out_project(y)
        residual_row,x=residual.add_norm(residual_row,projected,g.last.post_attention_layernorm)
        # One pointwise MLP row, preserving the original BF16 boundaries.
        mlp=g.last.mlp;x=mlp.down_proj._bp_original(mlp.act_fn(mlp.gate_proj._bp_original(x))*mlp.up_proj._bp_original(x))
        out,_=residual.add_norm(residual_row,x)
        self.counts['last_metadata']={'input_tokens':width,'kv_tokens':width,'last_query_rows':1,'prefix_layers':11,'total_layers':12}
        return out
    def close(self):
        if self.closed:return
        if self.active:raise RuntimeError('Cannot close active private readout')
        self.closed=True
        if self.model.logits==self.wrapper:self.model.logits=self.original

def install(owner):
    owner.long_readout=LongReadout(owner)
    return owner.long_readout.close
