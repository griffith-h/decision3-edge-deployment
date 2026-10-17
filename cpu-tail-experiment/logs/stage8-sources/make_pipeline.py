from pathlib import Path
root=Path(__file__).resolve().parent
extra=r'''
static int fuse_mlp_range(const uint16_t*gate,const uint16_t*up,uint16_t*out,const uint16_t*table,size_t count){
 int raw_bad=0,input_bad=0;size_t i=0;
 for(;i+8<=count;i+=8){
  uint16_t g[8],u[8],product[8],activation[8];
  raw_bad|=mm_fast_convert_output(gate+i,g,8)!=0;raw_bad|=mm_fast_convert_output(up+i,u,8)!=0;
  for(int j=0;j<8;j++)activation[j]=table[g[j]];
  for(int j=0;j<8;j+=4){float32x4_t a=vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(activation+j),16)),b=vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(u+j),16));
   uint32x4_t bits=vreinterpretq_u32_f32(vmulq_f32(a,b));uint32x4_t r=vshrq_n_u32(vaddq_u32(vaddq_u32(bits,vdupq_n_u32(0x7fff)),vandq_u32(vshrq_n_u32(bits,16),vdupq_n_u32(1))),16);vst1_u16(product+j,vmovn_u32(r));
  }
  input_bad|=mm_convert_input(product,out+i,8)!=0;
 }
 for(;i<count;i++){
  __fp16 gh,uh;std::memcpy(&gh,gate+i,2);std::memcpy(&uh,up+i,2);raw_bad|=(gate[i]&0x7fff)>=0x7bff || (up[i]&0x7fff)>=0x7bff;
  uint16_t gb=bf_round(static_cast<float>(gh)),ub=bf_round(static_cast<float>(uh));uint32_t a=uint32_t(table[gb])<<16,b=uint32_t(ub)<<16;float af,bf;std::memcpy(&af,&a,4);std::memcpy(&bf,&b,4);uint16_t prod=bf_round(af*bf);input_bad|=mm_convert_input(&prod,out+i,1)!=0;
 }
 return raw_bad?-3:input_bad?-2:0;
}
extern "C" int mm_fuse_mlp_half(const uint16_t*gate,const uint16_t*up,uint16_t*out,const uint16_t*table,size_t count){
 int raw_bad=0,input_bad=0;
 #pragma omp parallel for num_threads(3) schedule(static) if(count>4096) reduction(|:raw_bad,input_bad)
 for(size_t block=0;block<(count+2047)/2048;block++){
  size_t start=block*2048,n=std::min(size_t(2048),count-start);int status=fuse_mlp_range(gate+start,up+start,out+start,table,n);raw_bad|=status==-3;input_bad|=status==-2;
 }
 return raw_bad?-3:input_bad?-2:0;
}
extern "C" int mm_run_raw(void*handle,int rows){
 auto*p=static_cast<Matrix*>(handle);if(!p||p->prepared_rows!=rows){last_error="Raw input not prepared";return -1;}
 RAW_RUN
}
extern "C" int mm_prepare_mlp(void*down,void*gate,void*up,int rows,const uint16_t*table){
 auto*d=static_cast<Matrix*>(down);auto*g=static_cast<Matrix*>(gate);auto*u=static_cast<Matrix*>(up);
 if(!d||!g||!u||!table||g->n!=d->k||u->n!=d->k||g->prepared_rows!=rows||u->prepared_rows!=rows||!bind_shape(d,rows)){last_error="Invalid MLP pipeline shape";return -1;}
 d->prepared_rows=0;int status=mm_fuse_mlp_half(static_cast<uint16_t*>(g->c->virt_addr),static_cast<uint16_t*>(u->c->virt_addr),static_cast<uint16_t*>(d->a->virt_addr),table,size_t(rows)*d->k);
 if(status){last_error="MLP intermediate outside safe FP16 range";return status;}
 std::memset(static_cast<uint16_t*>(d->a->virt_addr)+size_t(rows)*d->k,0,size_t(d->shapes[d->selected].M-rows)*d->k*2);d->prepared_rows=rows;return 0;
}
'''
for name,base,raw in [('pipeline_bridge','conversion_bridge.cpp','return check(rknn_matmul_run(p->ctx),"run_raw")?0:-1;'),('pipeline_profile','profile_conversion.cpp','return prof_run(p,"run_raw")?0:-1;')]:
 src=(root.parent/'remaining-500ms-experiment'/base).read_text();(root/(name+'.cpp')).write_text(src+extra.replace('RAW_RUN',raw))
