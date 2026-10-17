from pathlib import Path
src=(Path(__file__).resolve().parent.parent/'remaining-500ms-experiment/conversion_bridge.cpp').read_text()
src=src.replace('info.AC_layout=RKNN_MM_LAYOUT_NORM;','info.AC_layout=RKNN_MM_LAYOUT_NATIVE;')
start=src.index('  int status=mm_convert_input(input,',src.index('extern "C" int mm_prepare_bf16'))
end=src.index('  p->prepared_rows=rows;',start)
src=src[:start]+'''  auto*dst=static_cast<uint16_t*>(p->a->virt_addr);int m=p->shapes[p->selected].M;
  if(p->attrs[p->selected].A.n_dims!=3 || p->attrs[p->selected].A.dims[2]!=8){last_error="Unexpected native A layout";return -1;}
  int bad=0;
  for(int kk=0;kk<p->k;kk+=64)for(int rr=0;rr<m;rr+=16)for(int j=kk;j<std::min(kk+64,p->k);j+=8)for(int r=rr;r<std::min(rr+16,m);r++){
   uint16_t*out=dst+(size_t)(j/8)*m*8+r*8;
   if(r<rows)bad|=mm_convert_input(input+(size_t)r*p->k+j,out,8)!=0;else std::memset(out,0,16);
  }
  if(bad){last_error="FP16 activation overflow";return -2;}
''' +src[end:]
start=src.index('  std::memcpy(dst->a->virt_addr',src.index('extern "C" int mm_copy_prepared'))
end=src.index('  dst->prepared_rows=rows;',start)
src=src[:start]+'''  if(src->shapes[src->selected].M!=dst->shapes[dst->selected].M){last_error="Native M mismatch";return -1;}
  std::memcpy(dst->a->virt_addr,src->a->virt_addr,src->attrs[src->selected].A.size);
''' +src[end:]
start=src.index('extern "C" int mm_run_prepared_bf16(')
end=src.index('extern "C" int mm_run_bf16(',start)
src=src[:start]+'''static int native_output(Matrix*p,int rows,int segments,const int*starts,const int*widths,const int*strides,uint16_t*const*targets,const float*bias){
 auto&attr=p->attrs[p->selected].C;int m=p->shapes[p->selected].M;
 if(attr.n_dims!=3 || attr.dims[2]!=8){last_error="Unexpected native C layout";return -1;}
 auto*src=static_cast<const uint16_t*>(p->c->virt_addr);bool bad=false;
 for(int s=0;s<segments;s++){
  if(starts[s]%8 || widths[s]%8){last_error="Native scatter requires 8 columns";return -1;}
  for(int nn=starts[s];nn<starts[s]+widths[s];nn+=64)for(int rr=0;rr<rows;rr+=16)for(int n=nn;n<std::min(nn+64,starts[s]+widths[s]);n+=8)for(int r=rr;r<std::min(rr+16,rows);r++){
   const uint16_t*in=src+(size_t)(n/8)*m*8+r*8;uint16_t*out=targets[s]+(size_t)r*strides[s]+n-starts[s];
   if(!bias)bad|=mm_fast_convert_output(in,out,8)!=0;
   else for(int j=0;j<8;j++){__fp16 h;std::memcpy(&h,in+j,2);bad|=(in[j]&0x7fff)>=0x7bff;out[j]=bf_round(static_cast<float>(h)+bias[n+j]);}
  }
 }
 if(bad){last_error="FP16 output saturated/nonfinite; original CPU fallback required";return -3;}return 0;
}
extern "C" int mm_run_prepared_bf16(void* handle,int rows,const float*bias,uint16_t*output){
 auto*p=static_cast<Matrix*>(handle);if(!p||!output||p->prepared_rows!=rows){last_error="Input not prepared";return -1;}
 if(!check(rknn_matmul_run(p->ctx),"run"))return -1;
 int start=0,width=p->n,stride=p->n;uint16_t*targets[1]={output};return native_output(p,rows,1,&start,&width,&stride,targets,bias);
}
''' +src[end:]
start=src.index('  const __fp16* source=',src.index('extern "C" int mm_run_prepared_scatter_bf16('))
src=src[:start]+'''  return native_output(p,rows,segments,starts,widths,strides,targets,bias);
}
'''
src+='\nextern "C" int mm_native_attrs(void*handle,int*out){auto*p=static_cast<Matrix*>(handle);if(!p)return -1;for(int i=0;i<3;i++){auto&a=i==0?p->attrs[0].A:i==1?p->attrs[0].B:p->attrs[0].C;out[i*12]=a.n_dims;out[i*12+1]=a.size;out[i*12+2]=a.type;for(unsigned j=0;j<a.n_dims;j++)out[i*12+3+j]=a.dims[j];}return 0;}\n'
Path(__file__).with_name('native_bridge.cpp').write_text(src)
# A separate variant tests whether eliminating M padding is supported/useful.
Path(__file__).with_name('m115_bridge.cpp').write_text((Path(__file__).resolve().parent.parent/'remaining-500ms-experiment/conversion_bridge.cpp').read_text().replace('{128, 256, 512}','{115, 128, 256, 512}').replace('int index=rows<=128?0:rows<=256?1:2;','int index=rows<=115?0:rows<=128?1:rows<=256?2:3;'))
