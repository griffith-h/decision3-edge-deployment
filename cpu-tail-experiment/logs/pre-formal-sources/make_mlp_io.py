from pathlib import Path
root=Path(__file__).resolve().parent
s=(root.parent/'remaining-500ms-experiment/conversion_bridge.cpp').read_text();start=s.index('static uint16_t bf_round(');end=s.index('static bool bind_shape(',start)
s=s[start:end].replace('extern "C" ','static ').replace('mm_fast_convert_output','mlp_convert_half').replace('mm_convert_input','mlp_convert_input').replace('mm_convert_output','mlp_convert_float')
(root/'mlp_io_codec.h').write_text('#include <arm_neon.h>\n#include <cstdint>\n#include <cstring>\n#include <cmath>\n'+s)
p=root/'mlp_graph_runtime.cpp'
s=(root.parent/'attention-chunk-experiment/graph_runtime.cpp').read_text().replace('Graph','D3MlpGraph').replace('graph_','mlp_graph_').replace('count.n_input!=4','count.n_input!=1').replace('four inputs','one input').replace('i<4','i<1').replace('rknn_input inputs[4]','rknn_input inputs[1]').replace('p->ctx,4,inputs','p->ctx,1,inputs').replace('inputs[i].fmt=p->in[i].fmt','inputs[i].fmt=RKNN_TENSOR_NCHW')
s=s.replace('#include "rknn_api.h"','#include "rknn_api.h"\n#include "mlp_io_codec.h"')
s=s.replace('size_t output_size=0;','size_t output_size=0;std::vector<uint16_t> half_input,half_output;std::vector<float> float_output;')
s+='''
extern "C" int mlp_graph_meta(void*h,int*out){
 auto*p=static_cast<D3MlpGraph*>(h);rknn_tensor_attr a{};a.index=0;int rc=rknn_query(p->ctx,RKNN_QUERY_OUTPUT_ATTR,&a,sizeof(a));if(rc)return rc;
 out[0]=p->in[0].type;out[1]=p->in[0].fmt;out[2]=a.type;out[3]=a.fmt;out[4]=a.n_elems;return 0;
}
extern "C" int mlp_graph_run_bf16(void*h,const uint16_t*x,uint16_t*y){
 auto*p=static_cast<D3MlpGraph*>(h);if(!p||!x||!y)return -1;size_t count=p->output_size;
 p->half_input.resize(count);int status=mlp_convert_input(x,p->half_input.data(),count);if(status)return status;
 rknn_input in{};in.index=0;in.buf=p->half_input.data();in.size=count*2;in.type=RKNN_TENSOR_FLOAT16;in.fmt=RKNN_TENSOR_NCHW;
 if(!ok(rknn_inputs_set(p->ctx,1,&in),"half_inputs_set"))return -1;if(!ok(rknn_run(p->ctx,nullptr),"half_run"))return -1;
 rknn_tensor_attr attr{};attr.index=0;if(!ok(rknn_query(p->ctx,RKNN_QUERY_OUTPUT_ATTR,&attr,sizeof(attr)),"half_query_out"))return -1;
 rknn_output out{};out.is_prealloc=1;out.want_float=0;
 if(attr.type==RKNN_TENSOR_FLOAT16){p->half_output.resize(count);out.buf=p->half_output.data();out.size=count*2;}
 else if(attr.type==RKNN_TENSOR_FLOAT32){p->float_output.resize(count);out.buf=p->float_output.data();out.size=count*4;}
 else{error="Unsupported nonquant MLP output dtype";return -1;}
 if(!ok(rknn_outputs_get(p->ctx,1,&out,nullptr),"half_outputs_get"))return -1;
 if(attr.type==RKNN_TENSOR_FLOAT16)status=mlp_convert_half(p->half_output.data(),y,count);
 else{mlp_convert_float(p->float_output.data(),y,count);for(auto v:p->float_output)if(!std::isfinite(v)||std::fabs(v)>=65504){status=-3;break;}}
 int rc=rknn_outputs_release(p->ctx,1,&out);return status?status:ok(rc,"half_outputs_release")?0:-1;
}
'''
p.write_text(s)
