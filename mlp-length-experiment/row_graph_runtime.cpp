// Own synchronous wrapper around the public RKNN graph API. No runtime changes.
#include <vector>
#include <fstream>
#include <iterator>
#include <string>
#include <memory>
#include "rknn_api.h"
#include "mlp_io_codec.h"
static thread_local std::string error;
struct D3RowGraph{rknn_context ctx=0;std::vector<char> model;std::vector<rknn_tensor_attr> in;size_t output_size=0;std::vector<uint16_t> half_input,half_output;std::vector<float> float_output;~D3RowGraph(){if(ctx)rknn_destroy(ctx);}};
static bool ok(int rc,const char*name){if(!rc)return true;error=std::string(name)+" returned "+std::to_string(rc);return false;}
extern "C" const char* row_graph_error(){return error.c_str();}
extern "C" void* row_graph_create(const char*path){
 auto p=std::make_unique<D3RowGraph>();std::ifstream f(path,std::ios::binary);if(!f){error="model file unavailable";return nullptr;}p->model=std::vector<char>(std::istreambuf_iterator<char>(f),{});
 if(!ok(rknn_init(&p->ctx,p->model.data(),p->model.size(),RKNN_FLAG_DISABLE_PROC_HIGH_PRIORITY,nullptr),"init"))return nullptr;
 rknn_input_output_num count{};if(!ok(rknn_query(p->ctx,RKNN_QUERY_IN_OUT_NUM,&count,sizeof(count)),"query_count"))return nullptr;
 if(count.n_input!=1||count.n_output!=1){error="expected one input and one output";return nullptr;}
 p->in.resize(count.n_input);
 for(unsigned i=0;i<count.n_input;++i){p->in[i].index=i;if(!ok(rknn_query(p->ctx,RKNN_QUERY_INPUT_ATTR,&p->in[i],sizeof(rknn_tensor_attr)),"query_input"))return nullptr;}
 rknn_tensor_attr out{};if(!ok(rknn_query(p->ctx,RKNN_QUERY_OUTPUT_ATTR,&out,sizeof(out)),"query_output"))return nullptr;p->output_size=out.n_elems;
 if(p->in[0].n_dims!=3 || p->in[0].dims[0]!=1 || p->in[0].dims[2]!=1024 || p->in[0].dims[1]==0 || p->in[0].dims[1]>2048 || out.n_dims!=3 || out.dims[0]!=1 || out.dims[1]!=p->in[0].dims[1] || out.dims[2]!=1024 || p->output_size!=p->in[0].n_elems){error="unexpected row graph dimensions";return nullptr;}return p.release();
}
extern "C" const char* row_graph_input_name(void*h,int i){auto*p=static_cast<D3RowGraph*>(h);return p&&i>=0&&i<1?p->in[i].name:nullptr;}
extern "C" int row_graph_run(void*h,const float*const*data,const int*sizes,float*output){
 auto*p=static_cast<D3RowGraph*>(h);if(!p||!data||!sizes||!output){error="invalid argument";return -1;}rknn_input inputs[1]{};
 for(int i=0;i<1;++i){if(!data[i]||sizes[i]!=int(p->in[i].n_elems)){error="invalid input size";return -1;}inputs[i].index=i;inputs[i].buf=const_cast<float*>(data[i]);inputs[i].size=sizes[i]*4;inputs[i].type=RKNN_TENSOR_FLOAT32;inputs[i].fmt=RKNN_TENSOR_NCHW;}
 if(!ok(rknn_inputs_set(p->ctx,1,inputs),"inputs_set"))return -1;
 if(!ok(rknn_run(p->ctx,nullptr),"run"))return -1;
 rknn_output out{};out.want_float=1;out.is_prealloc=1;out.buf=output;out.size=p->output_size*4;
 int rc=rknn_outputs_get(p->ctx,1,&out,nullptr);if(!ok(rc,"outputs_get"))return -1;
 return ok(rknn_outputs_release(p->ctx,1,&out),"outputs_release")?0:-1;
}
extern "C" void row_graph_destroy(void*h){delete static_cast<D3RowGraph*>(h);}

extern "C" int row_graph_meta(void*h,int*out){
 auto*p=static_cast<D3RowGraph*>(h);rknn_tensor_attr a{};a.index=0;int rc=rknn_query(p->ctx,RKNN_QUERY_OUTPUT_ATTR,&a,sizeof(a));if(rc)return rc;
 out[0]=p->in[0].type;out[1]=p->in[0].fmt;out[2]=a.type;out[3]=a.fmt;out[4]=a.n_elems;return 0;
}
extern "C" int row_graph_run_bf16(void*h,const uint16_t*x,uint16_t*y){
 auto*p=static_cast<D3RowGraph*>(h);if(!p||!x||!y)return -1;size_t count=p->output_size;
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

extern "C" int row_graph_set_core(void*h,int mask){auto*p=static_cast<D3RowGraph*>(h);if(!p)return -1;return rknn_set_core_mask(p->ctx,static_cast<rknn_core_mask>(mask));}

extern "C" int row_graph_rows(void*h){auto*p=static_cast<D3RowGraph*>(h);return p?int(p->in[0].dims[1]):0;}
