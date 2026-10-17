// Own synchronous wrapper around the public RKNN graph API. No runtime changes.
#include <vector>
#include <fstream>
#include <iterator>
#include <string>
#include <memory>
#include "rknn_api.h"
static thread_local std::string error;
struct Graph{rknn_context ctx=0;std::vector<char> model;std::vector<rknn_tensor_attr> in;size_t output_size=0;~Graph(){if(ctx)rknn_destroy(ctx);}};
static bool ok(int rc,const char*name){if(!rc)return true;error=std::string(name)+" returned "+std::to_string(rc);return false;}
extern "C" const char* graph_error(){return error.c_str();}
extern "C" void* graph_create(const char*path){
 auto p=std::make_unique<Graph>();std::ifstream f(path,std::ios::binary);if(!f){error="model file unavailable";return nullptr;}p->model=std::vector<char>(std::istreambuf_iterator<char>(f),{});
 if(!ok(rknn_init(&p->ctx,p->model.data(),p->model.size(),RKNN_FLAG_DISABLE_PROC_HIGH_PRIORITY,nullptr),"init"))return nullptr;
 rknn_input_output_num count{};if(!ok(rknn_query(p->ctx,RKNN_QUERY_IN_OUT_NUM,&count,sizeof(count)),"query_count"))return nullptr;
 if(count.n_input!=4||count.n_output!=1){error="expected four inputs and one output";return nullptr;}
 p->in.resize(count.n_input);
 for(unsigned i=0;i<count.n_input;++i){p->in[i].index=i;if(!ok(rknn_query(p->ctx,RKNN_QUERY_INPUT_ATTR,&p->in[i],sizeof(rknn_tensor_attr)),"query_input"))return nullptr;}
 rknn_tensor_attr out{};if(!ok(rknn_query(p->ctx,RKNN_QUERY_OUTPUT_ATTR,&out,sizeof(out)),"query_output"))return nullptr;p->output_size=out.n_elems;
 if(p->output_size!=115*2048){error="unexpected output size";return nullptr;}return p.release();
}
extern "C" const char* graph_input_name(void*h,int i){auto*p=static_cast<Graph*>(h);return p&&i>=0&&i<4?p->in[i].name:nullptr;}
extern "C" int graph_run(void*h,const float*const*data,const int*sizes,float*output){
 auto*p=static_cast<Graph*>(h);if(!p||!data||!sizes||!output){error="invalid argument";return -1;}rknn_input inputs[4]{};
 for(int i=0;i<4;++i){if(!data[i]||sizes[i]!=int(p->in[i].n_elems)){error="invalid input size";return -1;}inputs[i].index=i;inputs[i].buf=const_cast<float*>(data[i]);inputs[i].size=sizes[i]*4;inputs[i].type=RKNN_TENSOR_FLOAT32;inputs[i].fmt=RKNN_TENSOR_NCHW; /* caller arrays use ONNX NCHW, not the optimized NHWC attribute */}
 if(!ok(rknn_inputs_set(p->ctx,4,inputs),"inputs_set"))return -1;
 if(!ok(rknn_run(p->ctx,nullptr),"run"))return -1;
 rknn_output out{};out.want_float=1;out.is_prealloc=1;out.buf=output;out.size=p->output_size*4;
 int rc=rknn_outputs_get(p->ctx,1,&out,nullptr);if(!ok(rc,"outputs_get"))return -1;
 return ok(rknn_outputs_release(p->ctx,1,&out),"outputs_release")?0:-1;
}
extern "C" void graph_destroy(void*h){delete static_cast<Graph*>(h);}
