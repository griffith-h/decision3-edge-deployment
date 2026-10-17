#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
static thread_local std::string error;
static void check(cl_int c,const char*op){if(c!=CL_SUCCESS)throw std::runtime_error(std::string(op)+" code "+std::to_string(c));}
struct D3GPU {
 cl_context ctx=nullptr;cl_command_queue queue=nullptr;cl_program program=nullptr;cl_device_id device=nullptr;
 cl_kernel score=nullptr,softmax=nullptr,output=nullptr,scan=nullptr;cl_mem bufs[7]={};size_t sizes[7]={};
 ~D3GPU(){for(auto b:bufs)if(b)clReleaseMemObject(b);for(auto k:{score,softmax,output,scan})if(k)clReleaseKernel(k);if(program)clReleaseProgram(program);if(queue)clReleaseCommandQueue(queue);if(ctx)clReleaseContext(ctx);}
 cl_mem buffer(int i,size_t size){if(sizes[i]<size){if(bufs[i])clReleaseMemObject(bufs[i]);cl_int c;bufs[i]=clCreateBuffer(ctx,CL_MEM_READ_WRITE,size,nullptr,&c);sizes[i]=size;check(c,"create_buffer");}return bufs[i];}
 void write(int i,const void*p,size_t n){auto b=buffer(i,n);check(clEnqueueWriteBuffer(queue,b,CL_FALSE,0,n,p,0,nullptr,nullptr),"write_buffer");}
 void read(int i,void*p,size_t n){check(clEnqueueReadBuffer(queue,bufs[i],CL_TRUE,0,n,p,0,nullptr,nullptr),"read_buffer");}
 template<class T>void arg(cl_kernel k,int i,T x){check(clSetKernelArg(k,i,sizeof(T),&x),"set_arg");}
 void run(cl_kernel k,int n,const size_t*g,const size_t*l){check(clEnqueueNDRangeKernel(queue,k,n,nullptr,g,l,0,nullptr,nullptr),"enqueue_kernel");}
};
extern "C" const char*d3_gpu_error(){return error.c_str();}
extern "C" void*d3_gpu_create(const char*source){
 try{
  auto p=std::make_unique<D3GPU>();cl_uint n=0;check(clGetPlatformIDs(0,nullptr,&n),"platforms");std::vector<cl_platform_id>platforms(n);check(clGetPlatformIDs(n,platforms.data(),nullptr),"platforms");
  for(auto platform:platforms){cl_uint count=0;if(clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,1,&p->device,&count)==CL_SUCCESS&&count)break;}
  if(!p->device)throw std::runtime_error("No GPU device");cl_int c;p->ctx=clCreateContext(nullptr,1,&p->device,nullptr,nullptr,&c);check(c,"create_context");p->queue=clCreateCommandQueue(p->ctx,p->device,0,&c);check(c,"create_queue");
  std::ifstream f(source);if(!f)throw std::runtime_error("Cannot read own kernel source");std::stringstream ss;ss<<f.rdbuf();auto code=ss.str();const char*s=code.c_str();size_t size=code.size();p->program=clCreateProgramWithSource(p->ctx,1,&s,&size,&c);check(c,"create_program");
  c=clBuildProgram(p->program,1,&p->device,"-cl-std=CL1.2",nullptr,nullptr);
  if(c){size_t n=0;clGetProgramBuildInfo(p->program,p->device,CL_PROGRAM_BUILD_LOG,0,nullptr,&n);std::string log(n,'\0');clGetProgramBuildInfo(p->program,p->device,CL_PROGRAM_BUILD_LOG,n,log.data(),nullptr);throw std::runtime_error("Own kernel build failed: "+log);}
  p->score=clCreateKernel(p->program,"attn_scores",&c);check(c,"create_score");p->softmax=clCreateKernel(p->program,"attn_softmax",&c);check(c,"create_softmax");p->output=clCreateKernel(p->program,"attn_output",&c);check(c,"create_output");p->scan=clCreateKernel(p->program,"delta_scan",&c);check(c,"create_scan");return p.release();
 }catch(const std::exception&e){error=e.what();return nullptr;}
}
extern "C" void d3_gpu_destroy(void*h){delete static_cast<D3GPU*>(h);}
extern "C" int d3_gpu_attention(void*h,const uint16_t*q,const uint16_t*k,const uint16_t*v,const uint8_t*mask,uint16_t*out,int T,int H,int KH,int D,float scale){
 try{
  if(!h||T<1||T>512||H<1||H>16||KH<1||H%KH||D<1||D>256)throw std::runtime_error("Unsupported attention shape");auto*p=static_cast<D3GPU*>(h);
  p->write(0,q,(size_t)H*T*D*2);p->write(1,k,(size_t)KH*T*D*2);p->write(2,v,(size_t)KH*T*D*2);p->write(3,mask,T);p->buffer(4,(size_t)H*T*T*4);p->buffer(5,(size_t)H*T*D*2);
  p->arg(p->score,0,p->bufs[0]);p->arg(p->score,1,p->bufs[1]);p->arg(p->score,2,p->bufs[4]);p->arg(p->score,3,T);p->arg(p->score,4,H);p->arg(p->score,5,KH);p->arg(p->score,6,D);p->arg(p->score,7,scale);size_t local[3]={8,8,1},g[3]={(size_t)(T+7)/8*8,(size_t)(T+7)/8*8,(size_t)H};p->run(p->score,3,g,local);if(std::getenv("D3_GPU_DEBUG")){std::vector<float>v(H*T*T);p->read(4,v.data(),v.size()*4);std::fprintf(stderr,"OWN GPU SCORE %g %g\n",v[0],v.back());}
  p->arg(p->softmax,0,p->bufs[4]);p->arg(p->softmax,1,p->bufs[3]);p->arg(p->softmax,2,T);p->arg(p->softmax,3,H*T);size_t gs=(size_t)(H*T+63)/64*64,ls=64;p->run(p->softmax,1,&gs,&ls);if(std::getenv("D3_GPU_DEBUG")){std::vector<float>v(H*T*T);p->read(4,v.data(),v.size()*4);std::fprintf(stderr,"OWN GPU SOFTMAX %g %g\n",v[0],v.back());}
  p->arg(p->output,0,p->bufs[4]);p->arg(p->output,1,p->bufs[2]);p->arg(p->output,2,p->bufs[5]);p->arg(p->output,3,T);p->arg(p->output,4,H);p->arg(p->output,5,KH);p->arg(p->output,6,D);g[0]=(D+7)/8*8;p->run(p->output,3,g,local);p->read(5,out,(size_t)H*T*D*2);return 0;
 }catch(const std::exception&e){error=e.what();return -1;}
}
extern "C" int d3_gpu_scan(void*h,const float*q,const float*k,const float*v,const float*decay,const float*beta,uint16_t*out,int T,int H,int KD,int VD){
 try{
  if(!h||T<1||T>512||H<1||H>32||KD<1||KD>128||VD<1||VD>128)throw std::runtime_error("Unsupported scan shape");auto*p=static_cast<D3GPU*>(h);
  p->write(0,q,(size_t)H*T*KD*4);p->write(1,k,(size_t)H*T*KD*4);p->write(2,v,(size_t)H*T*VD*4);p->write(3,decay,(size_t)H*T*4);p->write(4,beta,(size_t)H*T*4);p->buffer(5,(size_t)H*T*VD*2);
  for(int i=0;i<6;i++)p->arg(p->scan,i,p->bufs[i]);p->arg(p->scan,6,T);p->arg(p->scan,7,H);p->arg(p->scan,8,KD);p->arg(p->scan,9,VD);
  size_t g[2]={(size_t)(VD+15)/16*16,(size_t)H},l[2]={16,1};p->run(p->scan,2,g,l);p->read(5,out,(size_t)H*T*VD*2);return 0;
 }catch(const std::exception&e){error=e.what();return -1;}
}
