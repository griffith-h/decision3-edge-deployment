// The included Matrix is our existing self-authored normal-layout wrapper.
// Read only after all engine workers joined; ownership stays with the Owner.
#include "mixed_bridge_reference.inc"
extern "C" int d3_projection_read_fp32(void*handle,int rows,int k,int n,float*output){
 auto*p=static_cast<Matrix*>(handle);
 if(!p||!output||!p->fp32||p->prepared_rows!=rows||rows<=0||rows>512||p->k!=k||p->n!=n)return -1;
 std::memcpy(output,p->c->virt_addr,size_t(rows)*n*sizeof(float));
 return 0;
}
