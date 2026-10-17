#pragma OPENCL FP_CONTRACT OFF
inline float bf(ushort v){return as_float(((uint)v)<<16);}
inline ushort rb(float v){uint u=as_uint(v);if((u&0x7fffffffU)>0x7f800000U)return (u>>16)|0x40U;return (u+0x7fffU+((u>>16)&1U))>>16;}
__kernel void attn_scores(__global const ushort*q,__global const ushort*k,__global float*s,int T,int H,int KH,int D,float scale){
 int x=get_local_id(0),y=get_local_id(1),j=get_group_id(0)*8+x,t=get_group_id(1)*8+y,h=get_group_id(2),kh=h/(H/KH);
 __local float a[8][16],b[16][8];float accum=0;
 for(int base=0;base<D;base+=16){
  for(int i=x;i<16;i+=8)a[y][i]=(t<T&&base+i<D)?bf(q[(h*T+t)*D+base+i]):0;
  for(int i=y;i<16;i+=8)b[i][x]=(j<T&&base+i<D)?bf(k[(kh*T+j)*D+base+i]):0;
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int i=0;i<16;i++)accum+=a[y][i]*b[i][x];
  barrier(CLK_LOCAL_MEM_FENCE);
 }
 if(t<T&&j<T)s[(h*T+t)*T+j]=accum*scale;
}
__kernel void attn_softmax_parallel(__global float*s,__global const uchar*mask,int T){
 int lane=get_local_id(0),row=get_group_id(0);__local float tmp[64];float mx=-INFINITY;
 for(int j=lane;j<T;j+=64){float v=mask[j]?s[row*T+j]:-INFINITY;s[row*T+j]=v;mx=fmax(mx,v);}
 tmp[lane]=mx;barrier(CLK_LOCAL_MEM_FENCE);
 for(int n=32;n;n/=2){if(lane<n)tmp[lane]=fmax(tmp[lane],tmp[lane+n]);barrier(CLK_LOCAL_MEM_FENCE);}
 mx=tmp[0];barrier(CLK_LOCAL_MEM_FENCE);float sum=0;
 for(int j=lane;j<T;j+=64){float v=exp(s[row*T+j]-mx);s[row*T+j]=v;sum+=v;}
 tmp[lane]=sum;barrier(CLK_LOCAL_MEM_FENCE);
 for(int n=32;n;n/=2){if(lane<n)tmp[lane]+=tmp[lane+n];barrier(CLK_LOCAL_MEM_FENCE);}
 sum=tmp[0];for(int j=lane;j<T;j+=64)s[row*T+j]/=sum;
}
__kernel void attn_output(__global const float*s,__global const ushort*v,__global ushort*out,int T,int H,int KH,int D){
 int x=get_local_id(0),y=get_local_id(1),d=get_group_id(0)*8+x,t=get_group_id(1)*8+y,h=get_group_id(2),kh=h/(H/KH);
 __local float a[8][16],b[16][8];float accum=0;
 for(int base=0;base<T;base+=16){
  for(int i=x;i<16;i+=8)a[y][i]=(t<T&&base+i<T)?s[(h*T+t)*T+base+i]:0;
  for(int i=y;i<16;i+=8)b[i][x]=(d<D&&base+i<T)?bf(v[(kh*T+base+i)*D+d]):0;
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int i=0;i<16;i++)accum+=a[y][i]*b[i][x];
  barrier(CLK_LOCAL_MEM_FENCE);
 }
 if(t<T&&d<D)out[(t*H+h)*D+d]=rb(accum);
}
/* Four disjoint K partitions, 16 independent V columns per workgroup.
   Zero initial state, causal recurrence, FP32 state throughout. */
__kernel void delta_scan(__global const float*q,__global const float*k,__global const float*v,__global const float*decay,__global const float*beta,__global ushort*out,int T,int H,int KD,int VD){
 int col=get_local_id(0),part=get_local_id(1),vc=get_group_id(0)*16+col,h=get_group_id(1);
 __local float sums[4][16];float state[32];for(int i=0;i<32;i++)state[i]=0;
 for(int t=0;t<T;t++){
  float d=decay[h*T+t],b=beta[h*T+t],pred=0;
  for(int i=0;i<32;i++){int ki=part*32+i;if(ki<KD){state[i]*=d;pred+=state[i]*k[(h*T+t)*KD+ki];}}
  sums[part][col]=pred;barrier(CLK_LOCAL_MEM_FENCE);
  pred=(sums[0][col]+sums[1][col])+(sums[2][col]+sums[3][col]);
  float change=(vc<VD)?(v[(h*T+t)*VD+vc]-pred)*b:0;
  barrier(CLK_LOCAL_MEM_FENCE);float value=0;
  for(int i=0;i<32;i++){int ki=part*32+i;if(ki<KD){state[i]+=k[(h*T+t)*KD+ki]*change;value+=state[i]*q[(h*T+t)*KD+ki];}}
  sums[part][col]=value;barrier(CLK_LOCAL_MEM_FENCE);
  if(part==0&&vc<VD)out[(t*H+h)*VD+vc]=rb((sums[0][col]+sums[1][col])+(sums[2][col]+sums[3][col]));
  barrier(CLK_LOCAL_MEM_FENCE);
 }
}

/* Independent row workitems avoid cross-lane reduction in this fallback. */
__kernel void attn_softmax(__global float*s,__global const uchar*mask,int T,int rows){
 int row=get_global_id(0);if(row>=rows)return;float mx=-INFINITY;
 for(int j=0;j<T;j++){float v=mask[j]?s[row*T+j]:-INFINITY;s[row*T+j]=v;mx=fmax(mx,v);}
 float sum=0;for(int j=0;j<T;j++){float v=exp(s[row*T+j]-mx);s[row*T+j]=v;sum+=v;}
 for(int j=0;j<T;j++)s[row*T+j]/=sum;
}
