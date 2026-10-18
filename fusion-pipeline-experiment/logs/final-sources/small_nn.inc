/* Own4x8 NEON FP32 microkernel for the measured small NN GEMMs.
 * Original FP32 state, input dimensions and block32/tile64 are retained. */
static void small_nn(int m,int n,int k,const float*a,int lda,const float*b,int ldb,float*c,int ldc,float beta){
 int mr=m/4*4,nr=n/8*8;
 for(int j=0;j<nr;j+=8)for(int i=0;i<mr;i+=4){
  float32x4_t c00=vdupq_n_f32(0),c01=c00,c10=c00,c11=c00,c20=c00,c21=c00,c30=c00,c31=c00;
  const float*a0=a+i*lda,*a1=a0+lda,*a2=a1+lda,*a3=a2+lda;
  for(int p=0;p<k;p++){
   float32x4_t b0=vld1q_f32(b+p*ldb+j),b1=vld1q_f32(b+p*ldb+j+4);
   c00=vfmaq_n_f32(c00,b0,a0[p]);c01=vfmaq_n_f32(c01,b1,a0[p]);
   c10=vfmaq_n_f32(c10,b0,a1[p]);c11=vfmaq_n_f32(c11,b1,a1[p]);
   c20=vfmaq_n_f32(c20,b0,a2[p]);c21=vfmaq_n_f32(c21,b1,a2[p]);
   c30=vfmaq_n_f32(c30,b0,a3[p]);c31=vfmaq_n_f32(c31,b1,a3[p]);
  }
  float*out=c+i*ldc+j;
  if(beta!=0){
   c00=vaddq_f32(c00,vmulq_n_f32(vld1q_f32(out),beta));c01=vaddq_f32(c01,vmulq_n_f32(vld1q_f32(out+4),beta));
   c10=vaddq_f32(c10,vmulq_n_f32(vld1q_f32(out+ldc),beta));c11=vaddq_f32(c11,vmulq_n_f32(vld1q_f32(out+ldc+4),beta));
   c20=vaddq_f32(c20,vmulq_n_f32(vld1q_f32(out+2*ldc),beta));c21=vaddq_f32(c21,vmulq_n_f32(vld1q_f32(out+2*ldc+4),beta));
   c30=vaddq_f32(c30,vmulq_n_f32(vld1q_f32(out+3*ldc),beta));c31=vaddq_f32(c31,vmulq_n_f32(vld1q_f32(out+3*ldc+4),beta));
  }
  vst1q_f32(out,c00);vst1q_f32(out+4,c01);vst1q_f32(out+ldc,c10);vst1q_f32(out+ldc+4,c11);vst1q_f32(out+2*ldc,c20);vst1q_f32(out+2*ldc+4,c21);vst1q_f32(out+3*ldc,c30);vst1q_f32(out+3*ldc+4,c31);
 }
 if(mr&&nr<n)gemm(101,111,111,mr,n-nr,k,1,a,lda,b+nr,ldb,beta,c+nr,ldc);
 if(mr<m)gemm(101,111,111,m-mr,n,k,1,a+mr*lda,lda,b,ldb,beta,c+mr*ldc,ldc);
}
