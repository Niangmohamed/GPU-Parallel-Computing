__global__  void somme( int taille, float * a, float * b, float *c  ){

     int index=threadIdx.x+blockDim.x*blockIdx.x;    
     if(index>=taille) return;
     c[index]=a[index]+b[index];
}


__global__  void prod( int taille, float * a, float  b, float *c  ){

     int index=threadIdx.x+blockDim.x*blockIdx.x;    
     if(index>=taille) return;
     c[index]=a[index]*b;
}