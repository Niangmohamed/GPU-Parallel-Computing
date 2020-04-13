__device__ float function_a_appli(float x);


__global__  void applique_fonc ( int nb_ligne, int nb_col, float * input, float * res ){

     int index_col=threadIdx.x+blockDim.x*blockIdx.x;   
     int index_ligne=threadIdx.y+blockDim.y*blockIdx.y;    
     int global_index;
     if ((index_col >= nb_col) || (index_ligne>=nb_ligne) ) return;
     global_index=index_ligne*nb_col+index_col;
     res[global_index]=function_a_appli(input[global_index]);
}

__device__ float function_a_appli(float x){
     return( cos(x));
}


__global__ void mandelbrot (  int nb_ligne, int nb_col, float seuil, float x_min, float x_max, float y_min, float y_max, float* res) {
     int max_ITER=10000;
     int iter=0;
     int index_col=threadIdx.x+blockDim.x*blockIdx.x;   
     int index_ligne=threadIdx.y+blockDim.y*blockIdx.y;    
     int global_index;
     float x,y,xtemp,x0,y0;
     if ((index_col >= nb_col) || (index_ligne>=nb_ligne) ) return;
     global_index=index_ligne*nb_col+index_col;
     x0=((float)index_col/(float)nb_col)*(x_max-x_min)+x_min;
     y0=((float)(nb_ligne-index_ligne)/(float)nb_ligne)*(y_max-y_min)+y_min;
     x=0;y=0;
     while((x*x+y*y <= seuil) && (iter < max_ITER))
            {   xtemp = x*x-y*y+x0;
                y = 2*x*y+y0;
                x = xtemp;
                iter++;
            }
     res[global_index]=((float) iter/(float)max_ITER); 
}
