#include <curand_kernel.h> 

//
//mettre devant le kernel pour que ce kernel puisse etre vu comme C et recuperé par Pycuda
// il faut le no_extern_c=true dans l'option compilation, pour que curand_kernel qui est C++ puisse être compilé
//
//
extern "C" {

__global__  void counthits(int n, uint *hitsp, unsigned decalage_index) {
    curandStatePhilox4_32_10_t state;
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned hits = 0;
    int tries=0;
    float x1 ;
    float y1 ;
    float x2 ;
    float y2 ;
// initialise avec un 'seed' egale a zero, choix de la suite pseudo aléatoire numero index+decalage_index (le decalage index permet de changer de suite si on relance...), et commence cette suite à partir de 0
/* Dans la doc Nvidia :
curand_init (
    unsigned long long seed, unsigned long long sequence,
    unsigned long long offset, curandState_t *state)*/
    curand_init(0, index+decalage_index, 0, &state);   
    float4 rand_vec;
    while (tries < n) {
        rand_vec=curand_uniform4 (&state);
	   x1 = 2*rand_vec.x-1;
        y1 = 2*rand_vec.y-1;
        x2 = 2*rand_vec.z-1;
        y2 = 2*rand_vec.w-1;
	if ( (x1*x1 + y1*y1) < 1) {
	    hits++;
	}
	if ( (x2*x2 + y2*y2) < 1 ) {
	    hits++;
	}
        tries+=2;
    }
    hitsp[index]=hits;
}


}
