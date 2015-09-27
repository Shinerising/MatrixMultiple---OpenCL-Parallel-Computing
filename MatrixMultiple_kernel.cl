#define NX 2000
#define NY 2000
#define UNUM 1000

__kernel void matrixmultiple(__global float *A, __global float * B, __global float *C){	
	int id=get_global_id(0);																
	int i, a;																				
	float b;																				
	for(i=id*NX*NY/UNUM;i<(id+1)*NX*NY/UNUM;i++){												
		b=0;																				
		for(a=0;a<NX;a++)b+=A[(int)(i/NX)*NX+a]*B[a*NX+(i % NY)];								
		C[i]=b;																	
	}																						
}																							