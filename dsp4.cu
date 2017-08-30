/*
 ============================================================================
 Name        : dsp4.cu
 Author      : Hyrum
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include <unistd.h>
#include <complex>
#include "filt1.h"
#include "filt2.h"
#include "filt3.h"
#include "filt4.h"
#include "filtd.h"
#include "filt7.h"

/////
// Secret Code
// F: Carrier frequency (chosen)
// D: Decimation factor (given)
// L: filter length (given)
// LL: filter length after appended by zeros ( L + D - ( L % D ) )
// B: number of blocks used (chosen)
// S: Slice Size (chosen)
// T: number of threads per block ( ( LL / D ) * 2 )

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const int D1 = 2;
const int L1 = CL1;
const int B1 = 10000;
const int S1 = 800;
const int LL1 = L1 + D1 - (L1 % D1);
const int T1 = (LL1/D1)*2;
const int CS1 = B1 * S1; //size of initial chunk for stage 1

const float Bsh = 640;
const float Tsh = 640;
const float F = -((92.9-98.0)/20.0);//Fcarrier over Fsample_rate. In MHz.
const float COS0 = cos(2.0*M_PI*F*Bsh*Tsh);
const float SIN0 = sin(2.0*M_PI*F*Bsh*Tsh);
const float COS1 = cos(2.0*M_PI*F*(CS1/2));
const float SIN1 = sin(2.0*M_PI*F*(CS1/2));

const int D2 = 2;
const int L2 = CL2;
const int B2 = 5000;
const int S2 = 800;
const int LL2 = L2 + D2 - (L2 % D2);
const int T2 = (LL2/D2)*2;
const int CS2 = CS1/D1;

const int D3 = 5;
const int L3 = CL3;
const int B3 = 2500;
const int S3 = 800;
const int LL3 = L3 + D3 - (L3 % D3);
const int T3 = (LL3/D3)*2;
const int CS3 = CS2/D2;

const int D4 = 5;
const int L4 = CL4;
const int B4 = 500;
const int S4 = 800;
const int LL4 = L4 + D4 - (L4 % D4);
const int T4 = (LL4/D4)*2;
const int CS4 = CS3/D3;

const int LD = CDL;
const int BD = 100;
const int SD = 800;
const int LLD = 140;
const int DEL = (LD-1)/2;
const int TD = (LLD*2);
const int CSD = CS4/D4;

const int D7 = 5;
const int L7 = CL7;
const int B7 = 50;
const int S7 = 800;
const int LL7 = L7 + D7 - (L7 % D7);
const int T7 = LL7;
const int CS7 = CSD/2;

const int OS = CS7/D7;

union semun {
    int              val;    /* Value for SETVAL */
    struct semid_ds *buf;    /* Buffer for IPC_STAT, IPC_SET */
    unsigned short  *array;  /* Array for GETALL, SETALL */
    struct seminfo  *__buf;  /* Buffer for IPC_INFO (Linux-specific) */
};

class meminfo {
 public:
  int siz; // number of bytes per element of shared memory
  int chk; // number of elements in each chunk of shared memory
  int csz; // number of bytes in each chunk of shared memory
  int num; // number of chunks of shared memory
  int ele; // total number of elements in shraed memory
  int tot; // total number of bytes in shared memory
  meminfo(int sz, int ck, int nm) : siz(sz), chk(ck), num(nm) {
    csz = siz * chk;
    ele = chk * num;
    tot = ele * siz;
  }
  ~meminfo() {}
};

#define DIE(msg) { perror(msg); return 1; }

__global__ void init_arr(float* C, float* S)
//run this kernel with the same number of blocks and threads as hshift
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	C[tid] = cos(2.0*M_PI*F*tid);
	S[tid] = sin(2.0*M_PI*F*tid);
}

__global__ void hshift(float* CHUNK, float* S, float* C, int LL , float SIN0, float COS0, float SIN1, float COS1)
//Remember Amplitude Instability
{
	int rid = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = 2*rid;
	//int gid = 2 * blockDim.x * gridDim.x;
	//int chs = chunkSize+2*LL;
	float xr;
	float xi;
	float s = S[rid];
	float c = C[rid];
	float nexts = 0;
	float nextc = 0;
	while(tid < (CS1+2*LL))
	{
		xr = CHUNK[tid];
		xi = CHUNK[tid + 1];
		nexts = (COS0)*s + (SIN0)*c;
		nextc = (COS0)*c - (SIN0)*s;
		s = nexts;
		c = nextc;
		CHUNK[tid    ] = xr*c - xi*s;
		CHUNK[tid + 1] = xr*s + xi*c;
		tid += 2 * blockDim.x * gridDim.x;
	}
	s = S[rid];
	c = C[rid];
	nexts = (COS1)*s + (SIN1)*c;
	nextc = (COS1)*c - (SIN1)*s;
	S[rid] = nexts;
	C[rid] = nextc;
}

__global__ void tempBufSet(float* temp, int LL, int size)
/* prepares a temporary buffer for the next filter.*/
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < 2*LL) { temp[tid] = temp[tid + size]; }
}

__global__ void filter(float* COEF, float* inbuf, float* dbuf, int D, int LL, int numThreads, int sliceSize, int nextOff)
{
	extern __shared__ float CO[];
	int cid = threadIdx.x;
	while (cid < LL) { CO[cid] = COEF[cid]; cid += numThreads;}

	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int slice_start = sliceSize * bid;
	int slice_stop  = (sliceSize * (bid + 1)) + (2 * LL);
	float x;
	int m;
	int z = tid % 2;
	int dt = (tid/2)*D;
	int time = 0;
	float y = 0;
	for(int ix = (slice_start + z); ix < slice_stop; ix+=2)
	{
		x = inbuf[ix];
		m = dt - time + LL;
		y += CO[m % LL] * x;
		if(time == dt)
		{
			if(ix >= ((2 * LL) + slice_start)) { dbuf[(((((ix/2)-LL)/D)*2)+z) + (2*nextOff)] = y; }
			y = 0;
		}
		time = (time + 1) % LL;
	}
}

__global__ void ddfilt(float* COEF, float* in, float* diffout, float* delout, int LL, int slice_size, int delay, int offset)
//remember to add offset
{
	extern __shared__ float CO[];
	int cid = threadIdx.x;
	if (cid < LL) { CO[cid] = COEF[cid]; }

	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int slice_start = slice_size * bid;
	int slice_stop = (slice_size * (bid + 1)) + (2 * LL);
	float x;
	int m;
	bool ready;
	int z = tid % 2;
	int dt = tid / 2;
	int time = 0;
	float y = 0;
	float yy = 0;
	for(int ix = (slice_start + z); ix < slice_stop; ix+=2)
	{
		ready = (ix >= ((2 * LL) + slice_start));
		x = in[ix];
		m = (dt - time + LL) % LL;
		y += CO[m] * x;
		if(m == delay) yy = x;
		if(time == dt)
		{
			if(ready)
			{
				diffout[(((ix/2)-LL)*2)+z + (2*offset)] = y;
				delout[(((ix/2)-LL)*2)+z + (2*offset)] = yy;
			}
			y = 0;
		}
		time = (time + 1) % LL;
	}
}

__global__ void crossmult(float* diffin, float* delin, float* out, int size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int idx = tid * 2;
	int num = blockDim.x * gridDim.x;
	int inc = num * 2;
	while(idx < size)
	{
		out[idx/2] = (delin[idx] * diffin[idx + 1]) - (delin[idx + 1] * diffin[idx]);
		//out[idx/2] = (delin[idx+1] * diffin[idx]) - (delin[idx] * diffin[idx+1]);
		idx += inc;
	}
}

__global__ void oldTempBufSet(float* temp, int LL, int size)
/* prepares a temporary buffer for the next filter.*/
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < LL) { temp[tid] = temp[tid + size]; }
}

__global__ void deciconvo(float* COEF, float* inbuf, float* dbuf, int D, int LL, int numThreads, int sliceSize, int nextOff)
{
	extern __shared__ float CO[];
	int cid = threadIdx.x;
	while (cid < LL) { CO[cid] = COEF[cid]; cid += numThreads;}

	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int slice_start = sliceSize * bid;
	int slice_stop  = (sliceSize * (bid + 1)) + LL;
	float x;
	int m;
	int time = 0;
	float y = 0;
	for(int ix = slice_start; ix < slice_stop; ix++) {
		x = inbuf[ix];
		m = (tid * D) - time + LL;
		y += CO[m % LL] * x;
		if(time == tid*D)
		{
			if(ix >= (LL+slice_start)) { dbuf[((ix - LL) / D) + nextOff] = y; }
			y = 0;
		}
		time = (time + 1) % LL;
	}
}

void setup_filter(float *A, const float *G, int L, int LL)
// A: host coefficients, G: pointer from filt.h file
{
	memset(A,0,sizeof(float)*LL);
	for(int i=0; i< L; i++) { A[i] = 3.5f * G[i]; }
}

int main(void)
{
	cudaError_t err = cudaSuccess;
	float *h_out, *h_CO1, *h_CO2, *h_CO3, *h_CO4, *h_COD, *h_CO7;
	float *d_CHUNK, *d_temp12, *d_temp23, *d_temp34, *d_temp4s, *d_del, *d_diff, *d_temp7, *d_out;
	float *d_CO1, *d_CO2, *d_CO3, *d_CO4, *d_COD, *d_CO7, *S, *C;

	int semid = 0;
	int old_semid = semid;

	meminfo smi(sizeof(float), CS1, 20);
	int shmid;
	key_t shm_key = 48879;
	float *shm, *s;
	if ((shmid = shmget(shm_key, smi.tot, IPC_CREAT | 0666)) == -1) DIE("shmget");
	if ((shm = (float*)shmat(shmid, NULL, 0)) == (void*) -1) DIE("shmat");
	s = shm;

	int marker;
	key_t marker_key = 15;
	union semun markerset;
	markerset.val = 1;
	if((marker = semget(marker_key, 1, IPC_CREAT | 0666)) == -1) DIE("semget");
	if(semctl(marker,0,SETVAL,markerset) == -1) DIE("semctl");

	int sempty;
	key_t sempty_key = 64206;
	union semun semptyset;
	int sfull;
	key_t sfull_key = 57005;
	union semun sfullset;
	if ((sempty = semget(sempty_key, smi.num, IPC_CREAT | 0666)) == -1) DIE("semget");
	if ((sfull = semget(sfull_key, smi.num, IPC_CREAT | 0666)) == -1) DIE("semget");

	int tick = 0;
	size_t bytes = 0;

	/*Allocate host memory*/
	h_out = (float*)malloc(OS*sizeof(float));
	h_CO1 = (float*)malloc(LL1*sizeof(float));
	h_CO2 = (float*)malloc(LL2*sizeof(float));
	h_CO3 = (float*)malloc(LL3*sizeof(float));
	h_CO4 = (float*)malloc(LL4*sizeof(float));
	h_COD = (float*)malloc(LLD*sizeof(float));
	h_CO7 = (float*)malloc(LL7*sizeof(float));
	if(h_out == NULL || h_CO1 == NULL || h_CO2 == NULL || h_CO3 == NULL || h_CO4 == NULL || h_COD == NULL || h_CO7 == NULL)
	{fprintf(stderr, "Could not allocate host memory\n"); exit(EXIT_FAILURE);}

	/*Allocate device memory*/
	err = cudaMalloc((void**)&d_CO1, LL1*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to allocate CO1 on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMalloc((void**)&d_CO2, LL2*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to allocate CO2 on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMalloc((void**)&d_CO3, LL3*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to allocate CO3 on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMalloc((void**)&d_CO4, LL4*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to allocate CO4 on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMalloc((void**)&d_COD, LLD*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to allocate COD on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMalloc((void**)&d_CO7, LL7*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to allocate CO7 on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMalloc((void**)&d_CHUNK, (CS1+(2*LL1))*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to allocate CHUNK on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMalloc((void**)&d_temp12, (CS2+(2*LL2))*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to allocate temp12 on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMalloc((void**)&d_temp23, (CS3+(2*LL3))*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to allocate temp23 on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMalloc((void**)&d_temp34, (CS4+(2*LL4))*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to allocate temp34 on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMalloc((void**)&d_temp4s, (CSD+(2*LLD))*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to allocate temp4s on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMalloc((void**)&d_del, (CSD+(2*LL7))*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to allocate del on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMalloc((void**)&d_diff, (CSD+(2*LL7))*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to allocate diff on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMalloc((void**)&d_temp7, (CS7+LL7)*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to allocate temp7 on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMalloc((void**)&d_out, OS*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to allocate output buffer on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMalloc((void**)&S, (Bsh*Tsh)*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to allocate S on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMalloc((void**)&C, (Bsh*Tsh)*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to allocate C on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}

	/*set up S and C on device*/
	err = cudaMemset(S, 0, (Bsh*Tsh)*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to set S on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMemset(C, 0, (Bsh*Tsh)*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to set C on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	init_arr<<<Bsh, Tsh>>>(C, S);

	/*set up coefficients on device*/
	setup_filter(h_CO1, C1, L1, LL1);
	err = cudaMemcpy(d_CO1, h_CO1, LL1*sizeof(float), cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {fprintf(stderr, "Failed to copy CO1 from host to device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	setup_filter(h_CO2, C2, L2, LL2);
	err = cudaMemcpy(d_CO2, h_CO2, LL2*sizeof(float), cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {fprintf(stderr, "Failed to copy CO2 from host to device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	setup_filter(h_CO3, C3, L3, LL3);
	err = cudaMemcpy(d_CO3, h_CO3, LL3*sizeof(float), cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {fprintf(stderr, "Failed to copy CO3 from host to device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	setup_filter(h_CO4, C4, L4, LL4);
	err = cudaMemcpy(d_CO4, h_CO4, LL4*sizeof(float), cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {fprintf(stderr, "Failed to copy CO4 from host to device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	setup_filter(h_COD, CD, LD, LLD);
	err = cudaMemcpy(d_COD, h_COD, LLD*sizeof(float), cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {fprintf(stderr, "Failed to copy COD from host to device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	setup_filter(h_CO7, C7, L7, LL7);
	err = cudaMemcpy(d_CO7, h_CO7,LL7*sizeof(float), cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {fprintf(stderr, "Failed to copy CO7 from host to device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}

	/*set up temp buffers on device*/
	err = cudaMemset(d_temp12, 0, (CS2+(2*LL2))*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to set temp12 on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMemset(d_temp23, 0, (CS3+(2*LL3))*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to set temp23 on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMemset(d_temp34, 0, (CS4+(2*LL4))*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to set temp34 on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMemset(d_temp4s, 0, (CSD+(2*LLD))*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to set temp4s on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMemset(d_del, 0, (CSD+(2*LL7))*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to set del on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMemset(d_diff, 0, (CSD+(2*LL7))*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to set diff on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
	err = cudaMemset(d_temp7, 0, (CS7+LL7)*sizeof(float));
	if(err != cudaSuccess) {fprintf(stderr, "Failed to set temp7 on device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}


	while(semctl(marker,0,GETVAL))
	{
		if(semctl(sfull,semid,GETVAL) == 1)
		{
			//printf("Chunk %d is full\n", semid);
			sfullset.val = 0;
			if(semctl(sfull,semid,SETVAL,sfullset) == -1) {fprintf(stderr,"semctl %d\n",semid); DIE("semctl");}

			tick += 1;
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start, 0);

			err = cudaMemcpy(d_CHUNK, s + (semid*smi.chk), CS1*sizeof(float), cudaMemcpyHostToDevice);
			if(err != cudaSuccess) {fprintf(stderr, "Failed to copy CHUNK to device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
			semid = (semid + 1) % smi.num;
			err = cudaMemcpy(d_CHUNK + CS1, s+(semid*smi.chk), 2*LL1*sizeof(float), cudaMemcpyHostToDevice);
			if(err != cudaSuccess) {fprintf(stderr, "Failed to copy CHUNK to device (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}

			/*Set up temp buffers*/
			tempBufSet<<<B2, T2>>>(d_temp12, LL2, CS2);
			tempBufSet<<<B3, T3>>>(d_temp23, LL3, CS3);
			tempBufSet<<<B4, T4>>>(d_temp34, LL4, CS4);
			tempBufSet<<<BD, TD>>>(d_temp4s, LLD, CSD);
			tempBufSet<<<BD, TD>>>(d_del, LL7, CSD);
			tempBufSet<<<BD, TD>>>(d_diff, LL7, CSD);
			oldTempBufSet<<<B7, T7>>>(d_temp7, LL7, CS7);

			/*kernel calls*/
			hshift<<<Bsh, Tsh>>>(d_CHUNK, S, C, LL1, SIN0, COS0, SIN1, COS1);
			filter<<<B1, T1, LL1*sizeof(float)>>>(d_CO1, d_CHUNK, d_temp12, D1, LL1, T1, S1, LL2);
			filter<<<B2, T2, LL2*sizeof(float)>>>(d_CO2, d_temp12, d_temp23, D2, LL2, T2, S2, LL3);
			filter<<<B3, T3, LL3*sizeof(float)>>>(d_CO3, d_temp23, d_temp34, D3, LL3, T3, S3, LL4);
			filter<<<B4, T4, LL4*sizeof(float)>>>(d_CO4, d_temp34, d_temp4s, D4, LL4, T4, S4, LLD);
			ddfilt<<<BD, TD, LLD*sizeof(float)>>>(d_COD, d_temp4s, d_diff, d_del, LLD, SD, DEL, LL7);
			crossmult<<<16, 16>>>(d_diff, d_del, d_temp7, (CSD+(2*LL7)));
			deciconvo<<<B7, T7, LL7*sizeof(float)>>>(d_CO7, d_temp7, d_out, D7, LL7, T7, S7, 0);

			/*write output buffer to file*/
			cudaMemcpy(h_out, d_out, OS*sizeof(float), cudaMemcpyDeviceToHost);
			if(err != cudaSuccess) {fprintf(stderr, "Failed to copy output buffer from device to host (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
			bytes = fwrite(h_out, sizeof(float), OS, stdout);
			//fprintf(stderr,"bytes: %lu\n", bytes);

			//fprintf(stderr, "Total bytes: %d\n", bytes);
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			float elapsedTime;
			cudaEventElapsedTime(&elapsedTime, start, stop);
			fprintf(stderr,"Elapsed time: %f\n", elapsedTime);
			cudaEventDestroy(start);
			cudaEventDestroy(stop);

			semptyset.val = 1;
			if(semctl(sempty, old_semid, SETVAL, semptyset) == -1) {fprintf(stderr,"semctl %d\n",old_semid); DIE("semctl");}
			//printf("Chunk %d is empty\n", old_semid);
			old_semid = semid;
		}
	}

	//printf("Number of times kernels were called: %d\n",tick);

	cudaFree(d_CHUNK);
	cudaFree(d_temp12);
	cudaFree(d_temp23);
	cudaFree(d_temp34);
	cudaFree(d_temp4s);
	cudaFree(d_del);
	cudaFree(d_diff);
	cudaFree(d_temp7);
	cudaFree(d_out);
	cudaFree(d_CO1);
	cudaFree(d_CO2);
	cudaFree(d_CO3);
	cudaFree(d_CO4);
	cudaFree(d_COD);
	cudaFree(d_CO7);
	cudaFree(S);
	cudaFree(C);
	free(h_out);
	free(h_CO1);
	free(h_CO2);
	free(h_CO3);
	free(h_CO4);
	free(h_COD);
	free(h_CO7);
}
