#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <iomanip>
#include <conio.h>
//#include <cutil.h>					// cutil32.lib
//#include <cutil_math.h>				// cutil32.lib


#include <GL/glut.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <driver_types.h>


#include "fluid_system_host.cuh"		
#include "fluid_system_kern.cuh"
#include "radixsort.cu"						// Build in RadixSort
#include "thrust\device_vector.h"	//thrust libs
#include "thrust\sort.h" 
#include "thrust\host_vector.h"

#include "cublas_v2.h" 


FluidParams		fcuda;
bufList			fbuf;
//initialInfo		elasticInfo;
__device__ FluidParams	simData;
__device__ uint			gridActive;
__device__ int			flagNumFT;  //for transfer
__device__ int			pNumFT;		//for transfer

#define BLOCK_SIZE 256
#define LOCAL_PMAX		896
#define NUM_CELL		27
#define LAST_CELL		26
#define CENTER_CELL		13

float**			g_scanBlockSums;
int**			g_scanBlockSumsInt;
unsigned int	g_numEltsAllocated = 0;
unsigned int	g_numLevelsAllocated = 0;

void cudaExit (int argc, char **argv)
{
	exit(EXIT_SUCCESS);
	//CUT_EXIT(argc, argv); 
}
void cudaInit(int argc, char **argv)
{   
	//CUT_DEVICE_INIT(argc, argv);
	findCudaDevice(argc, (const char **)argv);
	cudaDeviceProp p;
	cudaGetDeviceProperties ( &p, 0);
	
	printf ( "-- CUDA --\n" );
	printf ( "Name:       %s\n", p.name );
	printf ( "Revision:   %d.%d\n", p.major, p.minor );
	printf ( "Global Mem: %d\n", p.totalGlobalMem );
	printf ( "Shared/Blk: %d\n", p.sharedMemPerBlock );
	printf ( "Regs/Blk:   %d\n", p.regsPerBlock );
	printf ( "Warp Size:  %d\n", p.warpSize );
	printf ( "Mem Pitch:  %d\n", p.memPitch );
	printf ( "Thrds/Blk:  %d\n", p.maxThreadsPerBlock );
	printf ( "Const Mem:  %d\n", p.totalConstMem );
	printf ( "Clock Rate: %d\n", p.clockRate );	

	fbuf.mgridactive = 0x0;

	checkCudaErrors ( cudaMalloc ( (void**) &fbuf.mpos, sizeof(float)*3 ) );	
	checkCudaErrors ( cudaMalloc ( (void**) &fbuf.maccel, sizeof(float)*3) );	
	checkCudaErrors ( cudaMalloc((void**)&fbuf.vel_mid, sizeof(float) * 3));
	checkCudaErrors ( cudaMalloc ( (void**) &fbuf.mveleval, sizeof(float)*3) );	
	checkCudaErrors ( cudaMalloc ( (void**) &fbuf.mforce, sizeof(float)*3) );	
	checkCudaErrors ( cudaMalloc ( (void**) &fbuf.poroForce, sizeof(float) * 3));
	checkCudaErrors(cudaMalloc((void**)&fbuf.fluidForce, sizeof(float) * 3));
	checkCudaErrors ( cudaMalloc ( (void**) &fbuf.mpress, sizeof(float) ) );	
	checkCudaErrors ( cudaMalloc ( (void**) &fbuf.mdensity, sizeof(float) ) );	
	checkCudaErrors ( cudaMalloc ( (void**) &fbuf.mgcell, sizeof(uint)) );	
	checkCudaErrors ( cudaMalloc ( (void**) &fbuf.mgndx, sizeof(uint)) );	
	checkCudaErrors ( cudaMalloc ( (void**) &fbuf.mclr, sizeof(uint)) );	

	checkCudaErrors ( cudaMalloc((void**)&fbuf.delta_density, sizeof(float)));
	checkCudaErrors ( cudaMalloc((void**)&fbuf.aii, sizeof(float)));

	checkCudaErrors(cudaMalloc((void**)&fbuf.pressForce, sizeof(float) * 3));

	checkCudaErrors ( cudaMalloc((void**)&fbuf.rest_volume, sizeof(float)));
	checkCudaErrors ( cudaMalloc((void**)&fbuf.volume, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.rest_colorValue, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.colorValue, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.colorTensor, sizeof(float)));

	checkCudaErrors(cudaMalloc((void**)&fbuf.source, sizeof(float)));

	checkCudaErrors ( cudaMalloc ( (void**) &fbuf.msortbuf, sizeof(uint) ) );	

	checkCudaErrors ( cudaMalloc ( (void**) &fbuf.mgrid, 1 ) );
	checkCudaErrors ( cudaMalloc ( (void**) &fbuf.mgridcnt, 1 ) );

	//new sort
	checkCudaErrors ( cudaMalloc ( (void**) &fbuf.midsort, 1 ) );

	checkCudaErrors ( cudaMalloc ( (void**) &fbuf.mgridoff, 1 ) );	
	checkCudaErrors ( cudaMalloc ( (void**) &fbuf.mgridactive, 1 ) );

	//checkCudaErrors ( cudaMalloc ( (void**) &fbuf.mcluster, sizeof(uint) ) );	
	//implicit SPH formulation for elastic body
	checkCudaErrors ( cudaMalloc ( (void**) &fbuf.gradDeform, 1 ));
	checkCudaErrors ( cudaMalloc ( (void**) &fbuf.Rotation, 1));
	//checkCudaErrors ( cudaMalloc ( (void**) &fbuf.mf_fluidPercent, sizeof(float)));
	//checkCudaErrors ( cudaMalloc ( (void**) &fbuf.poroDriftVel, sizeof(float3)));
	//checkCudaErrors ( cudaMalloc ( (void**) &fbuf.percentChange, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.divDarcyFlux, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.isInside, sizeof(bool)));
	//checkCudaErrors ( cudaMalloc ( (void**) &fbuf.CorrectL, 1 ) );
	checkCudaErrors(cudaMalloc((void**)&fbuf.SurfaceForce, sizeof(float3)));
	//elastic information
	checkCudaErrors(cudaMalloc((void**)&fbuf.elasticID, sizeof(uint)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.particleID, sizeof(uint)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.initialVolume, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.neighborID, sizeof(uint)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.neighborDistance, sizeof(float3)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.kernelGrad, sizeof(float3)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.kernelRotate, sizeof(float3)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.neighborNum, sizeof(uint)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.neighborIndex, sizeof(uint)));
	//checkCudaErrors(cudaMalloc((void**)&fbuf.colorField, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.volumetricStrain, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.normal, sizeof(float3)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.isHead, sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.frame, sizeof(int)));

	checkCudaErrors(cudaMalloc((void**)&fbuf.bx, sizeof(float))); checkCudaErrors(cudaMalloc((void**)&fbuf.by, sizeof(float))); checkCudaErrors(cudaMalloc((void**)&fbuf.bz, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.vx, sizeof(float))); checkCudaErrors(cudaMalloc((void**)&fbuf.vy, sizeof(float))); checkCudaErrors(cudaMalloc((void**)&fbuf.vz, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.rx, sizeof(float))); checkCudaErrors(cudaMalloc((void**)&fbuf.ry, sizeof(float))); checkCudaErrors(cudaMalloc((void**)&fbuf.rz, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.r2x, sizeof(float))); checkCudaErrors(cudaMalloc((void**)&fbuf.r2y, sizeof(float))); checkCudaErrors(cudaMalloc((void**)&fbuf.r2z, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.px, sizeof(float))); checkCudaErrors(cudaMalloc((void**)&fbuf.py, sizeof(float))); checkCudaErrors(cudaMalloc((void**)&fbuf.pz, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.Apx, sizeof(float))); checkCudaErrors(cudaMalloc((void**)&fbuf.Apy, sizeof(float))); checkCudaErrors(cudaMalloc((void**)&fbuf.Apz, sizeof(float)));

	//porous
	//checkCudaErrors(cudaMalloc((void**)&fbuf.porosity, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.density_solid, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.pressure_water, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.totalDis, sizeof(float)*MAX_SOLIDNUM));
	checkCudaErrors(cudaMalloc((void**)&fbuf.solidCount, sizeof(int)));

	//checkCudaErrors(cudaMalloc((void**)&fbuf.Saturation, sizeof(float)));
	//checkCudaErrors(cudaMalloc((void**)&fbuf.AbsorbedFluidVolume, sizeof(float)));
	//checkCudaErrors(cudaMalloc((void**)&fbuf.Saturation, sizeof(float)));
	//checkCudaErrors(cudaMalloc((void**)&fbuf.DeltaSaturation, sizeof(float)));
	//checkCudaErrors(cudaMalloc((void**)&fbuf.elasticVolume, sizeof(float)));
	//checkCudaErrors(cudaMalloc((void**)&fbuf.gradPressure, sizeof(float3)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.poroVel, sizeof(float3)));
	//checkCudaErrors(cudaMalloc((void**)&fbuf.fluidVel, sizeof(float3)));

	preallocBlockSumsInt ( 1 );
};

int iDivUp (int a, int b) {
	return (a % b != 0) ? (a / b + 1) : (a / b);
}
inline bool isPowerOfTwo(int n) { return ((n&(n-1))==0) ; }
inline int floorPow2(int n) {
	#ifdef WIN32
		return 1 << (int)logb((float)n);
	#else
		int exp;
		frexp((float)n, &exp);
		return 1 << (exp - 1);
	#endif
}

// Compute number of blocks to create
void computeNumBlocks (int numPnts, int maxThreads, int &numBlocks, int &numThreads)
{
	numThreads = min( maxThreads, numPnts );
	numBlocks = iDivUp ( numPnts, numThreads );
}

void FluidClearCUDA ()
{
	checkCudaErrors ( cudaFree ( fbuf.mpos ) );	
	checkCudaErrors ( cudaFree ( fbuf.maccel ) );	
	checkCudaErrors ( cudaFree ( fbuf.vel_mid));
	checkCudaErrors ( cudaFree ( fbuf.mveleval ) );	
	checkCudaErrors ( cudaFree ( fbuf.mforce ) );	
	checkCudaErrors ( cudaFree ( fbuf.poroForce));
	checkCudaErrors(cudaFree(fbuf.fluidForce));
	checkCudaErrors ( cudaFree ( fbuf.mpress ) );	
	checkCudaErrors ( cudaFree ( fbuf.mdensity ) );		
	checkCudaErrors ( cudaFree ( fbuf.mgcell ) );	
	checkCudaErrors ( cudaFree ( fbuf.mgndx ) );	
	checkCudaErrors ( cudaFree ( fbuf.mclr ) );	

#ifdef NEW_BOUND
	checkCudaErrors ( cudaFree ( fbuf.misbound ) );	
#endif
	//checkCudaErrors ( cudaFree ( fbuf.mcluster ) );	

	//multi fluid
	checkCudaErrors ( cudaFree ( fbuf.mf_alpha ) );
	checkCudaErrors ( cudaFree ( fbuf.mf_alpha_next ) );
	//checkCudaErrors ( cudaFree ( fbuf.mf_pressure_modify ) );
	checkCudaErrors ( cudaFree ( fbuf.mf_vel_phrel) );
	checkCudaErrors ( cudaFree ( fbuf.mf_restdensity ) );
	checkCudaErrors ( cudaFree ( fbuf.mf_restdensity_out));
	checkCudaErrors ( cudaFree ( fbuf.mf_restmass ) );
	checkCudaErrors ( cudaFree ( fbuf.mf_alpha_sum));
	checkCudaErrors ( cudaFree ( fbuf.mf_visc ) );
	//checkCudaErrors ( cudaFree ( fbuf.mf_velxcor ) );
	//checkCudaErrors ( cudaFree ( fbuf.mf_alphagrad ) );
	checkCudaErrors(cudaFree(fbuf.mf_alphachange));
	//checkCudaErrors ( cudaFree ( fbuf.density_fluid ) );

	checkCudaErrors ( cudaFree ( fbuf.msortbuf ) );	

	checkCudaErrors ( cudaFree ( fbuf.mgrid ) );
	checkCudaErrors ( cudaFree ( fbuf.mgridcnt ) );
	//new sort
	checkCudaErrors ( cudaFree ( fbuf.midsort ) );

	checkCudaErrors ( cudaFree ( fbuf.mgridoff ) );
	checkCudaErrors ( cudaFree ( fbuf.mgridactive ) );
	//an implicit SPH formulation for elastic body
	checkCudaErrors ( cudaFree(fbuf.gradDeform));
	checkCudaErrors ( cudaFree(fbuf.elasticID));
	checkCudaErrors ( cudaFree(fbuf.Rotation));
	//checkCudaErrors ( cudaFree(fbuf.mf_fluidPercent));
	//checkCudaErrors ( cudaFree(fbuf.poroDriftVel));
	//checkCudaErrors ( cudaFree(fbuf.percentChange));
	checkCudaErrors(cudaFree(fbuf.divDarcyFlux));
	checkCudaErrors(cudaFree(fbuf.isInside));
	//checkCudaErrors(cudaFree(fbuf.CorrectL));
	checkCudaErrors(cudaFree(fbuf.SurfaceForce));
	//elastic information
	checkCudaErrors(cudaFree(fbuf.particleID));
	checkCudaErrors(cudaFree(fbuf.initialVolume));
	checkCudaErrors(cudaFree(fbuf.neighborNum));
	checkCudaErrors(cudaFree(fbuf.neighborID));
	checkCudaErrors(cudaFree(fbuf.neighborDistance));
	checkCudaErrors(cudaFree(fbuf.kernelGrad));
	checkCudaErrors(cudaFree(fbuf.kernelRotate));
	checkCudaErrors(cudaFree(fbuf.neighborIndex));
	//checkCudaErrors(cudaFree(fbuf.colorField));
	checkCudaErrors(cudaFree(fbuf.volumetricStrain));

	checkCudaErrors(cudaFree(fbuf.bx)); checkCudaErrors(cudaFree(fbuf.by)); checkCudaErrors(cudaFree(fbuf.bz));
	checkCudaErrors(cudaFree(fbuf.vx)); checkCudaErrors(cudaFree(fbuf.vy)); checkCudaErrors(cudaFree(fbuf.vz));
	checkCudaErrors(cudaFree(fbuf.rx)); checkCudaErrors(cudaFree(fbuf.ry)); checkCudaErrors(cudaFree(fbuf.rz));
	checkCudaErrors(cudaFree(fbuf.r2x)); checkCudaErrors(cudaFree(fbuf.r2y)); checkCudaErrors(cudaFree(fbuf.r2z));
	checkCudaErrors(cudaFree(fbuf.px)); checkCudaErrors(cudaFree(fbuf.py)); checkCudaErrors(cudaFree(fbuf.pz));
	checkCudaErrors(cudaFree(fbuf.Apx)); checkCudaErrors(cudaFree(fbuf.Apy)); checkCudaErrors(cudaFree(fbuf.Apz));

	checkCudaErrors(cudaFree(fbuf.normal));
	checkCudaErrors(cudaFree(fbuf.isHead));
	checkCudaErrors(cudaFree(fbuf.frame));


	checkCudaErrors(cudaFree(fbuf.isSurface));
	//porous
	//checkCudaErrors(cudaFree(fbuf.porosity));
	checkCudaErrors(cudaFree(fbuf.density_solid));
	checkCudaErrors(cudaFree(fbuf.pressure_water));
	checkCudaErrors(cudaFree(fbuf.solidCount));
	checkCudaErrors(cudaFree(fbuf.totalDis));

	//checkCudaErrors(cudaFree(fbuf.AbsorbedFluidVolume));
	//checkCudaErrors(cudaFree(fbuf.Saturation));
	//checkCudaErrors(cudaFree(fbuf.DeltaSaturation));
	//checkCudaErrors(cudaFree(fbuf.elasticVolume));
	//checkCudaErrors(cudaFree(fbuf.gradPressure));
	checkCudaErrors(cudaFree(fbuf.poroVel));
	//checkCudaErrors(cudaFree(fbuf.fluidVel));

	//IISPH
	checkCudaErrors(cudaFree(fbuf.aii));
	checkCudaErrors(cudaFree(fbuf.pressForce));

	checkCudaErrors(cudaFree(fbuf.delta_density));
	//pressure boundary for IISPH
	checkCudaErrors(cudaFree(fbuf.volume));
	checkCudaErrors(cudaFree(fbuf.rest_volume));
	checkCudaErrors(cudaFree(fbuf.source));
	checkCudaErrors(cudaFree(fbuf.colorValue));
	checkCudaErrors(cudaFree(fbuf.rest_colorValue));
}
void FluidSetupRotationCUDA ( float pan_r,float omega,int loadwhich, float capillaryForceRatio)
{
	fcuda.pan_r = pan_r;
	fcuda.omega = omega;
	fcuda.loadwhich = loadwhich;
	fcuda.capillaryForceRatio = capillaryForceRatio;
}
float FluidSetupCUDA ( int num, int gsrch, int3 res, float3 size, float3 delta, float3 gmin, float3 gmax, int total, int chk)
{	
	float cudaMem = 0;
	fcuda.pnum = num;	
	fcuda.gridRes = res;
	fcuda.gridSize = size;
	fcuda.gridDelta = delta;
	fcuda.gridMin = gmin;
	fcuda.gridMax = gmax;
	fcuda.gridTotal = total;
	fcuda.gridSrch = gsrch;
	fcuda.gridAdjCnt = gsrch*gsrch*gsrch;
	fcuda.gridScanMax = res;
	fcuda.gridScanMax -= make_int3( fcuda.gridSrch, fcuda.gridSrch, fcuda.gridSrch );
	fcuda.chk = chk;
	fcuda.mf_up=0;

	// Build Adjacency Lookup
	int cell = 0;
	for (int y=0; y < gsrch; y++ ) 
		for (int z=0; z < gsrch; z++ ) 
			for (int x=0; x < gsrch; x++ ) 
				fcuda.gridAdj [ cell++]  = ( y * fcuda.gridRes.z+ z )*fcuda.gridRes.x +  x ;			
	
	printf ( "CUDA Adjacency Table\n");
	for (int n=0; n < fcuda.gridAdjCnt; n++ ) {
		printf ( "  ADJ: %d, %d\n", n, fcuda.gridAdj[n] );
	}	
	// Compute number of blocks and threads
	computeNumBlocks ( fcuda.pnum, 384, fcuda.numBlocks, fcuda.numThreads);			// particles
	computeNumBlocks ( fcuda.gridTotal, 384, fcuda.gridBlocks, fcuda.gridThreads);		// grid cell
	// Allocate particle buffers
	fcuda.szPnts = (fcuda.numBlocks  * fcuda.numThreads);     
	printf ( "CUDA Allocate: \n" );
	printf ( "  Pnts: %d, t:%dx%d=%d, Size:%d\n", fcuda.pnum, fcuda.numBlocks, fcuda.numThreads, fcuda.numBlocks*fcuda.numThreads, fcuda.szPnts);
	printf ( "  Grid: %d, t:%dx%d=%d, bufGrid:%d, Res: %dx%dx%d\n", fcuda.gridTotal, fcuda.gridBlocks, fcuda.gridThreads, fcuda.gridBlocks*fcuda.gridThreads, fcuda.szGrid, (int) fcuda.gridRes.x, (int) fcuda.gridRes.y, (int) fcuda.gridRes.z );		
	
	checkCudaErrors ( cudaMalloc ( (void**) &fbuf.mpos, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float) * 3));
	checkCudaErrors(cudaMalloc((void**)&fbuf.mveleval, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float) * 3));
	checkCudaErrors(cudaMalloc((void**)&fbuf.mpress, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.mgcell, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(uint)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.mgndx, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(uint)));

	int temp_size = EMIT_BUF_RATIO*(2 * (sizeof(float) * 3) + sizeof(float)+ 2 *sizeof(uint));

#ifdef NEW_BOUND
	checkCudaErrors ( cudaMalloc ( (void**) &fbuf.misbound, EMIT_BUF_RATIO*fcuda.szPnts*sizeof(int)) );	
	temp_size += EMIT_BUF_RATIO*sizeof(int);
#endif

	//multi fluid
	checkCudaErrors ( cudaMalloc ( (void**) &fbuf.mf_alpha,					EMIT_BUF_RATIO*fcuda.szPnts*sizeof(float)*MAX_FLUIDNUM ));    //float* num
	checkCudaErrors ( cudaMalloc ( (void**) &fbuf.mf_alpha_next,			EMIT_BUF_RATIO*fcuda.szPnts*sizeof(float)*MAX_FLUIDNUM ) );    //float* num
	checkCudaErrors ( cudaMalloc ( (void**) &fbuf.mf_restmass,				EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float)));
	temp_size += EMIT_BUF_RATIO*(2*MAX_FLUIDNUM*sizeof(float) + sizeof(float));
	checkCudaErrors ( cudaMalloc ( (void**) &fbuf.MFtype,					EMIT_BUF_RATIO*fcuda.szPnts*sizeof(int) ) ); //indicator function
	temp_size += EMIT_BUF_RATIO*(sizeof(int));
	//an implicit SPH formulation for elastic body
	checkCudaErrors ( cudaMalloc ( (void**)&fbuf.elasticID,					EMIT_BUF_RATIO*fcuda.szPnts * sizeof(uint)));
	checkCudaErrors ( cudaMalloc ( (void**)&fbuf.mf_beta,			EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float)*MAX_FLUIDNUM * MAX_SOLIDNUM));
	checkCudaErrors ( cudaMalloc ( (void**)&fbuf.mf_beta_next,				EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float) * MAX_FLUIDNUM * MAX_SOLIDNUM));
	temp_size += EMIT_BUF_RATIO*(2*sizeof(float)*MAX_FLUIDNUM* MAX_SOLIDNUM +sizeof(uint));
	
	checkCudaErrors ( cudaMalloc ( (void**) &fbuf.msortbuf,	EMIT_BUF_RATIO*fcuda.szPnts*temp_size ) );	

	cudaMem += EMIT_BUF_RATIO*fcuda.szPnts*temp_size * 2;
	//no sort values
	checkCudaErrors(cudaMalloc((void**)&fbuf.density_solid, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.gradDeform, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float) * 9));
	checkCudaErrors(cudaMalloc((void**)&fbuf.Rotation, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float) * 9));
	checkCudaErrors(cudaMalloc((void**)&fbuf.mf_vel_phrel, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float) * 3 * MAX_FLUIDNUM));	//float*3*num
	checkCudaErrors(cudaMalloc((void**)&fbuf.mf_restdensity, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.mf_restdensity_out, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float)));
	cudaMem += EMIT_BUF_RATIO*fcuda.szPnts *(21 * sizeof(float) + sizeof(float) * 3 * MAX_FLUIDNUM);

	checkCudaErrors(cudaMalloc((void**)&fbuf.mf_alpha_sum, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.mf_visc, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.maccel, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float) * 3));
	checkCudaErrors(cudaMalloc((void**)&fbuf.mforce, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float) * 3));
	checkCudaErrors(cudaMalloc((void**)&fbuf.mdensity, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.mgcell, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(uint)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.mgndx, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(uint)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.mclr, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(uint)));
	cudaMem += EMIT_BUF_RATIO*fcuda.szPnts * (12 * sizeof(float));

	checkCudaErrors(cudaMalloc((void**)&fbuf.mf_alphachange, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float)*MAX_FLUIDNUM));    //float* num
	checkCudaErrors(cudaMalloc((void**)&fbuf.vel_mid, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float) * 3));
	checkCudaErrors(cudaMalloc((void**)&fbuf.poroForce, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float) * 3));
	checkCudaErrors(cudaMalloc((void**)&fbuf.fluidForce, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float) * 3));
	cudaMem += EMIT_BUF_RATIO*fcuda.szPnts * (9 * sizeof(float) + sizeof(float)*MAX_FLUIDNUM);

	checkCudaErrors(cudaMalloc((void**)&fbuf.pressure_water, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float)*MAX_FLUIDNUM*MAX_SOLIDNUM));
	checkCudaErrors(cudaMalloc((void**)&fbuf.gradPressure, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float3)*MAX_FLUIDNUM*MAX_SOLIDNUM));
	checkCudaErrors(cudaMalloc((void**)&fbuf.totalDis, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float)*MAX_SOLIDNUM));
	checkCudaErrors(cudaMalloc((void**)&fbuf.solidCount, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(int)*MAX_SOLIDNUM));
	checkCudaErrors(cudaMalloc((void**)&fbuf.divDarcyFlux, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float)*MAX_FLUIDNUM*MAX_SOLIDNUM));
	checkCudaErrors(cudaMalloc((void**)&fbuf.isInside, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float)));
	cudaMem += EMIT_BUF_RATIO*fcuda.szPnts * ( sizeof(float) + 2* sizeof(float)*MAX_SOLIDNUM + 5 * sizeof(float)*MAX_FLUIDNUM*MAX_SOLIDNUM);

	checkCudaErrors(cudaMalloc((void**)&fbuf.SurfaceForce, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float3)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.aii, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.delta_density, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.pressForce, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float) * 3));
	cudaMem += EMIT_BUF_RATIO*fcuda.szPnts * (8 * sizeof(float));

	//pressure boundary for IISPH
	checkCudaErrors(cudaMalloc((void**)&fbuf.rest_volume, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.volume, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.source, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.colorValue, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.rest_colorValue, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.colorTensor, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float) * 9));
	cudaMem += EMIT_BUF_RATIO*fcuda.szPnts * (14 * sizeof(float));
	checkCudaErrors(cudaMalloc((void**)&fbuf.poroVel, EMIT_BUF_RATIO*fcuda.szPnts * sizeof(float3)*MAX_FLUIDNUM*MAX_SOLIDNUM));
	// Allocate grid
	fcuda.szGrid = (fcuda.gridBlocks * fcuda.gridThreads);  
	checkCudaErrors ( cudaMalloc ( (void**) &fbuf.mgrid,		EMIT_BUF_RATIO*fcuda.szPnts*sizeof(int) ) );
	checkCudaErrors ( cudaMalloc ( (void**) &fbuf.mgridcnt,	fcuda.szGrid*sizeof(int) ) );
	//new sort
	checkCudaErrors ( cudaMalloc ( (void**) &fbuf.midsort,	EMIT_BUF_RATIO*fcuda.szPnts*sizeof(uint) ) );

	checkCudaErrors ( cudaMalloc ( (void**) &fbuf.mgridoff,	fcuda.szGrid*sizeof(int) ) );
	checkCudaErrors ( cudaMalloc ( (void**) &fbuf.mgridactive, fcuda.szGrid*sizeof(int) ) );
	checkCudaErrors ( cudaMemcpyToSymbol ( simData, &fcuda, sizeof(FluidParams) ) );

	cudaThreadSynchronize ();

	// Prefix Sum - Preallocate Block sums for Sorting
	deallocBlockSumsInt ();
	preallocBlockSumsInt ( fcuda.gridTotal );

	return cudaMem;
}
float ElasticSetupCUDA(int num,float miu,float lambda,float porosity,float* permeabilityRatio,int maxNeighborNum, float *pressRatio, float stRatio)
{
	float CudaMem = 0;
	fcuda.numElasticPoints = num;
	fcuda.maxNeighborNum = maxNeighborNum;
	printf("max neighbor num is %d\n",maxNeighborNum);
	fcuda.miu = miu;
	fcuda.lambda = lambda;
	fcuda.rest_porosity = porosity;
	fcuda.stRatio = stRatio;
	for (int i = 0; i < MAX_FLUIDNUM*MAX_SOLIDNUM; ++i)
	{
		fcuda.mf_permeability[i] = permeabilityRatio[i];
		//printf("permeability %d:%15f\n", i, permeabilityRatio[i]);
		std::cout << "permeability " << i << ":"  << 10000000000*permeabilityRatio[i];
		fcuda.pressRatio[i] = pressRatio[i];
		printf("pressure ratio:%f\n", fcuda.pressRatio[i]);
	}
	//fcuda.rest_permeability = permeability;
	//elastic information
	checkCudaErrors(cudaMalloc((void**)&fbuf.particleID, fcuda.numElasticPoints *sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.neighborNum, fcuda.numElasticPoints * sizeof(uint)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.initialVolume, fcuda.numElasticPoints *sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.normal, fcuda.numElasticPoints * sizeof(float3)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.isSurface, fcuda.numElasticPoints * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.isHead, fcuda.numElasticPoints * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.frame, fcuda.numElasticPoints * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.volumetricStrain, fcuda.numElasticPoints * sizeof(float)));
	CudaMem += fcuda.numElasticPoints * (7 * sizeof(float) + sizeof(float3));

	checkCudaErrors(cudaMalloc((void**)&fbuf.bx, fcuda.numElasticPoints * sizeof(float))); checkCudaErrors(cudaMalloc((void**)&fbuf.by, fcuda.numElasticPoints * sizeof(float))); checkCudaErrors(cudaMalloc((void**)&fbuf.bz, fcuda.numElasticPoints * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.vx, fcuda.numElasticPoints * sizeof(float))); checkCudaErrors(cudaMalloc((void**)&fbuf.vy, fcuda.numElasticPoints * sizeof(float))); checkCudaErrors(cudaMalloc((void**)&fbuf.vz, fcuda.numElasticPoints * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.rx, fcuda.numElasticPoints * sizeof(float))); checkCudaErrors(cudaMalloc((void**)&fbuf.ry, fcuda.numElasticPoints * sizeof(float))); checkCudaErrors(cudaMalloc((void**)&fbuf.rz, fcuda.numElasticPoints * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.r2x, fcuda.numElasticPoints * sizeof(float))); checkCudaErrors(cudaMalloc((void**)&fbuf.r2y, fcuda.numElasticPoints * sizeof(float))); checkCudaErrors(cudaMalloc((void**)&fbuf.r2z, fcuda.numElasticPoints * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.px, fcuda.numElasticPoints * sizeof(float))); checkCudaErrors(cudaMalloc((void**)&fbuf.py, fcuda.numElasticPoints * sizeof(float))); checkCudaErrors(cudaMalloc((void**)&fbuf.pz, fcuda.numElasticPoints * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&fbuf.Apx, fcuda.numElasticPoints * sizeof(float))); checkCudaErrors(cudaMalloc((void**)&fbuf.Apy, fcuda.numElasticPoints * sizeof(float))); checkCudaErrors(cudaMalloc((void**)&fbuf.Apz, fcuda.numElasticPoints * sizeof(float)));

	checkCudaErrors(cudaMalloc((void**)&fbuf.neighborID, fcuda.numElasticPoints *sizeof(uint)* maxNeighborNum));
	checkCudaErrors(cudaMalloc((void**)&fbuf.kernelRotate, fcuda.numElasticPoints * sizeof(float3) * maxNeighborNum));
	checkCudaErrors(cudaMalloc((void**)&fbuf.neighborDistance, fcuda.numElasticPoints *sizeof(float3)* maxNeighborNum));
	checkCudaErrors(cudaMalloc((void**)&fbuf.kernelGrad, fcuda.numElasticPoints * sizeof(float3) *  maxNeighborNum));
	checkCudaErrors(cudaMalloc((void**)&fbuf.neighborIndex, fcuda.numElasticPoints * sizeof(uint) *  maxNeighborNum));
	CudaMem += fcuda.numElasticPoints *maxNeighborNum*(2 * sizeof(uint) + 3 * sizeof(float3));

	cudaThreadSynchronize();

	return CudaMem;
}
void PorousParamCUDA(float bulkModulus_porous, float bulkModulus_grains, float bulkModulus_solid, float	bulkModulus_fluid, float poroDeformStrength, float capillary, float relax2)
{
	fcuda.bulkModulus_porous = bulkModulus_porous;
	fcuda.bulkModulus_grains = bulkModulus_grains;
	fcuda.bulkModulus_solid = bulkModulus_solid;
	fcuda.bulkModulus_fluid = bulkModulus_fluid;
	fcuda.poroDeformStrength = poroDeformStrength;
	fcuda.relax2 = relax2;
	float alpha = 1 - bulkModulus_porous / bulkModulus_grains;
	fcuda.CoCompressibility = bulkModulus_solid*bulkModulus_fluid / ((alpha - fcuda.rest_porosity)*bulkModulus_fluid + fcuda.rest_porosity*bulkModulus_solid);
	fcuda.capillary = capillary;
	printf("CoCompressibility is %f\n", fcuda.CoCompressibility);
}
void FluidParamCUDA ( float ss, float sr, float pr, float mass, float rest, float3 bmin, float3 bmax, float estiff, float istiff,float pbstiff, float visc, float damp, float fmin, float fmax, float ffreq, float gslope, float gx, float gy, float gz, float al, float vl )
{
	fcuda.psimscale = ss;
	fcuda.psmoothradius = sr;
	fcuda.pradius = pr;
	fcuda.r2 = sr * sr;
	fcuda.pmass = mass;
	fcuda.prest_dens = rest;	
	fcuda.pboundmin = bmin;
	fcuda.pboundmax = bmax;
	fcuda.pextstiff = estiff;
	fcuda.pintstiff = istiff;
	fcuda.pbstiff = pbstiff;
	fcuda.pvisc = visc;
	fcuda.pdamp = damp;
	fcuda.pforce_min = fmin;
	fcuda.pforce_max = fmax;
	fcuda.pforce_freq = ffreq;
	fcuda.pground_slope = gslope;
	fcuda.pgravity = make_float3( gx, gy, gz );
	fcuda.AL = al;
	fcuda.AL2 = al * al;
	fcuda.VL = vl;
	fcuda.VL2 = vl * vl;

	
	printf ( "Bound Min: %f %f %f\n", bmin.x, bmin.y, bmin.z );
	printf ( "Bound Max: %f %f %f\n", bmax.x, bmax.y, bmax.z );

	fcuda.pdist = pow ( fcuda.pmass / fcuda.prest_dens, 1/3.0f );
	fcuda.poly6kern = 315.0f / (64.0f * 3.141592 * pow( sr, 9.0f) );
	fcuda.spikykern = -45.0f / (3.141592 * pow( sr, 6.0f) );
	fcuda.lapkern = 45.0f / (3.141592 * pow( sr, 6.0f) );	

	//fcuda.CubicSplineKern1 = 1 / (4 * 3.141592*pow(sr, 3));
	//fcuda.CubicSplineKern2 = 1 / (3.141592*pow(sr, 3));

	fcuda.CubicSplineKern = 8 / (3.141592*pow(sr, 3));
	fcuda.gradCubicSplineKern = 48 / (3.141592*pow(sr, 4));

	fcuda.CubicSplineKern1 = 1 / (4 * 3.141592*pow(sr, 3));
	fcuda.CubicSplineKern2 = 8 / (3.141592*pow(sr, 3));

	fcuda.gradCubicSplineKern1 = -3 / (4 * 3.141592*pow(sr, 4));
	fcuda.gradCubicSplineKern2 = 1 / (3.141592*pow(sr, 4));
	//printf("fcuda.gradCubicSplineKern1 is %f,fcuda.gradCubicSplineKern2 is %f,fcuda.spikykern is %f\n",
	//	fcuda.gradCubicSplineKern1, fcuda.gradCubicSplineKern2, fcuda.spikykern);
	checkCudaErrors( cudaMemcpyToSymbol ( simData, &fcuda, sizeof(FluidParams) ) );
	cudaThreadSynchronize ();
}
void ParamUpdateCUDA(bool hidebound, bool hidefluid, bool hidesolid, bool hiderigid, float* colorValue)
{
	fcuda.HideBound = hidebound;
	fcuda.HideFluid = hidefluid;
	fcuda.HideSolid = hidesolid;
	fcuda.HideRigid = hiderigid;
	for(int i=0;i<MAX_FLUIDNUM;++i)
		fcuda.colorValue[i] = colorValue[i];
	checkCudaErrors(cudaMemcpyToSymbol(simData, &fcuda, sizeof(FluidParams)));
	cudaThreadSynchronize();
}
void FluidParamCUDA_projectu(float visc_factor, float fluid_pfactor,float solid_pfactor,float bdamp)
{
	fcuda.visc_factor = visc_factor;
	fcuda.fluid_pfactor = fluid_pfactor;
	fcuda.solid_pfactor = solid_pfactor;
	fcuda.bdamp = bdamp;
	fcuda.gravityfree = 0;
}

void FluidMfParamCUDA ( float *dens, float *visc, float *mass, float diffusion, float catnum, float dt,  float3 cont, float3 mb1,float3 mb2, float relax,int example)
{
	fcuda.mf_catnum = catnum;
	fcuda.mf_diffusion = diffusion;
	fcuda.mf_dt = dt;
	for(int i=0;i<MAX_FLUIDNUM;i++)
	{
		fcuda.mf_dens[i] = dens[i];
		fcuda.mf_visc[i] = visc[i];
		fcuda.mf_mass[i] = mass[i];
	}
	fcuda.mf_multiFlagPNum = 0;
	//fcuda.mf_splitVolume = splitV;
	//fcuda.mf_mergeVolume = mergeV;
	fcuda.mf_maxPnum = fcuda.pnum * EMIT_BUF_RATIO;
	fcuda.cont =  cont.x;	fcuda.cont1 = cont.y;	fcuda.cont2 = cont.z;	
	fcuda.mb1.x = mb1.x;	fcuda.mb1.y = mb1.y;	fcuda.mb1.z = mb1.z;
	fcuda.mb2.x = mb2.x;	fcuda.mb2.y = mb2.y;	fcuda.mb2.z = mb2.z;
	fcuda.bxmin = mb1.x;    fcuda.by = mb1.y;       fcuda.bzmin = mb1.z;
	fcuda.bxmax = mb2.x;							fcuda.bzmax = mb2.z; 
	
	fcuda.relax = relax;
	fcuda.example = example;
	checkCudaErrors( cudaMemcpyToSymbol ( simData, &fcuda, sizeof(FluidParams) ) );
	cudaThreadSynchronize ();
}



void preallocBlockSumsInt (unsigned int maxNumElements)
{
	assert(g_numEltsAllocated == 0); // shouldn't be called 

	g_numEltsAllocated = maxNumElements;
	unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
	unsigned int numElts = maxNumElements;
	int level = 0;

	do {       
		unsigned int numBlocks =   max(1, (int)ceil((float)numElts / (2.f * blockSize)));
		if (numBlocks > 1) level++;
		numElts = numBlocks;
	} while (numElts > 1);

	g_scanBlockSumsInt = (int**) malloc(level * sizeof(int*));
	g_numLevelsAllocated = level;
	
	numElts = maxNumElements;
	level = 0;
	
	do {       
		unsigned int numBlocks = max(1, (int)ceil((float)numElts / (2.f * blockSize)));
		if (numBlocks > 1) checkCudaErrors ( cudaMalloc((void**) &g_scanBlockSumsInt[level++], numBlocks * sizeof(int)) );
		numElts = numBlocks;
	} while (numElts > 1);
}
void deallocBlockSumsInt()
{
	for (unsigned int i = 0; i < g_numLevelsAllocated; i++) cudaFree(g_scanBlockSumsInt[i]);    
	free( (void**)g_scanBlockSumsInt );

	g_scanBlockSumsInt = 0;
	g_numEltsAllocated = 0;
	g_numLevelsAllocated = 0;
}

//Copy buffers
void CopyToCUDA ( float* pos, float* vel, float* veleval, float* force, float* pressure, float* density, uint* cluster, uint* gnext, char* clr)
{

	// Send particle buffers
	int numPoints = fcuda.pnum;
	checkCudaErrors( cudaMemcpy ( fbuf.mpos,		pos,			numPoints*sizeof(float)*3, cudaMemcpyHostToDevice ) );	
	checkCudaErrors( cudaMemcpy ( fbuf.maccel,		vel,			numPoints*sizeof(float)*3, cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( fbuf.mveleval, veleval,		numPoints*sizeof(float)*3, cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( fbuf.mforce,	force,			numPoints*sizeof(float)*3, cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( fbuf.mpress,	pressure,		numPoints*sizeof(float),  cudaMemcpyHostToDevice ) );
	//checkCudaErrors( cudaMemcpy ( fbuf.mpress_pre, pressure, numPoints * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpy ( fbuf.mdensity, density,		numPoints*sizeof(float),  cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( fbuf.mclr,		clr,			numPoints*sizeof(uint), cudaMemcpyHostToDevice ) );


	cudaThreadSynchronize ();	
}
void CopyMfToCUDA ( float* alpha, float* alpha_pre, float* pressure_modify, float* vel_phrel, float* restmass, float* restdensity, float* visc, float* velxcor, float* alphagrad)
{
	// Send particle buffers
	int numPoints = fcuda.pnum;
	checkCudaErrors( cudaMemcpy ( fbuf.mf_alpha,				alpha,				numPoints*MAX_FLUIDNUM*sizeof(float), cudaMemcpyHostToDevice ) );	
	checkCudaErrors( cudaMemcpy ( fbuf.mf_alpha_next,			alpha,			numPoints*MAX_FLUIDNUM*sizeof(float), cudaMemcpyHostToDevice ) );

	checkCudaErrors( cudaMemcpy ( fbuf.mf_vel_phrel,			vel_phrel,			numPoints*MAX_FLUIDNUM*sizeof(float)*3, cudaMemcpyHostToDevice ) );
	//checkCudaErrors( cudaMemcpy ( fbuf.mf_alphagrad,			alphagrad,			numPoints*MAX_FLUIDNUM*sizeof(float)*3, cudaMemcpyHostToDevice ) );

	//checkCudaErrors( cudaMemcpy ( fbuf.mf_pressure_modify,	pressure_modify,	numPoints*sizeof(float), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( fbuf.mf_restmass,			restmass,			numPoints*sizeof(float),  cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( fbuf.mf_restdensity,		restdensity,		numPoints*sizeof(float),  cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( fbuf.mf_visc,				visc,				numPoints*sizeof(float), cudaMemcpyHostToDevice ) );
	//checkCudaErrors( cudaMemcpy ( fbuf.mf_velxcor,			velxcor,			numPoints*sizeof(float)*3, cudaMemcpyHostToDevice ) );
	//checkCudaErrors( cudaMemcpy ( fbuf.MFtype,				mftype,				numPoints*sizeof(int), cudaMemcpyHostToDevice ) );
	cudaThreadSynchronize ();	
}
void CopyBoundToCUDA (int* isbound )
{
	int numPoints = fcuda.pnum;
	checkCudaErrors( cudaMemcpy ( fbuf.misbound,	isbound,		numPoints*sizeof(int), cudaMemcpyHostToDevice ) );
	cudaThreadSynchronize ();	
}
void CopyToCUDA_Uproject(int* mftype)
{
	int numPoints = fcuda.pnum;
	checkCudaErrors( cudaMemcpy( fbuf.MFtype, mftype, numPoints*sizeof(int), cudaMemcpyHostToDevice));

	cudaThreadSynchronize ();
}
void CopyToCUDA_elastic(uint* elasticID,float* porosity,float*signDistance)
{
	int numPoints = fcuda.pnum;
	int numElasticPoints = fcuda.numElasticPoints;
	checkCudaErrors(cudaMemcpy(fbuf.elasticID, elasticID, numPoints * sizeof(uint), cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(fbuf.porosity, porosity, numElasticPoints * sizeof(float), cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(fbuf.colorField, signDistance, numElasticPoints * sizeof(float), cudaMemcpyHostToDevice));
	cudaThreadSynchronize();
}
void CopyFromCUDA ( float* pos, float* vel, float* veleval, float* force, float* pressure, float* density, uint* cluster, uint* gnext, char* clr, int mode)
{
	// Return particle buffers
	int numPoints = fcuda.pnum;
	//printf("sizeof(float3) is %d and sizeof(float) is %d\n", sizeof(float3), sizeof(float));
	//printf("fbuf.mpos address : OX%p\n", fbuf.mpos);
	//printf("numPoints is %d\n", numPoints);
	if ( pos != 0x0 ) checkCudaErrors( cudaMemcpy ( pos,		fbuf.mpos,			numPoints*sizeof(float)*3, cudaMemcpyDeviceToHost ) );
	if ( clr != 0x0 ) checkCudaErrors( cudaMemcpy ( clr,		fbuf.mclr,			numPoints*sizeof(uint),  cudaMemcpyDeviceToHost ) );

	if( mode == 2){
		checkCudaErrors( cudaMemcpy ( vel,		fbuf.maccel,			numPoints*sizeof(float)*3, cudaMemcpyDeviceToHost ) );
		checkCudaErrors( cudaMemcpy ( veleval,	fbuf.mveleval,		numPoints*sizeof(float)*3, cudaMemcpyDeviceToHost ) );
		checkCudaErrors( cudaMemcpy ( force,	fbuf.mforce,		numPoints*sizeof(float)*3, cudaMemcpyDeviceToHost ) );
		checkCudaErrors( cudaMemcpy ( pressure,	fbuf.mpress,		numPoints*sizeof(float),  cudaMemcpyDeviceToHost ) );
		checkCudaErrors( cudaMemcpy ( density,	fbuf.mdensity,		numPoints*sizeof(float),  cudaMemcpyDeviceToHost ) );
	}
	
	cudaThreadSynchronize ();	
}
void CopyMfFromCUDA ( float* alpha, float* alpha_pre, float* pressure_modify, float* vel_phrel, float* restmass, float* restdensity, float* visc, float* velxcor, float* alphagrad, int mode)
{
	int numPoints = fcuda.pnum;
	checkCudaErrors( cudaMemcpy ( alpha,				fbuf.mf_alpha,				numPoints*MAX_FLUIDNUM*sizeof(float), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( restmass,			fbuf.mf_restmass,			numPoints*sizeof(float),  cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( restdensity,		fbuf.mf_restdensity,		numPoints*sizeof(float),  cudaMemcpyDeviceToHost ) );
	
	if( mode == 2){
		// Send particle buffers
		checkCudaErrors( cudaMemcpy ( alpha_pre,			fbuf.mf_alpha_next,			numPoints*MAX_FLUIDNUM*sizeof(float), cudaMemcpyDeviceToHost ) );
		//checkCudaErrors( cudaMemcpy ( pressure_modify,	fbuf.mf_pressure_modify,	numPoints*sizeof(float), cudaMemcpyDeviceToHost ) );
		checkCudaErrors( cudaMemcpy ( vel_phrel,			fbuf.mf_vel_phrel,			numPoints*MAX_FLUIDNUM*sizeof(float)*3, cudaMemcpyDeviceToHost ) );
		
		checkCudaErrors( cudaMemcpy ( visc,				fbuf.mf_visc,				numPoints*sizeof(float), cudaMemcpyDeviceToHost ) );
		//checkCudaErrors( cudaMemcpy ( velxcor,			fbuf.mf_velxcor,			numPoints*sizeof(float)*3, cudaMemcpyDeviceToHost ) );
		//checkCudaErrors( cudaMemcpy ( alphagrad,			fbuf.mf_alphagrad,			numPoints*MAX_FLUIDNUM*sizeof(float)*3, cudaMemcpyDeviceToHost ) );
	}
}
void CopyBoundFromCUDA (int* isbound )
{
	int numPoints = fcuda.pnum;
	if ( isbound != 0x0 ) checkCudaErrors( cudaMemcpy ( isbound,	fbuf.misbound,		numPoints*sizeof(int),  cudaMemcpyDeviceToHost ) );
	cudaThreadSynchronize ();	
}
void CopyFromCUDA_Uproject(int* mftype, float*beta)
{
	int numPoints = fcuda.pnum;
	checkCudaErrors( cudaMemcpy( mftype, fbuf.MFtype, numPoints*sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(beta, fbuf.mf_beta, numPoints * sizeof(float)*MAX_FLUIDNUM*MAX_SOLIDNUM, cudaMemcpyDeviceToHost));

	cudaThreadSynchronize ();
}


//Called when particles emitted
void CopyEmitToCUDA ( float* pos, float* vel, float* veleval, float* force, float* pressure, float* density, uint* cluster, uint* gnext, char* clr, int startnum, int numcount,int* isbound )
{

	// Send particle buffers
	checkCudaErrors( cudaMemcpy ( fbuf.mpos+startnum,		pos+startnum*3,			numcount*sizeof(float)*3, cudaMemcpyHostToDevice ) );	
	checkCudaErrors( cudaMemcpy ( fbuf.maccel+startnum,		vel+startnum*3,			numcount*sizeof(float)*3, cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( fbuf.mveleval+startnum,	veleval+startnum*3,		numcount*sizeof(float)*3, cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( fbuf.mforce+startnum,	force+startnum*3,			numcount*sizeof(float)*3, cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( fbuf.mpress+startnum,		pressure+startnum,		numcount*sizeof(float),  cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( fbuf.mdensity+startnum,	density+startnum,		numcount*sizeof(float),  cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( fbuf.mclr+startnum,		clr+startnum,			numcount*sizeof(uint), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( fbuf.misbound + startnum,				isbound + startnum,				numcount*sizeof(int), cudaMemcpyHostToDevice ) );

	cudaThreadSynchronize ();	
}	
void CopyEmitMfToCUDA ( float* alpha, float* alpha_pre, float* pressure_modify, float* vel_phrel, float* restmass, float* restdensity, float* visc, float* velxcor, float* alphagrad,int startnum, int numcount)
{
	// Send particle buffers
	int mulstartnum = startnum*MAX_FLUIDNUM;
	checkCudaErrors( cudaMemcpy ( fbuf.mf_alpha + mulstartnum,				alpha + mulstartnum,				numcount*MAX_FLUIDNUM*sizeof(float), cudaMemcpyHostToDevice ) );	
	checkCudaErrors( cudaMemcpy ( fbuf.mf_alpha_next + mulstartnum,			alpha_pre + mulstartnum,			numcount*MAX_FLUIDNUM*sizeof(float), cudaMemcpyHostToDevice ) );
	//checkCudaErrors( cudaMemcpy ( fbuf.mf_pressure_modify+startnum,			pressure_modify+startnum,			numcount*sizeof(float), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( fbuf.mf_vel_phrel + mulstartnum,			vel_phrel + mulstartnum*3,			numcount*MAX_FLUIDNUM*sizeof(float)*3, cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( fbuf.mf_restmass+startnum,					restmass+startnum,					numcount*sizeof(float),  cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( fbuf.mf_restdensity+startnum,				restdensity+startnum,				numcount*sizeof(float),  cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( fbuf.mf_visc+startnum,						visc+startnum,						numcount*sizeof(float), cudaMemcpyHostToDevice ) );
	//checkCudaErrors( cudaMemcpy ( fbuf.mf_velxcor+startnum,					velxcor+startnum*3,					numcount*sizeof(float)*3, cudaMemcpyHostToDevice ) );
	//checkCudaErrors( cudaMemcpy ( fbuf.mf_alphagrad + mulstartnum,			alphagrad + mulstartnum*3,			numcount*MAX_FLUIDNUM*sizeof(float)*3, cudaMemcpyHostToDevice ) );

	cudaThreadSynchronize ();	
}
void UpdatePNumCUDA( int newPnum)
{
	fcuda.pnum = newPnum;
	computeNumBlocks ( fcuda.pnum, 384, fcuda.numBlocks, fcuda.numThreads);    //threads changed!
	fcuda.szPnts = (fcuda.numBlocks  * fcuda.numThreads);					   //szPnts changed!	
	checkCudaErrors( cudaMemcpyToSymbol ( simData, &fcuda, sizeof(FluidParams) ) );
	cudaThreadSynchronize ();
}
int MfGetPnum(){
	return fcuda.pnum;
}


//Called in RunSimulateCudaFull
void InitialSortCUDA( uint* gcell, uint* ccell, int* gcnt )
{
	cudaMemset ( fbuf.mgridcnt, 0,			fcuda.gridTotal * sizeof(int));
	cudaMemset ( fbuf.mgridoff, 0,			fcuda.gridTotal * sizeof(int));
	cudaMemset ( fbuf.mgcell, 0,			fcuda.pnum * sizeof(uint));
	InitialSort<<< fcuda.numBlocks, fcuda.numThreads>>> ( fbuf, fcuda.pnum );
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf ( stderr,  "CUDA ERROR: InsertParticlesCUDA: %s\n", cudaGetErrorString(error) );
	}  
	cudaThreadSynchronize ();

	// Transfer data back if requested (for validation)
	if (gcell != 0x0) {
		checkCudaErrors( cudaMemcpy ( gcell,	fbuf.mgcell,	fcuda.pnum*sizeof(uint),		cudaMemcpyDeviceToHost ) );		
		checkCudaErrors( cudaMemcpy ( gcnt,	fbuf.mgridcnt,	fcuda.gridTotal*sizeof(int),	cudaMemcpyDeviceToHost ) );
		//checkCudaErrors( cudaMemcpy ( ccell,	fbuf.mcluster,	fcuda.pnum*sizeof(uint),		cudaMemcpyDeviceToHost ) );
	}
}
void SortGridCUDA( int* goff )
{
	thrust::device_ptr<uint> dev_keysg(fbuf.mgcell);
	thrust::device_ptr<uint> dev_valuesg(fbuf.midsort);
	thrust::sort_by_key(dev_keysg,dev_keysg+fcuda.pnum,dev_valuesg);
	cudaThreadSynchronize ();
	CalcFirstCnt <<< fcuda.numBlocks, fcuda.numThreads>>> ( fbuf, fcuda.pnum );
	//	cudaThreadSynchronize ();
	cudaThreadSynchronize ();
	GetCnt <<<fcuda.numBlocks,fcuda.numThreads>>> (fbuf,fcuda.pnum);
	cudaThreadSynchronize ();
	/*
	uint* test,*test1;
	test = (uint*)malloc(sizeof(uint)*fcuda.pnum);
	test1 = (uint*)malloc(sizeof(uint)*fcuda.gridTotal);
	cudaMemcpy(test,fbuf.mgcell,sizeof(uint)*fcuda.pnum,cudaMemcpyDeviceToHost);
	cudaMemcpy(test1,fbuf.mgridoff,sizeof(uint)*fcuda.gridTotal,cudaMemcpyDeviceToHost);
	for (int i = 0;i<fcuda.pnum;i++)
		if (test[i]!=GRID_UNDEF)
		printf("%u %u %u\n",test[i],test1[test[i]]);
	*/
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR:SortGridCUDA: %s\n", cudaGetErrorString(error));
	}
}
void CountingSortFullCUDA_( uint* ggrid )
{
	// Transfer particle data to temp buffers
	int n = fcuda.pnum;
	cudaMemcpy ( fbuf.msortbuf + n*BUF_POS,		fbuf.mpos,		n*sizeof(float)*3,	cudaMemcpyDeviceToDevice );
	cudaMemcpy ( fbuf.msortbuf + n*BUF_VELEVAL,	fbuf.mveleval,	n*sizeof(float)*3,	cudaMemcpyDeviceToDevice );
	cudaMemcpy ( fbuf.msortbuf + n*BUF_PRESS,	fbuf.mpress,	n*sizeof(float),	cudaMemcpyDeviceToDevice );
	cudaMemcpy(fbuf.msortbuf + n*BUF_GCELL, fbuf.mgcell, n * sizeof(uint), cudaMemcpyDeviceToDevice);
	cudaMemcpy(fbuf.msortbuf + n*BUF_GNDX, fbuf.mgndx, n * sizeof(uint), cudaMemcpyDeviceToDevice);
#ifdef NEW_BOUND
	cudaMemcpy(fbuf.msortbuf + n*BUF_ISBOUND, fbuf.misbound, n * sizeof(int), cudaMemcpyDeviceToDevice);
#endif
	//multi fluid
	cudaMemcpy(fbuf.msortbuf + n*BUF_ALPHA, fbuf.mf_alpha, n*MAX_FLUIDNUM * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(fbuf.msortbuf + n*BUF_ALPHAPRE, fbuf.mf_alpha_next, n*MAX_FLUIDNUM * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(fbuf.msortbuf + n*BUF_RMASS, fbuf.mf_restmass, n * sizeof(float), cudaMemcpyDeviceToDevice);
	//porous
	cudaMemcpy ( fbuf.msortbuf + n*BUF_INDICATOR,		fbuf.MFtype,			n*sizeof(int),					cudaMemcpyDeviceToDevice );
	
	//an implicit SPH formulation for elastic body
	
	cudaMemcpy ( fbuf.msortbuf + n*BUF_ELASTICID,		fbuf.elasticID,			n * sizeof(uint), cudaMemcpyDeviceToDevice);
	cudaMemcpy ( fbuf.msortbuf + n*BUF_ABSORBEDPERCENT, fbuf.mf_beta,	n * MAX_FLUIDNUM * sizeof(float) * MAX_SOLIDNUM, cudaMemcpyDeviceToDevice);
	cudaMemcpy(fbuf.msortbuf + n*BUF_BETANEXT, fbuf.mf_beta_next, n * MAX_FLUIDNUM * sizeof(float) * MAX_SOLIDNUM, cudaMemcpyDeviceToDevice);
	//cudaMemcpy(fbuf.msortbuf + n*BUF_POROVEL, fbuf.poroVel, n *MAX_FLUIDNUM * sizeof(float3), cudaMemcpyDeviceToDevice);

	// Counting Sort - pass one, determine grid counts
	cudaMemset ( fbuf.mgrid,	GRID_UCHAR,	fcuda.pnum * sizeof(int) );
	
	CountingSortFull_ <<< fcuda.numBlocks, fcuda.numThreads>>> ( fbuf, fcuda.pnum);
	cudaThreadSynchronize ();

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf ( stderr, "CUDA ERROR:Sorting Failed: %s\n", cudaGetErrorString(error) );
	} 
	////checkCudaErrors(cudaMemcpyFromSymbol(&(fcuda.pnum), pNumFT, sizeof(int)));  //total pnum changed!
	////computeNumBlocks ( fcuda.pnum, 384, fcuda.numBlocks, fcuda.numThreads);    //threads changed!
	////fcuda.szPnts = (fcuda.numBlocks  * fcuda.numThreads);					   //szPnts changed!
	////		printf("pnum:%d,Blocknum:%d,Threadnum:%d\n",fcuda.pnum,fcuda.numBlocks,fcuda.numThreads);
	////cudaThreadSynchronize ();
}

void initSPH(float* restdensity,int* mftype)
{
	initDensity<<<fcuda.numBlocks, fcuda.numThreads>>>(fbuf, fcuda.pnum);
	cudaThreadSynchronize();
	
}

void TestFunc()
{
	testFunc << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: MfFindNearestVelCUDA: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();
}
void MfComputePressureCUDA ()
{
	//mfFindNearest<<< fcuda.numBlocks, fcuda.numThreads>>> (fbuf, fcuda.pnum);
	//cudaError_t error = cudaGetLastError();
	//if (error != cudaSuccess) {
	//	fprintf ( stderr, "CUDA ERROR: MfFindNearestVelCUDA: %s\n", cudaGetErrorString(error) );
	//}    
	//cudaThreadSynchronize ();

	mfPreComputeDensity<<< fcuda.numBlocks, fcuda.numThreads>>> ( fbuf, fcuda.pnum );
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf ( stderr, "CUDA ERROR: MfPreComputeDensityVelCUDA: %s\n", cudaGetErrorString(error) );
	}    
	cudaThreadSynchronize ();

	mfComputePressure<<< fcuda.numBlocks, fcuda.numThreads>>> ( fbuf, fcuda.pnum );
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf ( stderr, "CUDA ERROR: MfComputePressureVelCUDA: %s\n", cudaGetErrorString(error) );
	}    
	cudaThreadSynchronize ();
}

void MfPredictAdvection(float time)
{
	applyAlphaAndBeta << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: MfComputeDriftVelCUDA: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();

	//step1:compute density
	mfPreComputeDensity << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: MfPreComputeDensityVelCUDA: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();

	FindNearbySolid << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: compute fluid percent change CUDA: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();

	ComputeOtherForce << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum, time);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: MfComputeOtherForceCUDA: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();

	//step2:compute intermediate velocity
	computeMidVel << <fcuda.numBlocks, fcuda.numThreads >> >(fbuf, fcuda.pnum);
	//updateVelocity << <fcuda.numBlocks, fcuda.numThreads >> >(time, fbuf, fcuda.pnum);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: Compute mid vel: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();

	ComputeBRestVolume << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: Compute rest volume: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();

	ComputeVolume << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: Compute volume: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();

	ComputeSource << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: Compute source: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();
	
	ComputeAII << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ComputeAII: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();

	ComputeSolidPorePressure << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: compute pore pressure CUDA: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();
}

void PressureSolve(int fluid_beginIndex,int fluid_endIndex)
{
	int l = 0;
	float averror;
	float sum, length = fluid_endIndex - fluid_beginIndex;
	float eta = 0.1;
	cudaError_t error;
	float last_error = 1;
	do {
		//iterate compute pressure
		l++;
		//upgrade force to compute the error
		ComputePressureForce << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
		error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf(stderr, "CUDA ERROR: ComputePressureForce: %s\n", cudaGetErrorString(error));
		}
		cudaThreadSynchronize();

		ComputeCriterion << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
		error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf(stderr, "CUDA ERROR: Compute Criterion: %s\n", cudaGetErrorString(error));
		}
		cudaThreadSynchronize();
		thrust::device_ptr<float> dev_deltadens(fbuf.delta_density);
		thrust::device_vector<float> deltadens(dev_deltadens + fluid_beginIndex, dev_deltadens + fluid_endIndex);
		
		//averror = thrust::reduce(deltadens.begin(), deltadens.end()) / thrust::reduce(dens.begin(), dens.end());
		averror = thrust::reduce(deltadens.begin(), deltadens.end()) / (fluid_endIndex - fluid_beginIndex);
		//printf("the %dth iteration over.\n", l);
		//if (l > 10)
		//	break;
		if (abs(averror-last_error)/last_error < 0.001||l>100)
			break;
		last_error = averror;
	} while (l<5 || abs(averror)>eta);

	//printf("iteration time is %d, ave error is %f\n", l, averror);
	/*PressCorrection << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: PressCorrection: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();*/

	ApplyPressureForce << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: ComputePressureForce: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();

}
void MfComputeDriftVelCUDA ()
{
	cudaError_t error;
	ComputeSolidPorePressure << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: compute pore pressure CUDA: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();

	mfComputeDriftVel<<< fcuda.numBlocks, fcuda.numThreads>>> ( fbuf, fcuda.pnum );
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf ( stderr, "CUDA ERROR: MfComputeDriftVelCUDA: %s\n", cudaGetErrorString(error) );
	}    
	cudaThreadSynchronize ();
}
void MfComputeAlphaAdvanceCUDA ()
{
	cudaError_t error;
	//mfComputeDriftVel << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	//error = cudaGetLastError();
	//if (error != cudaSuccess) {
	//	fprintf(stderr, "CUDA ERROR: MfComputeDriftVelCUDA: %s\n", cudaGetErrorString(error));
	//}
	//cudaThreadSynchronize();

	//mfComputeTDM << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	//error = cudaGetLastError();
	//if (error != cudaSuccess) {
	//	fprintf(stderr, "CUDA ERROR: MfComputeTDM CUDA: %s\n", cudaGetErrorString(error));
	//}
	//cudaThreadSynchronize();

	//mfComputeAlphaAdvance << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	//error = cudaGetLastError();
	//if (error != cudaSuccess) {
	//	fprintf(stderr, "CUDA ERROR: MfComputeAlphaAdvanceCUDA: %s\n", cudaGetErrorString(error));
	//}
	//cudaThreadSynchronize();
	//ComputeFluidAdvance << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	//error = cudaGetLastError();
	//if (error != cudaSuccess) {
	//	fprintf(stderr, "CUDA ERROR: compute fluid advance CUDA: %s\n", cudaGetErrorString(error));
	//}
	//cudaThreadSynchronize();

	mfComputeCorrection << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: MfComputeCorrectionCUDA: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();
}
void MfComputeCorrectionCUDA ()
{
	/*if(fcuda.example == 5)
		mfComputeCorrection5<<< fcuda.numBlocks, fcuda.numThreads>>> ( fbuf, fcuda.pnum );
	else*/
	mfComputeCorrection<<< fcuda.numBlocks, fcuda.numThreads>>> ( fbuf, fcuda.pnum );	
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf ( stderr, "CUDA ERROR: MfComputeCorrectionCUDA: %s\n", cudaGetErrorString(error) );
	}    
	cudaThreadSynchronize ();
}

//void ComputeForceCUDA_ProjectU(float time)
//{
//	////计算公式(8)除去T_Sm之外的项
//	//ComputeForce_projectu<<< fcuda.numBlocks, fcuda.numThreads>>> ( fbuf, fcuda.pnum );
//	//cudaError_t error = cudaGetLastError();
//	//if (error != cudaSuccess)
//	//	fprintf ( stderr, "CUDA ERROR: MfComputeForceCUDA: %s\n", cudaGetErrorString(error) );
//	//cudaThreadSynchronize ();
//
//	//cudaThreadSynchronize();
//
//	//AddSPHtensorForce<<<fcuda.numBlocks, fcuda.numThreads>>>(fbuf, fcuda.pnum, time);
//	//error = cudaGetLastError();
//	//if (error != cudaSuccess)
//	//	fprintf ( stderr, "CUDA ERROR: Adding SPH tensor Force: %s\n", cudaGetErrorString(error) );
//	//cudaThreadSynchronize ();
//
//
//}


//Mathematics
__device__	inline double RxPythag(const double a, const double b)
{
	double absa = abs(a), absb = abs(b);
	return (absa > absb ? absa*(double)sqrt((double)(1.0+(absb/absa)*(absb/absa))) :
		(absb == 0.0 ? 0.0 : absb*(double)sqrt((double)(1.0+(absa/absb)*(absa/absb)))));
}
__device__	inline double RXD_MIN(const double &a, const double &b){ return ((a < b) ? a : b); }
__device__	inline double RXD_MAX(const double &a, const double &b){ return ((a > b) ? a : b); }
__device__	inline double RXD_SIGN2(const double &a, const double &b){ return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a); }

__device__	int svdecomp3(float w[3], float u[9], float v[9], float eps)
{
	bool flag;
	int i, its, j, jj, k, l, nm;
	float anorm, c, f, g, h, s, scale, x, y, z;
	float rv1[3];
	g = scale = anorm = 0.0;
	for(i = 0; i < 3; ++i){
		l = i+2;
		rv1[i] = scale*g;
		g = s = scale = 0.0;
		for(k = i; k < 3; ++k) scale += abs(u[k*3+i]);
		if(scale != 0.0){
			for(k = i; k < 3; ++k){
				u[k*3+i] /= scale;
				s += u[k*3+i]*u[k*3+i];
			}
			f = u[i*3+i];
			g = -RXD_SIGN2(sqrt(s), f);
			h = f*g-s;
			u[i*3+i] = f-g;
			for(j = l-1; j < 3; ++j){
				for(s = 0.0, k = i; k < 3; ++k) s += u[k*3+i]*u[k*3+j];
				f = s/h;
				for(k = i; k < 3; ++k) u[k*3+j] += f*u[k*3+i];
			}
			for(k = i; k < 3; ++k) u[k*3+i] *= scale;
		}

		w[i] = scale*g;
		g = s = scale = 0.0;
		if(i+1 <= 3 && i+1 != 3){
			for(k = l-1; k < 3; ++k) scale += abs(u[i*3+k]);
			if(scale != 0.0){
				for(k = l-1; k < 3; ++k){
					u[i*3+k] /= scale;
					s += u[i*3+k]*u[i*3+k];
				}
				f = u[i*3+l-1];
				g = -RXD_SIGN2(sqrt(s), f);
				h = f*g-s;
				u[i*3+l-1] = f-g;
				for(k = l-1; k < 3; ++k) rv1[k] = u[i*3+k]/h;
				for(j = l-1; j < 3; ++j){
					for(s = 0.0,k = l-1; k < 3; ++k) s += u[j*3+k]*u[i*3+k];
					for(k = l-1; k < 3; ++k) u[j*3+k] += s*rv1[k];
				}
				for(k = l-1; k < 3; ++k) u[i*3+k] *= scale;
			}
		}
		anorm = RXD_MAX(anorm, (abs(w[i])+abs(rv1[i])));
	}
	for(i = 2; i >= 0; --i){
		if(i < 2){
			if(g != 0.0){
				for(j = l; j < 3; ++j){
					v[j*3+i] = (u[i*3+j]/u[i*3+l])/g;
				}
				for(j = l; j < 3; ++j){
					for(s = 0.0, k = l; k < 3; ++k) s += u[i*3+k]*v[k*3+j];
					for(k = l; k < 3; ++k) v[k*3+j] += s*v[k*3+i];
				}
			}
			for(j = l; j < 3; ++j) v[i*3+j] = v[j*3+i] = 0.0;
		}
		v[i*3+i] = 1.0;
		g = rv1[i];
		l = i;
	}
	for(i = 2; i >= 0; --i){
		l = i+1;
		g = w[i];
		for(j = l; j < 3; ++j) u[i*3+j] = 0.0;
		if(g != 0.0){
			g = 1.0/g;
			for(j = l; j < 3; ++j){
				for(s = 0.0, k = l; k < 3; ++k) s += u[k*3+i]*u[k*3+j];
				f = (s/u[i*3+i])*g;
				for(k = i; k < 3; ++k) u[k*3+j] += f*u[k*3+i];
			}
			for(j = i; j < 3; ++j) u[j*3+i] *= g;
		}
		else{
			for(j = i; j < 3; ++j) u[j*3+i] = 0.0;
		}
		++u[i*3+i];
	}
	for(k = 2; k >= 0; --k){
		for(its = 0; its < 30; ++its){
			flag = true;
			for(l = k; l >= 0; --l){
				nm = l-1;
				if(l == 0 || abs(rv1[l]) <= eps*anorm){
					flag = false;
					break;
				}
				if(abs(w[nm]) <= eps*anorm) break;
			}
			if(flag){
				c = 0.0;
				s = 1.0;
				for(i = l; i < k+1; ++i){
					f = s*rv1[i];
					rv1[i] = c*rv1[i];
					if(abs(f) <= eps*anorm) break;
					g = w[i];
					h = RxPythag(f, g);
					w[i] = h;
					h = 1.0/h;
					c = g*h;
					s = -f*h;
					for(j = 0; j < 3; ++j){
						y = u[j*3+nm];
						z = u[j*3+i];
						u[j*3+nm] = y*c+z*s;
						u[j*3+i] = z*c-y*s;
					}
				}
			}
			z = w[k];
			if(l == k){
				if(z < 0.0){
					w[k] = -z;
					for(j = 0; j < 3; ++j) v[j*3+k] = -v[j*3+k];
				}
				break;
			}
			if(its == 29){
				//printf("no convergence in 30 svdcmp iterations");
				return 0;
			}
			x = w[l];
			nm = k-1;
			y = w[nm];
			g = rv1[nm];
			h = rv1[k];
			f = ((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);
			g = RxPythag(f, 1.0f);
			f = ((x-z)*(x+z)+h*((y/(f+RXD_SIGN2(g, f)))-h))/x;
			c = s = 1.0;
			for(j = l; j <= nm; ++j){
				i = j+1;
				g = rv1[i];
				y = w[i];
				h = s*g;
				g = c*g;
				z = RxPythag(f, h);
				rv1[j] = z;
				c = f/z;
				s = h/z;
				f = x*c+g*s;
				g = g*c-x*s;
				h = y*s;
				y *= c;
				for(jj = 0; jj < 3; ++jj){
					x = v[jj*3+j];
					z = v[jj*3+i];
					v[jj*3+j] = x*c+z*s;
					v[jj*3+i] = z*c-x*s;
				}
				z = RxPythag(f, h);
				w[j] = z;
				if(z){
					z = 1.0/z;
					c = f*z;
					s = h*z;
				}
				f = c*g+s*y;
				x = c*y-s*g;
				for(jj = 0; jj < 3; ++jj){
					y = u[jj*3+j];
					z = u[jj*3+i];
					u[jj*3+j] = y*c+z*s;
					u[jj*3+i] = z*c-y*s;
				}
			}
			rv1[l] = 0.0;
			rv1[k] = f;
			w[k] = x;
		}
	}

	// reorder
	int inc = 1;
	float sw;
	float su[3], sv[3];

	do{
		inc *= 3;
		inc++; 
	}while(inc <= 3);

	do{
		inc /= 3;
		for(i = inc; i < 3; ++i){
			sw = w[i];
			for(k = 0; k < 3; ++k) su[k] = u[k*3+i];
			for(k = 0; k < 3; ++k) sv[k] = v[k*3+i];
			j = i;
			while (w[j-inc] < sw){
				w[j] = w[j-inc];
				for(k = 0; k < 3; ++k) u[k*3+j] = u[k*3+j-inc];
				for(k = 0; k < 3; ++k) v[k*3+j] = v[k*3+j-inc];
				j -= inc;
				if (j < inc) break;
			}
			w[j] = sw;
			for(k = 0; k < 3; ++k) u[k*3+j] = su[k];
			for(k = 0; k < 3; ++k) v[k*3+j] = sv[k];

		}
	}while(inc > 1);

	for(k = 0; k < 3; ++k){
		s = 0;
		for(i = 0; i < 3; ++i) if(u[i*3+k] < 0.) s++;
		for(j = 0; j < 3; ++j) if(v[j*3+k] < 0.) s++;
		if(s > 3){
			for(i = 0; i < 3; ++i) u[i*3+k] = -u[i*3+k];
			for(j = 0; j < 3; ++j) v[j*3+k] = -v[j*3+k];
		}
	}

	return 1;
}
__device__ void multiply_matrix3(float* a, float* b, float* c){
	float d[9];
	for(int i=0; i<3; i++)
		for(int j=0; j<3; j++)
			d[i*3+j] = a[i*3+0]*b[0*3+j]+a[i*3+1]*b[1*3+j]+a[i*3+2]*b[2*3+j];
	for(int k=0; k<9; k++)
		c[k] = d[k];
}
__device__ float3 multiply_mv3(float*m,float3 v)
{
	float3 a;
	a.x = m[0] * v.x + m[1] * v.y + m[2] * v.z;
	a.y = m[3] * v.x + m[4] * v.y + m[5] * v.z;
	a.z = m[6] * v.x + m[7] * v.y + m[8] * v.z;
	return a;
}
__device__ void transmit3(float* a,float* b){
	float c[9];
	c[0]=a[0]; c[1]=a[3]; c[2]=a[6];
	c[3]=a[1]; c[4]=a[4]; c[5]=a[7];
	c[6]=a[2]; c[7]=a[5]; c[8]=a[8];
	for(int k=0; k<9; k++)
		b[k]=c[k];
}
//__device__ float3 cross(const float3 v1,const float3 v2)
//{
//	float3 result;
//	result.x = v1.y*v2.z - v1.z*v2.y;
//	result.y = v1.z*v2.x - v1.x*v2.z;
//	result.z = v1.x*v2.y - v1.y*v2.x;
//	return result;
//}
__device__ float3 col(const float* matrix,int col)
{
	float3 result = make_float3(matrix[col], matrix[col + 3], matrix[col + 6]);
	return result;
}

////四元数q转化为旋转矩阵R
//__device__ void QuaternionToMatrix(const float*q, float*R)
//{
//	R[0] = 1 - 2 * q[1] * q[1] - 2 * q[2] * q[2];
//	R[1] = 2 * q[0] * q[1] - 2 * q[3] * q[2];
//	R[2] = 2 * q[0] * q[2] + 2 * q[3] * q[1];
//	R[3] = 2 * q[0] * q[1] + 2 * q[3] * q[2];
//	R[4] = 1 - 2 * q[0] * q[0] - 2 * q[2] * q[2];
//	R[5] = 2 * q[1] * q[2] - 2 * q[3] * q[0];
//	R[6] = 2 * q[0] * q[2] - 2 * q[3] * q[1];
//	R[7] = 2 * q[1] * q[2] + 2 * q[3] * q[0];
//	R[8] = 1 - 2 * q[0] * q[0] - 2 * q[1] * q[1];
//}
////q是一个四元数 x,y,z,w
//__device__ void extractRotation(int i,const float* A, float *q, const unsigned int maxIter) 
//{ 
//	float R[9];
//	float temp_q[4];
//	float norm;
//	for (unsigned int iter = 0; iter < maxIter; iter++) 
//	{ 
//		//translate q to matrix R
//		QuaternionToMatrix(q, R);
//		/*if (i == 37000)
//			printf("R is (%f,%f,%f)(%f,%f,%f)(%f,%f,%f)\n",
//				R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8]);
//		if (i == 37000)
//			printf("A is (%f,%f,%f)(%f,%f,%f)(%f,%f,%f)\n",
//				A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7], A[8]);*/
//		/*for (int i = 0; i < 9; ++i)
//			R[i] = q[i];*/
//		//Matrix3d R = q.matrix(); 
//		float3 omega = 
//			(cross(col(R, 0),col(A,0)) 
//				+ cross(col(R, 1),col(A,1))
//				+ cross(col(R, 2),col(A,2))) 
//			* (1.0 / fabs(dot(col(R, 0),col(A,0))
//					+ dot(col(R, 1),col(A,1)) + dot(col(R, 2),col(A,2))) + 1.0e-9);
//		if (i == 37000 && iter == 0)
//			printf("omega is (%f,%f,%f)\n", omega.x, omega.y, omega.z);
//		float w = sqrt(dot(omega,omega));
//		if (w < 1.0e-9) 
//			break; 
//		omega /= w;
//		temp_q[3] = w*q[3] - omega.x*q[0] - omega.y*q[1] - omega.z*q[2];
//		temp_q[0] = w*q[0] + omega.x*q[3] + omega.y*q[2] - omega.z*q[1];
//		temp_q[1] = w*q[1] + omega.y*q[3] + omega.z*q[0] - omega.x*q[2];
//		temp_q[2] = w*q[2] + omega.z*q[3] + omega.x*q[1] - omega.y*q[0];
//		//if (i == 37000)
//		//	printf("omega is (%f,%f,%f,%f)\n", omega.x, omega.y, omega.z, w);
//		/*a.w() * b.w() - a.x() * b.x() - a.y() * b.y() - a.z() * b.z(),
//			a.w() * b.x() + a.x() * b.w() + a.y() * b.z() - a.z() * b.y(),
//			a.w() * b.y() + a.y() * b.w() + a.z() * b.x() - a.x() * b.z(),
//			a.w() * b.z() + a.z() * b.w() + a.x() * b.y() - a.y() * b.x()*/
//		norm = sqrt(temp_q[0] * temp_q[0] + temp_q[1] * temp_q[1] + temp_q[2] * temp_q[2] + temp_q[3] * temp_q[3]);
//		//if (norm < 1.0e-9)
//		//	break;
//		for (int i = 0; i < 4; ++i)
//			q[i] = temp_q[i] / (norm + 1.0e-9);
//		
//	} 
//}
__device__ void AxisToRotation(float* R,const float3 axis,const float angle)
{
	float co = cos(angle), si = sin(angle);
	R[0] = co + (1 - co)*axis.x*axis.x; R[1] = (1 - co)*axis.x*axis.y - si*axis.z; R[2] = (1 - co)*axis.x*axis.z + si*axis.y;
	R[3] = (1 - co)*axis.y*axis.x + si*axis.z; R[4] = co + (1 - co)*axis.y*axis.y; R[5] = (1 - co)*axis.y*axis.z - si*axis.x;
	R[6] = (1 - co)*axis.z*axis.x - si*axis.y; R[7] = (1 - co)*axis.z*axis.y + si*axis.x; R[8] = co + (1 - co)*axis.z*axis.z;
}
__device__ void extractRotation(const float*A, float*q, const unsigned int maxIter)
{
	float R[9];
	float norm;
	float3 sum = make_float3(0, 0, 0);
	float sum2 = 0;
	//float error = 100000,error2;
	for (unsigned int iter = 0; iter < maxIter; iter++)
	//while(true)
	{
		sum = make_float3(0, 0, 0);
		sum2 = 0;
		for (int i = 0; i < 3; ++i)
		{
			sum += cross(col(q, i), col(A, i));
			sum2 += dot(col(q, i), col(A, i));
		}
		
		sum2 = fabs(sum2) + 1.0e-9;
		sum /= sum2;
		sum2 = sqrt(dot(sum, sum));
		if (sum2 < 1.0e-9)
			break;
		sum /= sum2;
		AxisToRotation(R, sum, sum2);
		multiply_matrix3(R, q, q);
		/*error2 = 0;
		for (int k = 0; k < 3; ++k)
			error2 += dot(col(q, k), col(A, k));
		if (fabs(error - error2) < 1 || fabs((error - error2) / error) < 0.001)
			break;*/
	}

}
__device__ float det(const float* a){
	float det = a[0]*a[4]*a[8] + a[1]*a[5]*a[6] + a[2]*a[3]*a[7];
	det -= (a[2]*a[4]*a[6] + a[1]*a[3]*a[8] + a[5]*a[7]*a[0]);
	return det;
}
__device__ void tensorProduct(const float3 a,const float3 b,float* r)
{
	r[0] = a.x * b.x; r[1] = a.x * b.y; r[2] = a.x * b.z;
	r[3] = a.y * b.x; r[4] = a.y * b.y; r[5] = a.y * b.z;
	r[6] = a.z * b.x; r[7] = a.z * b.y; r[8] = a.z * b.z;
}

//逆矩阵求解
__device__ void InverseMatrix3(float * B)
{
	float  E[9];
	for (int i = 0; i<3; ++i)
	{
		for (int j = 0; j<3; ++j)
			E[i*3 + j] = 0;
		E[i*3 + i] = 1;
	}
	for (int k = 0; k<3; ++k)
	{
		//对自己这行除以a[k][k]
		for (int j = k + 1; j<3; ++j)
			B[k*3 + j] = B[k*3 + j] / B[k*3 + k];
		for (int j = 0; j<3; ++j)
			E[k*3 + j] /= B[k*3 + k];
		B[k*3 + k] = 1.0;
		//对每一行减去a[i][k] * a[k][j]
		for (int i = k + 1; i<3; ++i)
		{
			for (int j = k + 1; j<3; ++j)
			{
				B[i*3 + j] = B[i*3 + j] - B[i*3 + k] * B[k*3 + j];
			}
			for (int j = 0; j<3; ++j)
				E[i*3 + j] -= B[i*3 + k] * E[k*3 + j];
			B[i*3 + k] = 0;
		}
	}
	for (int k = 2; k >= 0; --k)
	{
		//对每一行减去B[i][k]
		for (int i = k - 1; i >= 0; --i)
		{
			for (int j = 0; j<3; ++j)
				E[i*3 + j] -= B[i*3 + k] * E[k*3 + j];
			B[i*3 + k] = 0;
		}
	}
	for (int i = 0; i < 9; ++i)
		B[i] = E[i];

}

//Change density if needed
__global__ void mfChangeDensity (bufList buf,int pnum,const float scale)
{
	simData.mf_dens[1] *= scale;
	simData.mf_up = 1;
	simData.mf_visc[1] = simData.mf_visc[0];
	simData.VL = 0.3;
	simData.VL2 = 0.3*0.3;
}

//The forces of boundary to fluid
__device__ float3 nor(float3 p)
{
	float n1 = 0,n2 = 0,n3 = 0;
	if (p.y<(int)simData.pboundmin.y) n2 = 1.0;
	if (p.x<(int)simData.pboundmin.x) n1 = 1.0;
	if (p.x>(int)simData.pboundmax.x) n1 = -1.0;
	if (p.z<(int)simData.pboundmin.z) n3 = 1.0;
	if (p.z>(int)simData.pboundmax.z) n3 = -1.0;
	return make_float3(n1,n2,n3);
}
__device__ double flushData ( int i, float3 p, int cell, bufList buf )
{			
	float3 dist;
	float dsq, c, sum;
	//float massj;
	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2/d2;
	int j;
	//float maxdis = 88888;
//	register float cmterm;

	sum = 0.0;
	if ( buf.mgridcnt[cell] == 0 ) return 0;
	
	int cfirst = buf.mgridoff[ cell ];
	int clast = cfirst + buf.mgridcnt[ cell ];
	for ( int cndx = cfirst; cndx < clast; cndx++ ){ 
		if (buf.misbound[buf.mgrid[cndx]] == 0)
		{
			j = buf.mgrid[cndx];
			dist = p - buf.mpos[ buf.mgrid[cndx] ];
			dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
			if ( dsq < r2 && dsq > 0.0)
			{
				c = (r2 - dsq)*d2;
				sum += c * c * c * buf.mf_restmass[j]*dot(buf.mveleval[j],nor(buf.mpos[i]));
			}
		}
	}
	//c = r2*d2;
	//sum += c*c*c*buf.mf_restmass[i];
	return sum;
}

__device__ void findNearest ( int i, float3 p, int cell, bufList buf )
{			
	float3 dist;
	float dsq;
//	float massj;
	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2/d2;
	int j;
	float maxdis = 88888;

//	register float cmterm;
	//register float3 alphagrad[MAX_FLUIDNUM];

	//sum = 0.0;

	if ( buf.mgridcnt[cell] == 0 ) return ;
	
	int cfirst = buf.mgridoff[ cell ];
	int clast = cfirst + buf.mgridcnt[ cell ];
	for ( int cndx = cfirst; cndx < clast; cndx++ ) {
#ifdef NEW_BOUND
		if (buf.misbound[buf.mgrid[cndx]] == 0)
		{
			j = buf.mgrid[cndx];
			dist = p - buf.mpos[ buf.mgrid[cndx] ];
			dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);

			if ( dsq < r2 && dsq > 0.0 && dsq*d2<maxdis) 
			{
				maxdis = dsq*d2;
				buf.midsort[i] = j;
			} 
		}
#else
		j = buf.mgrid[cndx];
		dist = p - buf.mpos[ buf.mgrid[cndx] ];
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);

		if ( dsq < r2 && dsq > 0.0 && dsq*d2<maxdis) 
		{
			maxdis = dsq*d2;
			buf.midsort[i] = j;
		} 
#endif
	}
	
	return ;
}
__global__ void mfFindNearest (bufList buf,int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;
	
	// Get search cell
	int nadj = (1*simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[ i ];
	if ( gc == GRID_UNDEF ) return;						// particle out-of-range
	gc -= nadj;
	
	// Sum Pressures
	float3 pos = buf.mpos[ i ];
#ifdef NEW_BOUND 
	if (buf.misbound[i]==1)
	{
		buf.midsort[i] = i;
		buf.mf_restmass[i] = simData.pmass;
		for (int c = 0; c<simData.gridAdjCnt; c++)
		{
			findNearest(i,pos,gc+simData.gridAdj[c],buf);
		}
		if (buf.midsort[i]!=i)
			buf.mf_restmass[i] = buf.mf_restmass[buf.midsort[i]];

	}
#endif
}

//Sorting
__global__ void InitialSort ( bufList buf, int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;
	register float3 gridMin = simData.gridMin;
	register float3 gridDelta = simData.gridDelta;
	register int3 gridRes = simData.gridRes;
	register int3 gridScan = simData.gridScanMax;
//	register float poff = simData.psmoothradius / simData.psimscale;

	register int		gs;
	register float3		gcf;
	register int3		gc;

	gcf = (buf.mpos[i] - gridMin) * gridDelta; 
	gc = make_int3( int(gcf.x), int(gcf.y), int(gcf.z) );
	gs = (gc.y * gridRes.z + gc.z)*gridRes.x + gc.x;
	if ( gc.x >= 1 && gc.x <= gridScan.x && gc.y >= 1 && gc.y <= gridScan.y && gc.z >= 1 && gc.z <= gridScan.z ) {
		buf.mgcell[i] = gs;											// Grid cell insert.
		buf.midsort[i] = i;
//		buf.mgndx[i] = atomicAdd ( &buf.mgridcnt[ gs ], 1 );		// Grid counts.
//		gcf = (-make_float3(poff,poff,poff) + buf.mpos[i] - gridMin) * gridDelta;
//		gc = make_int3( int(gcf.x), int(gcf.y), int(gcf.z) );
//		gs = ( gc.y * gridRes.z + gc.z)*gridRes.x + gc.x;
		//buf.mcluster[i] = gs;				-- make sure it is allocated!
	} else {
		buf.mgcell[i] = GRID_UNDEF;
		buf.midsort[i] = i;
		//buf.mcluster[i] = GRID_UNDEF;		-- make sure it is allocated!
	}
}
__global__ void CalcFirstCnt ( bufList buf, int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i>=pnum) return;
	if ((i == 0 || buf.mgcell[i]!=buf.mgcell[i-1]))
	{
		if (buf.mgcell[i]!=GRID_UNDEF)buf.mgridoff[buf.mgcell[i]] = i;
	}
	__syncthreads();
	if (i!=0 && buf.mgcell[i]!=buf.mgcell[i-1] && buf.mgcell[i-1]!=GRID_UNDEF)
		buf.mgridcnt[buf.mgcell[i-1]] = i;
	if (i == pnum-1 && buf.mgcell[i]!=GRID_UNDEF)
		buf.mgridcnt[buf.mgcell[i]] = i + 1;
	/*
	__shared__ uint scell[512];   // [blockDim.x+1}
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;		// particle index
	uint cel;
	if (i<pnum && buf.mgcell[i] != GRID_UNDEF)
	{
		cel=buf.mgcell[i];
		scell[threadIdx.x+1]=cel;
		if(i&&!threadIdx.x)scell[0]=buf.mgcell[i-1];
	}
	__syncthreads();
	if(i<pnum && buf.mgcell[i] != GRID_UNDEF)
	{
		if(!i||cel!=scell[threadIdx.x])
		{
			buf.mgridoff[cel]=i;
			if (i)
			{
				buf.mgridcnt[scell[threadIdx.x]] = i;
			}
			if (i == pnum - 1)
				buf.mgridcnt[scell[threadIdx.x]] = i+1;
		}
	}
	else if (i<pnum)
	{
		if (buf.mgcell[i] != scell[threadIdx.x])
		{
			buf.mgridcnt[scell[threadIdx.x]] = i;
		}
	}
	*/
}
__global__ void GetCnt ( bufList buf, int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;		// particle index
	if (i>=pnum) return ;
	if (buf.mgcell[i]!=GRID_UNDEF)
	{
		buf.mgndx[i] = i - buf.mgridoff[buf.mgcell[i]];
		if (buf.mgndx[i] == 0)
			buf.mgridcnt[buf.mgcell[i]] -= buf.mgridoff[buf.mgcell[i]];
	}
}
__global__ void CountingSortFull_ ( bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;		// particle index				
	if ( i >= pnum ) return;

	uint icell = *(uint*) (buf.msortbuf + pnum*BUF_GCELL + i*sizeof(uint) );
	uint indx =  *(uint*) (buf.msortbuf + pnum*BUF_GNDX + i*sizeof(uint) );
	int sort_ndx = buf.mgridoff[ icell ] + indx;				// global_ndx = grid_cell_offet + particle_offset
//	uint j = i;
	i = buf.midsort[i];
	if ( icell != GRID_UNDEF ) {
		buf.mgrid[ sort_ndx ] = sort_ndx;			// full sort, grid indexing becomes identity
		char* bpos = buf.msortbuf + i*sizeof(float3);
		buf.mpos[ sort_ndx ] =		*(float3*) (bpos);
		buf.mveleval[ sort_ndx ] =	*(float3*) (bpos + pnum*BUF_VELEVAL );
		buf.mpress[ sort_ndx ] =	*(float*) (buf.msortbuf + pnum*BUF_PRESS + i*sizeof(float) );
#ifdef NEW_BOUND
		buf.misbound[ sort_ndx ] =		*(int*) (buf.msortbuf + pnum*BUF_ISBOUND+ i*sizeof(int) );		// ((uint) 255)<<24; -- dark matter
#endif
		buf.mgcell[ sort_ndx ] =	icell;
		buf.mgndx[ sort_ndx ] =		indx;	

		//multi fluid
		int mul_sort_ndx = sort_ndx*MAX_FLUIDNUM;
		for( uint fcount = 0; fcount < simData.mf_catnum; fcount++)
		{
			//char* bmul = buf.msortbuf + i*sizeof(float)*MAX_FLUIDNUM + fcount * sizeof(float);
			buf.mf_alpha[mul_sort_ndx+fcount] =			*(float*)(buf.msortbuf +  pnum*BUF_ALPHA +   i*sizeof(float)*MAX_FLUIDNUM + fcount * sizeof(float));
			buf.mf_alpha_next[mul_sort_ndx+fcount] =		*(float*)(buf.msortbuf +  pnum*BUF_ALPHAPRE+ i*sizeof(float)*MAX_FLUIDNUM + fcount * sizeof(float));
		
			//porous
			for (int l = 0; l < MAX_SOLIDNUM; ++l)
			{
				buf.mf_beta[mul_sort_ndx*MAX_SOLIDNUM + fcount*MAX_SOLIDNUM+l]
					= *(float*)(buf.msortbuf + pnum*BUF_ABSORBEDPERCENT + i * sizeof(float)*MAX_FLUIDNUM*MAX_SOLIDNUM + fcount * sizeof(float)*MAX_SOLIDNUM + l*sizeof(float));
				buf.mf_beta_next[mul_sort_ndx*MAX_SOLIDNUM + fcount*MAX_SOLIDNUM + l] 
					= *(float*)(buf.msortbuf + pnum*BUF_BETANEXT + i * sizeof(float)*MAX_FLUIDNUM*MAX_SOLIDNUM + fcount * sizeof(float)*MAX_SOLIDNUM + l * sizeof(float));
			}
			//buf.capillaryPotentials[mul_sort_ndx + fcount] = *(float*)(buf.msortbuf + pnum*BUF_CP + i * sizeof(float)*MAX_FLUIDNUM + fcount * sizeof(float));

		}
		//buf.mf_pressure_modify[ sort_ndx ] = *(float*) (buf.msortbuf + pnum*BUF_PRESSMODI + i*sizeof(float));
		buf.mf_restmass[ sort_ndx ] = *(float*) (buf.msortbuf + pnum*BUF_RMASS + i*sizeof(float));

		//buf.mf_velxcor[sort_ndx] = *(float3*)(buf.msortbuf + pnum*BUF_VELXCOR + i*sizeof(float3));
		buf.MFtype[sort_ndx] = *(int*)(buf.msortbuf+ pnum*BUF_INDICATOR + i*sizeof(int));
		//elastic information
		buf.elasticID[sort_ndx] = *(uint*)(buf.msortbuf + pnum*BUF_ELASTICID + i * sizeof(uint));

		if(buf.MFtype[sort_ndx] == 2)
			buf.particleID[buf.elasticID[sort_ndx]] = sort_ndx;
		if(_example == 2 && buf.MFtype[sort_ndx] >= 2)
			buf.particleID[buf.elasticID[sort_ndx]] = sort_ndx;
	}
}

//compute pressure
__device__ float mfContributePressure ( int i, float3 p, int cell, bufList buf, float& sum_solid, float& sum_fluid)
{			
	float3 dist;
	float dsq, c, sum;
	float massj;
	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2/d2;

	sum = 0.0;
	int j;

	if ( buf.mgridcnt[cell] == 0 )
		return 0.0;
	
	int cfirst = buf.mgridoff[ cell ];
	int clast = cfirst + buf.mgridcnt[ cell ];

	for ( int cndx = cfirst; cndx < clast; cndx++ ) {
		j = buf.mgrid[cndx];
		dist = p - buf.mpos[ buf.mgrid[cndx] ];

		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
		if ( dsq < r2 && dsq > 0.0) {
			c = (r2 - dsq)*d2;
			sum += c * c * c * buf.mf_restmass[i];
			if (buf.MFtype[i] == buf.MFtype[j]) 
			{
				if (buf.MFtype[i] == 0)
					sum_fluid += c * c * c * buf.mf_restmass[i];
				else
					sum_solid += c * c * c * buf.mf_restmass[i];
			}
			if (buf.MFtype[i] + buf.MFtype[j] == 9)
				sum_solid += c * c * c * buf.mf_restmass[i];
		} 
	}
	return sum;
}

__device__ float mfContributePressureInit ( int i, float3 p, int cell, bufList buf )
{			
	float3 dist;
	float dsq, c, sum;
	float massj;
	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2/d2;

	sum = 0.0;
	int j;

	if ( buf.mgridcnt[cell] == 0 )
		return 0.0;
	
	int cfirst = buf.mgridoff[ cell ];
	int clast = cfirst + buf.mgridcnt[ cell ];

	for ( int cndx = cfirst; cndx < clast; cndx++ ) {
		j = buf.mgrid[cndx];
		//if( buf.MFtype[i] == 2 && buf.MFtype[j]!=2)
		if(buf.MFtype[i]!=buf.MFtype[j])
			continue;
		dist = p - buf.mpos[ buf.mgrid[cndx] ];
		massj = buf.mf_restmass[ buf.mgrid[cndx] ];
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
		if ( dsq < r2 && dsq > 0.0) {
			c = (r2 - dsq)*d2;
			sum += c * c * c * massj;	
		} 
	}

	return sum;
}

__global__ void mfPreComputeDensity ( bufList buf, int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;

	// Get search cell
	int nadj = (1*simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[ i ];
	if ( gc == GRID_UNDEF ) return;						// particle out-of-range
	gc -= nadj;

	// Sum Pressures
	float3 pos = buf.mpos[ i ];
	float dens = buf.mf_restdensity[i];
	float sum = 0.0;
	float sum_solid = 0.0;
	float sum_fluid = 0.0;
	for (int c=0; c < simData.gridAdjCnt; c++) {
		sum += mfContributePressure ( i, pos, gc + simData.gridAdj[c], buf, sum_solid, sum_fluid);
			//__syncthreads();
	}

	// Compute Density & Pressure
	sum += simData.r2 * simData.r2 * simData.r2 * buf.mf_restmass[i];
	sum_solid += simData.r2 * simData.r2 * simData.r2 * buf.mf_restmass[i];
	//sum_fluid += simData.r2 * simData.r2 * simData.r2 * buf.mf_restmass[i];

	sum = sum * simData.poly6kern;
	sum_solid = sum_solid * simData.poly6kern;
	sum_fluid = sum_fluid * simData.poly6kern;

	if ( sum == 0.0 ) sum = 1.0;

#ifdef NEW_BOUND
	buf.mdensity[ i ] = 1.0f / sum;

	if (buf.MFtype[i] != 0) 
	{
		buf.density_solid[i] = 1.0f / sum_solid;
		//if (i % 10 == 0)
		//	printf("solid density is %f\n", buf.density_solid[i]);0.0026
	}
#else
	buf.mpress[ i ] = ( sum - dens ) * simData.pintstiff;
	//buf.mpress[ i ] = (pow( sum/dens,7.0f )-1) * simData.pintstiff;
	//buf.mpress[ i ] = simData.pintstiff * dens * (pow( sum/dens,7.0f )-1);
	buf.mdensity[ i ] = 1.0f / sum;
#endif

}

__global__ void mfComputePressure ( bufList buf, int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;

	// Get search cell
	int nadj = (1*simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[ i ];
	if ( gc == GRID_UNDEF ) return;						// particle out-of-range
	gc -= nadj;

	// Sum Pressures
	float3 pos = buf.mpos[ i ];
	float dens = buf.mf_restdensity[i];
	float sum = 0.0;
	float sum_solid = 0;
	float sum_fluid = 0;
	for(uint fcount = 0; fcount<simData.mf_catnum;fcount++)
	{
		//buf.mf_alphagrad[i*MAX_FLUIDNUM+fcount] = make_float3(0,0,0);
		buf.mf_alpha_next[i*MAX_FLUIDNUM+fcount] = buf.mf_alpha[i*MAX_FLUIDNUM+fcount];

		buf.mf_beta[i*MAX_FLUIDNUM + fcount] = 0;
		buf.mf_beta_next[i*MAX_FLUIDNUM + fcount] = 0;
	}
	/*if (buf.MFtype[i] == 0 && buf.mpos[i].y < 30) 
	{
		buf.mf_alpha_next[i*MAX_FLUIDNUM + 2] = buf.mf_alpha[i*MAX_FLUIDNUM + 2] = 0;
		buf.mf_beta[i*MAX_FLUIDNUM + 2] = 0.5;
	}*/
	for (int c=0; c < simData.gridAdjCnt; c++) {
		sum += mfContributePressure ( i, pos, gc + simData.gridAdj[c], buf, sum_solid, sum_fluid );
			//__syncthreads();
	}

		// Compute Density & Pressure
		sum += simData.r2 * simData.r2 * simData.r2 * buf.mf_restmass[i];

		sum = sum * simData.poly6kern;
		if ( sum == 0.0 ) sum = 1.0;
#ifdef NEW_BOUND
	if (buf.misbound[i] ==1)
	{
		//buf.mpress[i] = ( sum - dens ) * simData.pextstiff;
		//buf.mpress[ i ] = (pow( sum/dens,7.0f )-1) * simData.pintstiff;
		//buf.mpress[ i ] += simData.pintstiff * dens * (pow( sum/dens,7.0f )-1);
		buf.mpress[ i ] = ( sum - dens ) * simData.pintstiff;
		//if (buf.mpress[i]<0) buf.mpress[i] = 0;
	}
	else
	{
		//buf.mpress[ i ] = ( sum - dens ) * simData.pintstiff;
		//buf.mpress[ i ] = (pow( sum/dens,7.0f )-1) * simData.pintstiff;
		//buf.mpress[ i ] = ( sum - dens ) * simData.pintstiff;

		if( buf.MFtype[i]>=2)
			buf.mpress[ i ] = simData.solid_pfactor * dens * (pow( sum/dens,7.0f )-1);
		if( buf.MFtype[i]==0){
			buf.mpress[ i ] = simData.fluid_pfactor * dens * (pow( sum/dens,7.0f )-1);
		if(buf.mpress[i]<0)
			buf.mpress[i]=0;
		
		}
//		buf.mdensity[ i ] = 1.0f / sum;
	}
#else
	buf.mpress[ i ] = ( sum - dens ) * simData.pintstiff;
	//buf.mpress[ i ] = (pow( sum/dens,7.0f )-1) * simData.pintstiff;
	//buf.mpress[ i ] = simData.pintstiff * dens * (pow( sum/dens,7.0f )-1);
	buf.mdensity[ i ] = 1.0f / sum;
#endif
	//buf.mpress[ i ] = (pow( sum/dens,7.0f )-1) * simData.pintstiff;
	//buf.mpress[ i ] = simData.pintstiff * dens * (pow( sum/dens,7.0f )-1);

	buf.vel_mid[i] = buf.mveleval[i];
}

__global__ void initDensity(bufList buf,int pnum){
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;
	if(buf.MFtype[i] == 0) //no need for fluid particles
		return;

	// Get search cell
	int nadj = (1*simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[ i ];
	if ( gc == GRID_UNDEF ) return;						// particle out-of-range
	gc -= nadj;

	float3 pos = buf.mpos[ i ];
	float sum = 0.0;
	for (int c=0; c < simData.gridAdjCnt; c++) {
		sum += mfContributePressureInit ( i, pos, gc + simData.gridAdj[c], buf );
	}
	sum += simData.r2 * simData.r2 * simData.r2 * buf.mf_restmass[i];
	sum = sum * simData.poly6kern;
	//now sum is density
	buf.mf_restdensity[i] = sum;
	//if (i == 0)
	//	printf("rest density is %f\n", buf.mf_restdensity[i]);
	buf.mveleval[i] = make_float3(0, 0, 0);
	buf.vel_mid[i] = make_float3(0, 0, 0);
}

//compute drift velocity
__device__ void contributeDriftVel( int i, int muli, float3 ipos, float idens, float ipress, int cell, bufList buf, float* ialpha, float* imassconcen, float3* idriftvelterm, float relax_coef, float3*ialphagrad){
	float dsq, c;
	register float d2 = simData.psimscale * simData.psimscale;	
	register float r2 = simData.r2/d2;

	float3 dist;		
	float cmterm;
	float pmterm;
	int j, mulj;

	if ( buf.mgridcnt[cell] == 0 ) return;	

	int cfirst = buf.mgridoff[ cell ];
	int clast = cfirst + buf.mgridcnt[ cell ];

	float3 force = make_float3(0,0,0);
	float3 pgrad[MAX_FLUIDNUM];
	float3 pgradsum;

	float3 cpgrad[MAX_FLUIDNUM];
	float3 cpgradsum;
	for ( int cndx = cfirst; cndx < clast; cndx++ ) {										
		j = buf.mgrid[ cndx ];
		mulj = j * MAX_FLUIDNUM;
		dist = ( ipos - buf.mpos[ j ] );		// dist in cm
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
		dist *= simData.psimscale;
		if ( dsq < r2 && dsq > 0) {
			//cx = (r2-dsq)*d2;
			dsq = sqrt(dsq*d2);
			
			c = ( simData.psmoothradius - dsq ); 
			cmterm = simData.spikykern * c * c / dsq * buf.mf_restmass[j] * buf.mdensity[j];
			
			if (buf.MFtype[j] == 0)
			{
				if (buf.mf_alpha_sum[j] < 0.000001)
					continue;
				//pressure
				pgradsum = make_float3(0, 0, 0);
				for (uint fcount = 1; fcount < simData.mf_catnum; fcount++)
				{
					float jalphaprecount = buf.mf_alpha[mulj + fcount] / buf.mf_alpha_sum[j];
					//float ialphaprecount = ialpha_pre[fcount];
					pmterm = cmterm * (-ialpha[fcount] * ipress + jalphaprecount*buf.mpress[j]);
					//pmterm = cmterm * (-ialpha_pre[fcount]*ipress + buf.mf_alpha_pre[mulj+fcount]*buf.mpress[j]);
					pgrad[fcount] = pmterm * dist;
					if (isnan(dot(cpgrad[fcount], cpgrad[fcount])))
						continue;
					pgradsum += pgrad[fcount] * imassconcen[fcount];
					//grad alpha
					ialphagrad[fcount] += (jalphaprecount - ialpha[fcount]) * cmterm * dist;
				}

				for (uint fcount = 1; fcount < simData.mf_catnum; fcount++)
				{
					if (isnan(dot(cpgrad[fcount], cpgrad[fcount])))
						continue;
					idriftvelterm[fcount] -= relax_coef * (pgrad[fcount] - pgradsum);
				}
			}
			if(buf.MFtype[j] >= 2)//capillary term
			{
				cpgradsum = make_float3(0, 0, 0);
				for(int k=1;k<simData.mf_catnum;++k)
				{
					//float jalphaprecount = buf.mf_alpha[mulj + k] / buf.mf_alpha_sum[j];
					pmterm = cmterm * (-buf.pressure_water[i*simData.mf_catnum*MAX_SOLIDNUM+k*MAX_SOLIDNUM+buf.MFtype[j]-2] + buf.pressure_water[j*simData.mf_catnum*MAX_SOLIDNUM + k*MAX_SOLIDNUM + buf.MFtype[j] - 2]);
					cpgrad[k] = pmterm * dist;
					if (isnan(dot(cpgrad[k], cpgrad[k])))
					{
						//printf("cpgrad %d is (%f,%f,%f)\n", k, cpgrad[k].x, cpgrad[k].y, cpgrad[k].z);
						continue;
					}
					cpgradsum += cpgrad[k] * imassconcen[k];

				}
				for (uint fcount = 1; fcount < simData.mf_catnum; fcount++)
				{
					if (isnan(dot(cpgrad[fcount], cpgrad[fcount])))
						continue;
					idriftvelterm[fcount] -= relax_coef*simData.relax2* (cpgrad[fcount] - cpgradsum);
				}
			}
		}
	}
}
__global__ void applyAlphaAndBeta(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum)
		return;
	if (buf.MFtype[i] != 0)
		return;
	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	//if (gc == GRID_UNDEF) return;						// particle out-of-range
	gc -= nadj;

	register float3 accel = -buf.mforce[i];				// final accel (g-a) of last step was stored in here cf. advance, 
	register uint muloffseti = i * MAX_FLUIDNUM;

	float alphasum = 0;
	for (uint fcount = 0; fcount < simData.mf_catnum; fcount++)
	{
		//float temp = buf.mf_alpha[muloffseti+fcount];
		//buf.mf_alpha_pre[muloffseti+fcount] = temp;				//alpha->alpha_pre
		buf.mf_alpha[muloffseti + fcount] = buf.mf_alpha_next[muloffseti + fcount];
		alphasum += buf.mf_alpha_next[muloffseti + fcount];
		//buf.mf_alphagrad[i*MAX_FLUIDNUM + fcount] = make_float3(0, 0, 0);
	}
	for (uint fcount = 0; fcount < MAX_FLUIDNUM*MAX_SOLIDNUM; fcount++)
		buf.mf_beta[i*MAX_FLUIDNUM*MAX_SOLIDNUM + fcount] = buf.mf_beta_next[i*MAX_FLUIDNUM*MAX_SOLIDNUM + fcount];

	float newdens, newvisc, newmass, newdensout;
	//Restdensity Update
	newdens = 0.0;
	newvisc = 0.0;
	//newdensout = 0.0;
	
	//newmass = 0.0;
	for (uint fcount = 1; fcount < simData.mf_catnum; fcount++)
	{
		newdens += buf.mf_alpha[i*MAX_FLUIDNUM + fcount] * simData.mf_dens[fcount];
		newvisc += buf.mf_alpha[i*MAX_FLUIDNUM + fcount] * simData.mf_visc[fcount];
	}
	float betasum = 0;
	for (uint fcount = 1; fcount < simData.mf_catnum; fcount++)
		for(int l=0;l<MAX_SOLIDNUM;++l)
		{
			newdens += buf.mf_beta[i*MAX_FLUIDNUM*MAX_SOLIDNUM + fcount*MAX_SOLIDNUM + l] * simData.mf_dens[fcount];
			newvisc += buf.mf_beta[i*MAX_FLUIDNUM*MAX_SOLIDNUM + fcount*MAX_SOLIDNUM + l] * simData.mf_visc[fcount];
			betasum += buf.mf_beta[i*MAX_FLUIDNUM*MAX_SOLIDNUM + fcount*MAX_SOLIDNUM + l];
		}
	//if (buf.MFtype[i] == 0)
	{
		buf.mf_restdensity[i] = newdens;
		//buf.mf_restmass[i] = newmass;
		buf.mf_visc[i] = newvisc;
		buf.mf_restdensity_out[i] = newdensout;
	}

	if (buf.mf_restdensity[i] <= 10)
		printf("rest den is %f, alpha is (%f,%f,%f), betasum is %f\n",
			buf.mf_restdensity[i], buf.mf_alpha[i*MAX_FLUIDNUM + 1], buf.mf_alpha[i*MAX_FLUIDNUM + 2], buf.mf_alpha[i*MAX_FLUIDNUM + 3],
			betasum);
}
__global__ void mfComputeDriftVel( bufList buf, int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum)
		return;
	//if (i % 1000 == 0)
	//	for (uint fcount = 0; fcount < simData.mf_catnum; fcount++)
	//		printf("particle %d's pressure is %f\n",
	//			i, buf.mpress[i]);
	if (buf.MFtype[i] != 0)
		return;
	if (buf.mf_alpha_sum[i] <= 0.01)
	{
		for (uint fcount = 1; fcount < simData.mf_catnum; fcount++)
			buf.mf_vel_phrel[i*MAX_FLUIDNUM + fcount] = make_float3(0, 0, 0);
		return;
	}
	// Get search cell
	int nadj = (1*simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[ i ];
	if ( gc == GRID_UNDEF ) return;						// particle out-of-range
	gc -= nadj;
	float relax_coef = simData.relax;					// temporary relax time related coefficient
	//register float relax_coef = 0;
	float sigma = 0.001f;//0.001f;						//diffusion&tension coefficient
	float cont, conts, contr;
	cont = simData.cont;
	conts = simData.cont1;
	contr = simData.cont2;
	float3 accel = -buf.mforce[i];				// final accel (g-a) of last step was stored in here cf. advance, 
	//register float massFrack[MAX_FLUIDNUM];
	uint muloffseti = i * MAX_FLUIDNUM;
	float invdens = 1.0/buf.mf_restdensity_out[i];
	float dsum;
	float vrx, vry, vrz;
	float tdiff;
	float3 ssum;

	float alpha[MAX_FLUIDNUM],mass_concen[MAX_FLUIDNUM];
	float ipress = buf.mpress[ i ];
	float3 ipos = buf.mpos[ i ];
	float idens = buf.mdensity[ i ];
	float3 driftVelterm[MAX_FLUIDNUM],alphaGradterm[MAX_FLUIDNUM];
	float3 sterm[MAX_FLUIDNUM];

	//various viscosity
	relax_coef /= buf.mf_visc[i];

	//relax_coef *= (99*buf.mf_alpha_pre[i*MAX_FLUIDNUM+2]+1);
	//third term
	for(uint fcount = 1;fcount < simData.mf_catnum; fcount++)
	{
		//float temp = buf.mf_alpha[muloffseti+fcount];
		//buf.mf_alpha_pre[muloffseti+fcount] = temp;				//alpha->alpha_pre
		if (buf.mf_alpha_sum[i] > 0.0001)
			alpha[fcount] = buf.mf_alpha[muloffseti + fcount] / buf.mf_alpha_sum[i];
		else
			alpha[fcount] = 0;
		//mass_concen[fcount] = alpha[fcount]*simData.mf_dens[fcount]*invdens;
		mass_concen[fcount] = alpha[fcount] * simData.mf_dens[fcount] * invdens;
		//if (isnan(mass_concen[fcount]))
		//	printf("alpha pre is %f, invdens is %f\n",
		//		alpha_pre[fcount], invdens);
		driftVelterm[fcount] = make_float3(0,0,0);
		alphaGradterm[fcount] = make_float3(0,0,0);
	}
	
	for (int c=0; c < simData.gridAdjCnt; c++) {
		contributeDriftVel ( i, muloffseti, ipos, idens, ipress, gc + simData.gridAdj[c], buf, alpha, mass_concen, driftVelterm, relax_coef, alphaGradterm);
	}

	for( uint fcount = 1; fcount < simData.mf_catnum; fcount++)
	{
		//buf.mf_vel_phrel[muloffseti+fcount] = cont * contr * driftVelterm[fcount];
		float3 vel = cont * contr * driftVelterm[fcount];
		buf.mf_vel_phrel[muloffseti+fcount] = vel;
	}

	//first term & second term
	dsum = 0;
	ssum = make_float3(0,0,0);
	if(buf.mf_alpha_sum[i] > 0.01)
	for( uint fcount = 1; fcount < simData.mf_catnum; fcount++)
	{
		float temp = buf.mf_alpha[muloffseti+fcount] / buf.mf_alpha_sum[i];
		dsum += temp * simData.mf_dens[fcount] * simData.mf_dens[fcount] * invdens;
		if (temp > 0.0001)
			//sterm[fcount] = buf.mf_alphagrad[muloffseti+fcount]/temp;
			sterm[fcount] = alphaGradterm[fcount] / temp;
		else
			sterm[fcount] = make_float3(0,0,0);
			//sterm[fcount] = alphaGradterm[fcount];
		ssum += sterm[fcount] * temp * simData.mf_dens[fcount] * invdens;
	}
	for( uint fcount = 1; fcount < simData.mf_catnum; fcount++)
	{
		tdiff = simData.mf_dens[fcount]-dsum;
		tdiff *= relax_coef;
		vrx = accel.x * tdiff;
		vry = accel.y * tdiff;
		vrz = accel.z * tdiff;
		buf.mf_vel_phrel[muloffseti+fcount] += make_float3(vrx,vry,vrz);

		buf.mf_vel_phrel[muloffseti+fcount] -= 
			cont * conts * sigma * (sterm[fcount]-ssum);
		if (isnan(dot(buf.mf_vel_phrel[muloffseti + fcount], buf.mf_vel_phrel[muloffseti + fcount])))
		//if(i%1000 ==0)
			printf("particle %d phase %d's vel is (%f,%f,%f),accel is (%f,%f,%f),alpha is %f, sterm is (%f,%f,%f), driftVelterm is (%f,%f,%f), press is %f, mass concern is (%f,%f,%f), alphaSum is %f, densityout is %f, pressure water is (%f,%f,%f,%f), visco is %f, relax_coef is %f\n",
				i, fcount, buf.mf_vel_phrel[muloffseti + fcount].x, buf.mf_vel_phrel[muloffseti + fcount].y,
				buf.mf_vel_phrel[muloffseti + fcount].z, accel.x, accel.y, accel.z,
				buf.mf_alpha[muloffseti + fcount], sterm[fcount].x, sterm[fcount].y, sterm[fcount].z,
				driftVelterm[fcount].x, driftVelterm[fcount].y, driftVelterm[fcount].z, buf.mpress[i],
				mass_concen[1], mass_concen[2], mass_concen[3], buf.mf_alpha_sum[i],buf.mf_restdensity_out[i],
				buf.pressure_water[muloffseti*MAX_SOLIDNUM+fcount*MAX_SOLIDNUM+0], buf.pressure_water[muloffseti*MAX_SOLIDNUM + fcount*MAX_SOLIDNUM + 1],
				buf.pressure_water[muloffseti*MAX_SOLIDNUM + fcount*MAX_SOLIDNUM + 2], buf.pressure_water[muloffseti*MAX_SOLIDNUM + fcount*MAX_SOLIDNUM + 3], buf.mf_visc[i], relax_coef);
	
	}
	
}
__device__ float3 contributeTDM(int i, int muli, float idens, float3 pos, int cell, bufList buf, float* ialpha_pre, float3* ivmk)
{
	float3 force = make_float3(0, 0, 0);
	if (buf.mgridcnt[cell] == 0) return force;
	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2;
	float3 dist, sf;
	float c, dsq2, dsq, q;
	int j, mulj;
	float cmterm;
	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];

	for (int cndx = cfirst; cndx < clast; cndx++)
	{
		j = buf.mgrid[cndx];
		if (buf.MFtype[j] != 0)
			continue;
		mulj = j * MAX_FLUIDNUM;
		dist = (buf.mpos[i] - buf.mpos[j])*simData.psimscale;
		dsq2 = dot(dist, dist);
		q = sqrt(dsq2 / r2);
		if (!(dsq2 < r2&&dsq2>0))
			continue;
		dsq = sqrt(dsq2);

		if (q <= 0.5)
			cmterm = simData.gradCubicSplineKern * (3 * q*q - 2 * q);
		else
			cmterm = -simData.gradCubicSplineKern * pow(1 - q, 2);
		cmterm *= buf.mf_restmass[j] * buf.mdensity[j] / dsq;

		//T_dm
		for (uint fcount = 0; fcount < simData.mf_catnum; fcount++)
		{
			float3 dtermj = cmterm * dot(buf.mf_vel_phrel[mulj + fcount], dist) * buf.mf_alpha[mulj + fcount] * buf.mf_vel_phrel[mulj + fcount];
			float3 dtermi = cmterm * dot(ivmk[fcount], dist) * ialpha_pre[fcount] * ivmk[fcount];
			//example 2 doesn't have this term
			force += (dtermj + dtermi) * simData.mf_dens[fcount] * idens;
		}
	}
	return force;
}
__global__ void mfComputeTDM(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;
	if (buf.MFtype[i] != 0) {
		return;
	}
	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) return;						// particle out-of-range
	gc -= nadj;
	bool error = false;
	// Sum Pressures
	float3 pos = buf.mpos[i];
	float dens = buf.mf_restdensity[i];
	float3 force = make_float3(0, 0, 0);

	register uint muloffseti = i * MAX_FLUIDNUM;
	register float alpha[MAX_FLUIDNUM];
	register float3 ivmk[MAX_FLUIDNUM];

	for (uint fcount = 1; fcount < simData.mf_catnum; fcount++)
	{
		alpha[fcount] = buf.mf_alpha[muloffseti + fcount];
		ivmk[fcount] = buf.mf_vel_phrel[muloffseti + fcount];
	}
	for (int c = 0; c < simData.gridAdjCnt; c++)
	{
		force += contributeTDM(i, muloffseti, buf.mdensity[i], pos, gc + simData.gridAdj[c], buf, alpha, ivmk);
	}
	if (isnan(dot(force,force)))
		//if(i%1000 ==0)
		printf("particle %d tdm is nan,  press is %f, alphaSum is %f, densityout is %f\n",
			i,  buf.mpress[i],
			buf.mf_alpha_sum[i], buf.mf_restdensity_out[i]);

	//bound force and gravity
	buf.mforce[i] += force;
	//buf.fluidForce[i] += force;
	buf.maccel[i] = buf.mforce[i];
}
//compute alpha advance
__device__ void contributeAlphaChange( int i, int muli, float3 ipos, float3 iveleval, float ipress, float idens, int cell, bufList buf, float* ialpha, float* ialphachange, float3* ivmk)
{
	float dsq, c;
	register float d2 = simData.psimscale * simData.psimscale;	
	register float r2 = simData.r2/d2;

	float3 dist, vmr, vkr;		
	float cmterm;
	int j, mulj;	
	//float3 jvmk[MAX_FLUIDNUM];
	float jalpha_prek;
	//float alphachange = 0.0;

	if ( buf.mgridcnt[cell] == 0 ) return;// make_float3(0,0,0);	

	int cfirst = buf.mgridoff[ cell ];
	int clast = cfirst + buf.mgridcnt[ cell ];

	//force = make_float3(0,0,0);
	//vterm = simData.lapkern * simData.pvisc;

	for ( int cndx = cfirst; cndx < clast; cndx++ ) {										
		j = buf.mgrid[ cndx ];	
#ifdef NEW_BOUND
		if (buf.misbound[j] ==1) continue;
#endif

		if(buf.MFtype[j] != buf.MFtype[i])
			continue;

		mulj = j * MAX_FLUIDNUM;
		dist = ( ipos - buf.mpos[ j ] );		// dist in cm
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
		dist *= simData.psimscale;
		if ( dsq < r2 && dsq > 0) {			
			dsq = sqrt(dsq*d2);
			c = ( simData.psmoothradius - dsq ); 
			cmterm = simData.spikykern * c * c / dsq * buf.mf_restmass[j] * buf.mdensity[j];
			vmr = buf.mveleval[j] - iveleval;

			for(uint fcount = 1; fcount < simData.mf_catnum; fcount++)
			{
				jalpha_prek = buf.mf_alpha[mulj+fcount];
				//-alpha_k * (nabla cdot v_m)

				ialphachange[fcount] -= 0.5 * cmterm * (jalpha_prek+ialpha[fcount]) * (vmr.x * dist.x + vmr.y * dist.y + vmr.z * dist.z);
				//buf.mf_alpha[muli+fcount] -= 0.5 * cmterm * (jalpha_prek+ialpha_pre[fcount]) * (vmr.x * dist.x + vmr.y * dist.y + vmr.z * dist.z);
				//-nabla cdot (alpha_k * u_mk)
				vkr = make_float3((jalpha_prek * buf.mf_vel_phrel[mulj+fcount].x + ialpha[fcount] * ivmk[fcount].x),
						(jalpha_prek * buf.mf_vel_phrel[mulj+fcount].y + ialpha[fcount] * ivmk[fcount].y),
						(jalpha_prek * buf.mf_vel_phrel[mulj+fcount].z + ialpha[fcount] * ivmk[fcount].z));
				ialphachange[fcount] -= cmterm * (vkr.x * dist.x + vkr.y * dist.y + vkr.z * dist.z);

				//buf.mf_alpha[muli+fcount] -= cmterm * (vkr.x * dist.x + vkr.y * dist.y + vkr.z * dist.z);
			}
			//pterm = simData.psimscale * -0.5f * c * simData.spikykern * ( ipress + buf.mpress[ j ] ) / dsq;
			//dterm = c * idens * (buf.mdensity[ j ] );
			//force += ( pterm * dist + vterm * ( buf.mveleval[ j ] - iveleval )) * dterm;
		}	
	}
	//return force;
	//return alphachange;
}
__global__ void mfComputeAlphaAdvance( bufList buf, int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum)
		return ;
	if (buf.MFtype[i] != 0)
		return;
	if (buf.mf_alpha_sum[i] < 0.01)
		return;
	// Get search cell
	int nadj = (1*simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[ i ];
	if ( gc == GRID_UNDEF ) return;						// particle out-of-range
	gc -= nadj;

	register uint muloffseti = i * MAX_FLUIDNUM;
	register float3 ipos = buf.mpos[ i ];
	register float3 iveleval = buf.mveleval[ i ];
	register float ipress = buf.mpress[ i ];
	register float idens = buf.mdensity[ i ];
	register float alpha[MAX_FLUIDNUM],alphachange[MAX_FLUIDNUM];
	register float3 ivmk[MAX_FLUIDNUM];

	for(uint fcount = 1;fcount < simData.mf_catnum; fcount++)
	{
		alpha[fcount] = buf.mf_alpha[muloffseti+fcount];
		alphachange[fcount] = 0.0f;
		ivmk[fcount] = buf.mf_vel_phrel[muloffseti+fcount];
		//buf.mf_alpha[muloffseti+fcount] = 0.0f;
	}

	for (int c=0; c < simData.gridAdjCnt; c++) {
		contributeAlphaChange ( i, muloffseti, ipos, iveleval, ipress, idens, gc + simData.gridAdj[c], buf, alpha, alphachange, ivmk);
	}

	for(uint fcount = 1;fcount < simData.mf_catnum; fcount++)
	{
		//buf.mf_alpha[muloffseti+fcount] += alphachange[fcount] * simData.mf_dt;
		alphachange[fcount] *= simData.mf_dt;

		//alphachange limit
		if(alphachange[fcount]<-0.99)
		{
			alphachange[fcount] = -0.99;// * ((int)(buf.mf_alpha[muloffseti+fcount]>0)-(int)(buf.mf_alpha[muloffseti+fcount]<0));
		}
		buf.mf_alphachange[i*MAX_FLUIDNUM + fcount] = alphachange[fcount];
		//if (abs(alphachange[fcount]) >= 0.001)
		//	printf("particle %d's phase %d's alpha change is %f\n", i, fcount, alphachange[fcount]);
		buf.mf_alpha_next[muloffseti+fcount] = alphachange[fcount] + alpha[fcount];
		
		//buf.mf_alpha_next[muloffseti + fcount] = alpha[fcount];
		if (isnan(alphachange[fcount]) || isnan(alpha[fcount]))
			printf("particle %d phase %d's alpha change is %f, pre alpha is %f, vmk is (%f,%f,%f)\n",
				i, fcount, alphachange[fcount], alpha[fcount],
				buf.mf_vel_phrel[i*MAX_FLUIDNUM + fcount].x,
				buf.mf_vel_phrel[i*MAX_FLUIDNUM + fcount].y, buf.mf_vel_phrel[i*MAX_FLUIDNUM + fcount].z);

		//buf.mf_alpha[muloffseti+fcount] *= simData.mf_dt;
		//if(buf.mf_alpha[muloffseti+fcount]<-0.99)
		//{
		//	buf.mf_alpha[muloffseti+fcount] = -0.99;// * ((int)(buf.mf_alpha[muloffseti+fcount]>0)-(int)(buf.mf_alpha[muloffseti+fcount]<0));
		//}
		//buf.mf_alpha[muloffseti+fcount] += alpha_pre[fcount];
	}
}

//compute correction
__global__ void mfComputeCorrection( bufList buf, int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum)
		return;

	if (buf.MFtype[i] != 0)
		return;
	// Get search cell
	int nadj = (1*simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[ i ];
	if ( gc == GRID_UNDEF ) return;						// particle out-of-range
	gc -= nadj;

	uint muloffseti = i * MAX_FLUIDNUM;
	float sum, alphasum = 0, betasum = 0, alphaPercent, betaPercent;
	int flag;
	sum = 0.0f;

	for (uint fcount = 1; fcount < simData.mf_catnum; fcount++)
	{
		for (int l = 0; l<MAX_SOLIDNUM; ++l)
		{
			if (buf.mf_beta_next[muloffseti*MAX_SOLIDNUM + fcount*MAX_SOLIDNUM + l] < 0.01)
				buf.mf_beta_next[muloffseti*MAX_SOLIDNUM + fcount*MAX_SOLIDNUM + l] = 0;
			//if (buf.mf_beta_next[muloffseti*MAX_SOLIDNUM + fcount*MAX_SOLIDNUM + l] > 0.99)
			//	buf.mf_beta_next[muloffseti*MAX_SOLIDNUM + fcount*MAX_SOLIDNUM + l] = 1.0f;
			betasum += buf.mf_beta_next[muloffseti*MAX_SOLIDNUM + fcount*MAX_SOLIDNUM + l];
		}
		if (buf.mf_alpha_next[muloffseti + fcount] < 0.01)
			buf.mf_alpha_next[muloffseti + fcount] = 0.0f;

		//if (buf.mf_alpha_next[muloffseti + fcount] > 0.99)
		//	buf.mf_alpha_next[muloffseti + fcount] = 1.0f;
		alphasum += buf.mf_alpha_next[muloffseti + fcount];
	}
	sum = alphasum + betasum;

	flag = (sum>0.0f);
	sum = flag*sum + (1 - flag)*1.0f;
	sum = 1.0 / sum;
	alphaPercent = alphasum * sum;
	betaPercent = betasum * sum;

	if (betaPercent == 0)
		betasum = 1;
	else
		betasum = 1 / betasum;
	if (alphaPercent == 0)
		alphasum = 1;
	else
		alphasum = 1 / alphasum;
	//int cat = findMaxCat(alpha_pre, simData.mf_catnum, idx, idxlen);
	int maxcat = 3*MAX_SOLIDNUM + 3;

	for (uint fcount = 1; fcount < simData.mf_catnum; fcount++)
	{
		buf.mf_alpha_next[muloffseti + fcount] = (flag)*buf.mf_alpha_next[muloffseti + fcount] * alphaPercent * alphasum + (1 - flag)*(fcount == maxcat ? 1 : 0);
		for (int l = 0; l<MAX_SOLIDNUM; ++l)
			buf.mf_beta_next[muloffseti*MAX_SOLIDNUM + fcount*MAX_SOLIDNUM + l] =
			(flag)*buf.mf_beta_next[muloffseti*MAX_SOLIDNUM + fcount*MAX_SOLIDNUM + l] * betaPercent * betasum
			+ (1 - flag)*(fcount*MAX_SOLIDNUM + l == maxcat ? 1 : 0);
	}
	//sum = 0;
	//for (uint fcount = 1; fcount < simData.mf_catnum; fcount++)
	//{
	//	sum += buf.mf_alpha_next[muloffseti + fcount];
	//	for (int l = 0; l < MAX_SOLIDNUM; ++l)
	//		sum += buf.mf_beta_next[muloffseti*MAX_SOLIDNUM + fcount*MAX_SOLIDNUM + l];
	//}
	//if (abs(sum - 1) > 0.001)
	//	printf("correction lose function, sum is %f\n", sum);
}
__device__ float gamma(float q)
{
	if (q<2.0/3.0 && q>0)
		return 2.0/3.0;
	if (q>=2.0/3.0 && q<1)
		return 2*q-3.0/2.0*q*q;
	if (q>=1 && q<2)
		return (2-q)*(2-q)/2.0;
	return 0;
}

////compute force
//__device__ float3 contributeMfForce( int i, int muli, float3 ipos, float3 iveleval, float ipress, float idens, int cell, bufList buf, float* ialpha_pre, float ipressure_modify, float3* ivmk, float3* ivelxcor, float ivisc)
//{
//	float dsq, c;
//	register float d2 = simData.psimscale * simData.psimscale;	
//	register float r2 = simData.r2/d2;
//
//	float3 dist, vmr;		
//	float cmterm;
//	float pmterm, vmterm;
//	int j, mulj;	
//	float aveDenij,cx,xterm;
//	//float3 jvmk[MAX_FLUIDNUM];
//	//float jalpha_prek;
//
//	if ( buf.mgridcnt[cell] == 0 ) return make_float3(0,0,0);	
//
//	int cfirst = buf.mgridoff[ cell ];
//	int clast = cfirst + buf.mgridcnt[ cell ];
//
//	float3 force = make_float3(0,0,0);
//	//massi = buf.mf_restmass[i];
//	for ( int cndx = cfirst; cndx < clast; cndx++ ) {	
//		j = buf.mgrid[ cndx ];	
////		massj = buf.mf_restmass[j];
//		mulj = j * MAX_FLUIDNUM;
//		dist = ( ipos - buf.mpos[ j ] );		// dist in cm
//		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
//		dist *= simData.psimscale;
//		if ( dsq < r2 && dsq > 0) {	
//			cx = (r2-dsq)*d2;
//			dsq = sqrt(dsq*d2);
//			c = ( simData.psmoothradius - dsq ); 
//			cmterm = simData.spikykern * c * c / dsq * buf.mf_restmass[j] * buf.mdensity[j];
//			//pressure
//			if (buf.misbound[j] != 1)
//			{
//				pmterm = -0.5f * cmterm * (ipress + ipressure_modify + buf.mpress[j] + buf.mf_pressure_modify[j])*idens;
//				//pmterm = -0.5f * cmterm * (ipress + buf.mpress[j])*idens;
//				force += pmterm * dist;
//				//viscosity
//				vmr = iveleval - buf.mveleval[j]; //This is different from that in contributeAlphaChange()
//				vmterm = cmterm * (ivisc+buf.mf_visc[j]) * idens;
//				force += vmterm * vmr;
//			}
//			else
//			{
//				pmterm = -0.5f * cmterm * (ipress + ipressure_modify + buf.mpress[j])*idens;
//				//pmterm = -0.5f * cmterm * (ipress + buf.mpress[j])*idens;
//				force += pmterm * dist*0.03;
//				//viscosity
//				vmr = iveleval - buf.mveleval[j]; //This is different from that in contributeAlphaChange()
//				vmterm = cmterm * (ivisc+buf.mf_visc[j]) * idens;
//				force += vmterm * vmr*0.03;
//			}
//			/*
//			else pmterm = -0.5f * cmterm * (ipress + ipressure_modify + buf.mpress[j])*idens/30.0;
//				if (buf.misbound[j] ==1)
//				vmterm/= 30.0;
//				*/
//			if (buf.misbound[j] != 1)
//				//T_dm
//				for(uint fcount = 0; fcount < simData.mf_catnum; fcount++)
//				{
//					float3 dtermj = cmterm * (buf.mf_vel_phrel[mulj+fcount].x * dist.x + buf.mf_vel_phrel[mulj+fcount].y * dist.y + buf.mf_vel_phrel[mulj+fcount].z * dist.z) * buf.mf_alpha_next[mulj+fcount] * buf.mf_vel_phrel[mulj+fcount];
//					float3 dtermi = cmterm * (ivmk[fcount].x * dist.x + ivmk[fcount].y * dist.y + ivmk[fcount].z * dist.z) * ialpha_pre[fcount] * ivmk[fcount];
//					force += (dtermj + dtermi) * simData.mf_dens[fcount] * idens;
//				}
//#ifndef _nXSPH
//			//XSPH correction
//			aveDenij = 2/(1/buf.mdensity[j]+1/idens);
//			xterm = cx*cx*cx*buf.mf_restmass[j]*aveDenij*simData.poly6kern*0.5; //0.5=epsilon
//			ivelxcor->x += -vmr.x * xterm;
//			ivelxcor->y += -vmr.y * xterm;
//			ivelxcor->z += -vmr.z * xterm;
//		}	
//#endif
//	}
//	return force;
//}

//advance particles
__device__ void mfChRebalance(int i, int muli, bufList buf, int firstReactor, int secondReactor, int product)
{
	float chGamma = 0.01;
	register float alpha1 = buf.mf_alpha[muli+firstReactor];
	register float alpha2 = buf.mf_alpha[muli+secondReactor];
	//register float alphap;
	register float massTrans1, massTrans2;
	//register float V0 = buf.mf_restmass[i] * buf.mdensity[i];
	register float Vp;
	register float rhop1 = simData.mf_dens[firstReactor];
	register float rhop2 = simData.mf_dens[secondReactor];
	register float rhopp = simData.mf_dens[product];
	register float deltaAlphaP;

	//chGamma *= (alpha1*alpha2);
	chGamma *= (alpha1+alpha2);
	if(chGamma == 0)return;
	if(chGamma > alpha1)chGamma = alpha1;
	if(chGamma > alpha2)chGamma = alpha2;

	massTrans1 = chGamma * rhop1;
	massTrans2 = chGamma * rhop2;

	deltaAlphaP = (massTrans1 + massTrans2) / rhopp;

	Vp = 1 + deltaAlphaP - 2 * chGamma;
	Vp = 1/Vp;
	buf.mf_alpha[muli+firstReactor] -= chGamma;
	buf.mf_alpha[muli+secondReactor] -= chGamma;
	buf.mf_alpha[muli+product] += deltaAlphaP;

	for(uint fcount = 0; fcount<simData.mf_catnum; fcount++)
	{
		buf.mf_alpha[muli+fcount] *= Vp;
	}
	
	buf.mf_restdensity[i] *= Vp;
}

//**** shadow functions *******

__global__ void mfComputeDriftVelVelLimit( bufList buf, int pnum ) 
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum)
		return ;
#ifdef NEW_BOUND
	if(buf.misbound[i]==1) 
		return;
#endif

	// Get search cell
	int nadj = (1*simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[ i ];
	if ( gc == GRID_UNDEF ) return;						// particle out-of-range
	gc -= nadj;
	register float relax_coef = simData.relax;					// temporary relax time related coefficient
	register float sigma = 0.001f;//0.001f;						//diffusion&tension coefficient
	register float cont, conts, contr;

	cont = simData.cont;
	conts = simData.cont1;
	contr = simData.cont2;

	register float3 accel = buf.mforce[i];				// final accel (g-a) of last step was stored in here cf. advance, 
	//register float massFrack[MAX_FLUIDNUM];
	register uint muloffseti = i * MAX_FLUIDNUM;
	register float invdens = 1.0/buf.mf_restdensity[i];
	register float dsum;
	register float vrx, vry, vrz;
	register float tdiff;
	register float3 ssum;

	register float alpha_pre[MAX_FLUIDNUM],mass_concen[MAX_FLUIDNUM];
	register float ipress = buf.mpress[ i ];
	register float3 ipos = buf.mpos[ i ];
	register float idens = buf.mdensity[ i ];
	register float3 driftVelterm[MAX_FLUIDNUM],alphaGradterm[MAX_FLUIDNUM];
	register float3 sterm[MAX_FLUIDNUM];

	//various viscosity
	relax_coef /= buf.mf_visc[i];

	//relax_coef *= (99*buf.mf_alpha_pre[i*MAX_FLUIDNUM+2]+1);
	//third term
	for(uint fcount = 0;fcount < simData.mf_catnum; fcount++)
	{
		//float temp = buf.mf_alpha[muloffseti+fcount];
		//buf.mf_alpha_pre[muloffseti+fcount] = temp;				//alpha->alpha_pre
		alpha_pre[fcount] = buf.mf_alpha_next[muloffseti+fcount];
		mass_concen[fcount] = alpha_pre[fcount]*simData.mf_dens[fcount]*invdens;
		driftVelterm[fcount] = make_float3(0,0,0);
		alphaGradterm[fcount] = make_float3(0,0,0);
	}

	for (int c=0; c < simData.gridAdjCnt; c++) {
		contributeDriftVel ( i, muloffseti, ipos, idens, ipress, gc + simData.gridAdj[c], buf, alpha_pre, mass_concen, driftVelterm, relax_coef, alphaGradterm);
	}

	for( uint fcount = 0; fcount < simData.mf_catnum; fcount++)
	{
		//buf.mf_vel_phrel[muloffseti+fcount] = cont * contr * driftVelterm[fcount];
		float3 vel = cont * contr * driftVelterm[fcount];
		float speed = vel.x*vel.x + vel.y*vel.y + vel.z*vel.z;
		if ( speed > simData.VL2 ) {
			vel *= simData.VL / sqrt(speed);
		}
		buf.mf_vel_phrel[muloffseti+fcount] = vel;

	}

	//first term & second term
	dsum = 0;
	ssum = make_float3(0,0,0);
	for( uint fcount = 0; fcount < simData.mf_catnum; fcount++)
	{
		//float temp = buf.mf_alpha[muloffseti+fcount];
		//dsum += temp * simData.mf_dens[fcount] * simData.mf_dens[fcount] * invdens;
		//buf.mf_alpha_pre[muloffseti+fcount] = temp;				//alpha->alpha_pre

		float temp = buf.mf_alpha_next[muloffseti+fcount];
		dsum += temp * simData.mf_dens[fcount] * simData.mf_dens[fcount] * invdens;

		if(temp>0.0001)
			//sterm[fcount] = buf.mf_alphagrad[muloffseti+fcount]/temp;
			sterm[fcount] = alphaGradterm[fcount]/temp;
		else
			sterm[fcount] = make_float3(0,0,0);
		ssum += sterm[fcount] * temp * simData.mf_dens[fcount] * invdens;
	}
	for( uint fcount = 0; fcount < simData.mf_catnum; fcount++)
	{
		tdiff = simData.mf_dens[fcount]-dsum;
		tdiff *= relax_coef;
		vrx = accel.x * tdiff;
		vry = accel.y * tdiff;
		vrz = accel.z * tdiff;
		buf.mf_vel_phrel[muloffseti+fcount] += make_float3(vrx,vry,vrz);

		buf.mf_vel_phrel[muloffseti+fcount] -= cont * conts * sigma * (sterm[fcount]-ssum);
	}
}


//***** End Shadow Functions *******

// **********   Project-u  Functions *********
//__device__ float3 contributeForce_projectu(int i, int muli, float3 ipos, float3 iveleval, float ipress, float idens, int cell, bufList buf, float* ialpha_pre, float ipressure_modify, float3* ivmk, float3* ivelxcor, float ivisc)
//{
//	//Force here represents the acceleration
//	float dsq, c;
//	register float d2 = simData.psimscale * simData.psimscale;	
//	register float r2 = simData.r2/d2;
//
//	float3 dist, vmr ;		
//	float cmterm,cmterm1;
////	float massj;
//	float pmterm, vmterm;
////	float q;
//	int j, mulj;	
//	float aveDenij,cx,xterm;
//
//	if ( buf.mgridcnt[cell] == 0 ) return make_float3(0,0,0);	
//
//	int cfirst = buf.mgridoff[ cell ];
//	int clast = cfirst + buf.mgridcnt[ cell ];
//
//	float3 force = make_float3(0,0,0);
//	//massi = buf.mf_restmass[i];
//	
//	for ( int cndx = cfirst; cndx < clast; cndx++ ) 
//	{	
//		j = buf.mgrid[ cndx ];	
//		
//		//massj = buf.mf_restmass[j];
//		mulj = j * MAX_FLUIDNUM;
//		dist = ( ipos - buf.mpos[ j ] );		// dist in cm
//		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
//		dist *= simData.psimscale;
//
//		if ( dsq < r2 && dsq > 0) {	
//			cx = (r2-dsq)*d2;
//			dsq = sqrt(dsq*d2);
//			c = ( simData.psmoothradius - dsq ); 
//
//			cmterm1 = simData.spikykern * c * c / dsq;
//			cmterm = simData.spikykern * c * c / dsq * buf.mf_restmass[j] * buf.mdensity[j];
//			//pressure
//#ifdef NEW_BOUND
//			if (buf.misbound[j] != 1) //force between fluid and solid, force within fluid
//			{
//				
//				if( buf.MFtype[j]==0)
//					pmterm = -0.5f * cmterm * (ipress  + buf.mpress[j] + buf.mf_pressure_modify[j] )*idens;
//				else
//					pmterm = -0.5f * cmterm * (ipress  + buf.mpress[j])*idens;
//
//				if(buf.MFtype[i]==0 && buf.MFtype[j]==1 && buf.mpress[j]<0)
//					pmterm = -0.5f * cmterm * (ipress + 0)*idens;
//				
//				//pmterm = -0.5f * cmterm * (ipress + buf.mpress[j])*idens;
//				//if( (buf.MFtype[i]==0 && buf.MFtype[j]==0))
//				//	force += pmterm * dist;
//				////固体与液体之间的排斥力，暂时注释掉
//				//if(! (buf.MFtype[i]==1 && buf.MFtype[j]==1)){
//				//	force += pmterm * dist;
//				//}
//				if(buf.MFtype[i] == 0 && buf.MFtype[j] == 0)
//				{
//					force += pmterm * dist;
//				}
//
//			}
//			else if(buf.MFtype[i]==0) //force from boundary particles to fluid particles
//			{
//				//pmterm = -0.5f * cmterm * (ipress + ipressure_modify + buf.mpress[j])*idens;
//				//pmterm = -0.5f * cmterm * (ipress + buf.mpress[j])*idens;
//				
//				////注释掉
//				////pressure
//				//pmterm = - cmterm1 * buf.mf_restdensity[i] * buf.mf_restmass[j] /buf.mf_restdensity[j] *ipress *buf.mdensity[i]*buf.mdensity[i];
//				//force += pmterm * dist * simData.omega;
//
//				////viscosity
//				//vmr = iveleval - buf.mveleval[j]; //This is different from that in contributeAlphaChange()
//				//float pi_ij = vmr.x*dist.x + vmr.y*dist.y + vmr.z*dist.z;
//				//if(pi_ij < 0){
//				//	pi_ij = pi_ij / (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z + r2 * 0.01);
//				//	pi_ij = pi_ij * 2 * simData.psmoothradius * (ivisc + buf.mf_visc[j]) * idens /2;
//				//	pi_ij = - cmterm1 * buf.mf_restdensity[i] * buf.mf_restmass[j]/buf.mf_restdensity[j] * pi_ij;
//				//	force += pi_ij * dist * simData.visc_factor;
//				//	
//				//}
//				
//				//vmterm = cmterm * (ivisc+buf.mf_visc[j]) * idens;
//				//force += vmterm * vmr*0.03;
//			}
//			else{ //force from boundary particles to deformable/rigid particles
//				/*
//				pmterm = -0.5f * cmterm * (ipress + buf.mpress[j])*idens;
//				force += pmterm * dist*0.03;
//				vmr = iveleval - buf.mveleval[j];
//				vmterm = cmterm * (ivisc+buf.mf_visc[j]) * idens;
//				force += vmterm * vmr*0.03;*/
//
//				//pressure
//				pmterm = - cmterm1 * buf.mf_restdensity[i] * buf.mf_restmass[j] / buf.mf_restdensity[j] * (ipress) *buf.mdensity[i]*buf.mdensity[i];
//				force += pmterm * dist * simData.omega;
//				
//				//viscosity
//				vmr = iveleval - buf.mveleval[j]; //This is different from that in contributeAlphaChange()
//				float pi_ij = vmr.x*dist.x + vmr.y*dist.y + vmr.z*dist.z;
//				if(pi_ij < 0){
//					pi_ij = pi_ij / (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z + r2 * 0.01);
//					pi_ij = pi_ij * 2 * simData.psmoothradius * (ivisc + buf.mf_visc[j]) * idens /2;
//					pi_ij = - cmterm1 * buf.mf_restdensity[i] * buf.mf_restmass[j]/buf.mf_restdensity[j] * pi_ij;
//					force += pi_ij * dist * simData.visc_factor;
//				}
//			}
//
//			if (buf.misbound[j] != 1)
//				//T_dm
//				for(uint fcount = 0; fcount < simData.mf_catnum; fcount++)
//				{
//					float3 dtermj = cmterm * (buf.mf_vel_phrel[mulj+fcount].x * dist.x + buf.mf_vel_phrel[mulj+fcount].y * dist.y + buf.mf_vel_phrel[mulj+fcount].z * dist.z) * buf.mf_alpha_next[mulj+fcount] * buf.mf_vel_phrel[mulj+fcount];
//					float3 dtermi = cmterm * (ivmk[fcount].x * dist.x + ivmk[fcount].y * dist.y + ivmk[fcount].z * dist.z) * ialpha_pre[fcount] * ivmk[fcount];
//					force += (dtermj + dtermi) * simData.mf_dens[fcount] * idens;
//				}
//
//#else
//			pmterm = -0.5f * cmterm * (ipress + ipressure_modify + buf.mpress[j] + buf.mf_pressure_modify[j])*idens;
//			//pmterm = -0.5f * cmterm * (ipress + buf.mpress[j])*idens;
//			force += pmterm * dist;
//			//viscosity
//			vmr = iveleval - buf.mveleval[j]; //This is different from that in contributeAlphaChange()
//			vmterm = cmterm * (ivisc+buf.mf_visc[j]) * idens;
//			force += vmterm * vmr;
//			for(uint fcount = 0; fcount < simData.mf_catnum; fcount++)
//			{
//				float3 dtermj = cmterm * (buf.mf_vel_phrel[mulj+fcount].x * dist.x + buf.mf_vel_phrel[mulj+fcount].y * dist.y + buf.mf_vel_phrel[mulj+fcount].z * dist.z) * buf.mf_alpha_pre[mulj+fcount] * buf.mf_vel_phrel[mulj+fcount];
//				float3 dtermi = cmterm * (ivmk[fcount].x * dist.x + ivmk[fcount].y * dist.y + ivmk[fcount].z * dist.z) * ialpha_pre[fcount] * ivmk[fcount];
//				force += (dtermj + dtermi) * simData.mf_dens[fcount] * idens;
//			}
//
//#endif
//#ifndef _nXSPH
//			//XSPH correction
//			aveDenij = 2/(1/buf.mdensity[j]+1/idens);
//			xterm = cx*cx*cx*buf.mf_restmass[j]*aveDenij*simData.poly6kern*0.5; //0.5=epsilon
//			ivelxcor->x += -vmr.x * xterm;
//			ivelxcor->y += -vmr.y * xterm;
//			ivelxcor->z += -vmr.z * xterm;
//		}	
//#endif
//
//	}
//	return force;
//}
//__global__ void ComputeForce_projectu ( bufList buf, int pnum)
//{			
//	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
//	if ( i >= pnum)
//		return;
//#ifdef NEW_BOUND
//	if(buf.misbound[i]==1)
//		return;
//#endif
//	// Get search cell
//	int nadj = (1*simData.gridRes.z + 1)*simData.gridRes.x + 1;
//	uint gc = buf.mgcell[ i ];
//	if ( gc == GRID_UNDEF ) return;						// particle out-of-range
//	gc -= nadj;
//
//	register uint muloffseti = i * MAX_FLUIDNUM;
//	register float3 ipos = buf.mpos[ i ];
//	register float3 iveleval = buf.mveleval[ i ];
//	
//	register float idens = buf.mdensity[ i ];
//	register float alpha_pre[MAX_FLUIDNUM];
//	register float3 ivmk[MAX_FLUIDNUM];
//	register float pressure_modify = buf.mf_pressure_modify[i];
//	register float3 *ivelxcor = buf.mf_velxcor+i;
//	register float ivisc = buf.mf_visc[i];
//
//	register float ipress;
//	if(buf.MFtype[i]==0)
//		ipress = buf.mpress[i] + buf.mf_pressure_modify[i];
//	else
//		ipress = buf.mpress[i];
//
//	register float3 force = make_float3(0,0,0);	
//	*ivelxcor = make_float3(0,0,0);
//
//	for(uint fcount = 0;fcount < simData.mf_catnum; fcount++)
//	{
//		alpha_pre[fcount] = buf.mf_alpha_next[muloffseti+fcount];
//		ivmk[fcount] = buf.mf_vel_phrel[muloffseti+fcount];
//	}
//
//	for (int c=0; c < simData.gridAdjCnt; c++) {
//		force += contributeForce_projectu (i, muloffseti, ipos, iveleval, ipress, idens, gc + simData.gridAdj[c], buf, alpha_pre, pressure_modify, ivmk, ivelxcor, ivisc);
//	}
//	/*if (buf.MFtype[i] == 0 && i % 1000 == 0)
//		printf("fluid force is (%f,%f,%f)\n", force.x, force.y, force.z);*/
//	//if (buf.MFtype[i] == 1 && buf.elasticID[i] == 6)
//	//	printf("fluid force is (%f,%f,%f)\n", force.x, force.y, force.z);
//	buf.mforce[ i ] = force;
//}

//__device__ void contributeVelocityGradient(float* result, int i, float3 ipos, float3 iveleval, int cell, bufList buf)
//{
//	float dsq, c;
//	register float d2 = simData.psimscale * simData.psimscale;	
//	register float r2 = simData.r2/d2;
//
//	float3 dist, jveleval;		
//	float cmterm;
////	float massj,massi;
//	
////	float q;
//	int j;	
////	float aveDenij,cx,xterm;
//
//	if ( buf.mgridcnt[cell] == 0 ) return;	
//
//	int cfirst = buf.mgridoff[ cell ];
//	int clast = cfirst + buf.mgridcnt[ cell ];
//
//	//massi = buf.mf_restmass[i];
//	for ( int cndx = cfirst; cndx < clast; cndx++ ) 
//	{	
//		j = buf.mgrid[ cndx ];	
//		if( buf.MFtype[j] != 2)
//			continue;
//
//		//massj = buf.mf_restmass[j];
//		//jveleval = buf.mveleval[j]*buf.mdensity[j]*buf.mdensity[j] + iveleval*buf.mdensity[i]*buf.mdensity[i];
//		jveleval = buf.mveleval[j]-iveleval;
//
//		dist = ( ipos - buf.mpos[ j ] );		// dist in cm
//		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
//		dist *= simData.psimscale;
//		
//		if ( dsq < r2 && dsq > 0) {	
//			dsq = sqrt(dsq * d2);
//			c = ( simData.psmoothradius - dsq ); 
//			cmterm = simData.spikykern * c * c / dsq * buf.mf_restmass[j] * buf.mdensity[j];
//			//cmterm = simData.spikykern * c * c / dsq;
//			jveleval = jveleval * cmterm;
//			result[0] += jveleval.x * dist.x;	result[1] += jveleval.x * dist.y;	result[2] += jveleval.x * dist.z;
//			result[3] += jveleval.y * dist.x;	result[4] += jveleval.y * dist.y;	result[5] += jveleval.y * dist.z;
//			result[6] += jveleval.z * dist.x;	result[7] += jveleval.z * dist.y;	result[8] += jveleval.z * dist.z;
//		}
//	}
//}

__device__ void print9(char* string,float* buf){

	printf("%s\n%f %f %f\n%f %f %f\n%f %f %f\n",string,buf[0],buf[1],buf[2],
	buf[3],buf[4],buf[5],buf[6],buf[7],buf[8]);
	return;
}



__device__ float3 getBoundForce(int i,bufList buf, float3 force, float time){

	register float3 accel, norm;
	register float diff, adj, speed;
	register float3 pos = buf.mpos[i];
	register float3 veval = buf.mveleval[i];
	accel = force;
//	if (buf.MFtype[i] == 1) 
//	{
//		// Boundaries
//		// Y-axis
//		diff = simData.pradius - (pos.y - (simData.pboundmin.y + (pos.x - simData.pboundmin.x)*simData.pground_slope)) * simData.psimscale;
//		//	if (diff>simData.pradius) diff += simData.pradius*1000;
//		if (diff > EPSILON) {
//			norm = make_float3(-simData.pground_slope, 1.0 - simData.pground_slope, 0);
//			adj = simData.pextstiff * diff - simData.pdamp * dot(norm, veval);
//			norm *= adj; accel += norm;//*scale_dens;
//
//			//float3 veldamp=make_float3(veval.x, 0, veval.z);
//			//buf.mveleval[i] -= veldamp * simData.omega;
//			//veldamp=make_float3(vel.x, 0, vel.z);
//			//buf.mvel[i] -= veldamp * simData.omega;
//		}
//
//		diff = simData.pradius - (simData.pboundmax.y - pos.y)*simData.psimscale;
//		//	if (diff>simData.pradius) diff += simData.pradius*1000;
//		if (diff > EPSILON) {
//			norm = make_float3(0, -1, 0);
//			adj = simData.pextstiff * diff - simData.pdamp * dot(norm, veval);
//			norm *= adj; accel += norm;//*scale_dens;
//		}
//
//#ifdef _xzsoftmargin
//		// X-axis
//		diff = simData.pradius - (pos.x - (simData.pboundmin.x + (sin(time*simData.pforce_freq) + 1)*0.5 * simData.pforce_min))*simData.psimscale;
//		//	if (diff>simData.pradius) diff += simData.pradius*1000;
//		if (diff > EPSILON) {
//			norm = make_float3(1, 0, 0);
//			adj = (simData.pforce_min + 1) * simData.pextstiff * diff - simData.pdamp * dot(norm, veval);
//			norm *= adj; accel += norm;//*scale_dens;
//		}
//		diff = simData.pradius - ((simData.pboundmax.x - (sin(time*simData.pforce_freq) + 1)*0.5*simData.pforce_max) - pos.x)*simData.psimscale;
//		//	if (diff>simData.pradius) diff += simData.pradius*1000;
//		if (diff > EPSILON) {
//			norm = make_float3(-1, 0, 0);
//			adj = (simData.pforce_max + 1) * simData.pextstiff * diff - simData.pdamp * dot(norm, veval);
//			norm *= adj; accel += norm;//*scale_dens;
//		}
//
//		// Z-axis
//		diff = simData.pradius - (pos.z - simData.pboundmin.z) * simData.psimscale;
//		//	if (diff>simData.pradius) diff += simData.pradius*1000;
//		if (diff > EPSILON) {
//			norm = make_float3(0, 0, 1);
//			adj = simData.pextstiff * diff - simData.pdamp * dot(norm, veval);
//			norm *= adj; accel += norm;//*scale_dens;
//		}
//		diff = simData.pradius - (simData.pboundmax.z - pos.z)*simData.psimscale;
//		//	if (diff>simData.pradius) diff += simData.pradius*1000;
//		if (diff > EPSILON) {
//			norm = make_float3(0, 0, -1);
//			adj = simData.pextstiff * diff - simData.pdamp * dot(norm, veval);
//			norm *= adj; accel += norm;//*scale_dens;
//		}
//#endif 
//	}
	//if (i % 500 == 0&&buf.misbound[i]!=1)
	//	printf("particle %d's accel is (%f,%f,%f)\n", i, accel.x, accel.y, accel.z);
	// Accel Limit
	/*speed = accel.x*accel.x + accel.y*accel.y + accel.z*accel.z;
	if ( speed > simData.AL2 ) {
		accel *= simData.AL / sqrt(speed);
	}*/
	// Gravity
	//accel += simData.pgravity;
	return accel;
}

//__global__ void AddSPHtensorForce( bufList buf, int pnum, float time)
//{
//	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
//	if ( i >= pnum) return;
//	//if(buf.MFtype[i] != 1)
//	//	return;
//
//	// Get search cell
//	int nadj = (1*simData.gridRes.z + 1)*simData.gridRes.x + 1;
//	uint gc = buf.mgcell[ i ];
//	if ( gc == GRID_UNDEF ) return;						// particle out-of-range
//	gc -= nadj;
//
////	register float3 ipos = buf.mpos[ i ];
////	float *itensor = buf.MFtemptensor + i*9;
////	float3 tensorForce =  make_float3(0,0,0);
////	
////	
//	
////	/*if(i%1000==0&&buf.misbound[i]!=1)
////		printf("%d tensorforce: %f %f %f\n",i, tensorForce.x, tensorForce.y, tensorForce.z);
////*/
////	buf.mforce[i] = buf.mforce[i] + tensorForce;
////	if (buf.MFtype[i] == 1 && buf.elasticID[i] == 1600)
////		printf("tensor force is (%f,%f,%f)\n", tensorForce.x, tensorForce.y, tensorForce.z);
//	//Get Other force!
//	buf.maccel[i] = buf.mforce[i];
//	//if (buf.MFtype[i] == 1 && (buf.elasticID[i] == 6 || buf.elasticID[i] == 31))
//	//	printf("final force %d's is %f,%f,%f\n", buf.elasticID[i], buf.mvel[i].x, buf.mvel[i].y, buf.mvel[i].z);
//	buf.mforce[i] = make_float3(0,0,0); 
//}

//**********************  end project-u    ************************
void floatup_cuda(int mode){
	fcuda.gravityfree = mode;
	checkCudaErrors ( cudaMemcpyToSymbol ( simData, &fcuda, sizeof(FluidParams) ) );
	return;
}

__global__ void updatePosition(float time, bufList buf, int pnum){
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;
	
	if ( buf.mgcell[i] == GRID_UNDEF ) {
		buf.mpos[i] = make_float3(-1000,-1000,-1000);
		buf.maccel[i] = make_float3(0,0,0);
		return;
	}
	
	// Get particle vars
	register float3 accel, norm;
	register float diff, adj, speed;
	register float3 pos = buf.mpos[i];
	register float3 veval = buf.mveleval[i];
	float3 vel = buf.maccel[i];
	register float newdens,newvisc, newmass;
	// Leapfrog integration						
	accel = buf.maccel[i];
	float beta[MAX_FLUIDNUM];
	if (buf.misbound[i] != 1)
	{
		//float3 vnext = accel*simData.mf_dt + vel;				// v(t+1/2) = v(t-1/2) + a(t) dt		
		//float3 tmpdeltaPos = (vnext + buf.mf_velxcor[i]) * (simData.mf_dt/simData.psimscale);
		//float3 tmpPos = buf.mpos[i] + tmpdeltaPos;

		buf.mforce[i] = accel; //use mvel to restore the first acceleration

		float3 dPos = (buf.mveleval[i]*simData.mf_dt + 0.5* accel* simData.mf_dt* simData.mf_dt)/simData.psimscale;
		buf.mpos[i] = buf.mpos[i] + dPos;

		//Color Setting
		//buf.mclr[i] = COLORA(buf.mf_alpha[i*MAX_FLUIDNUM+2],buf.mf_alpha[i*MAX_FLUIDNUM+1],buf.mf_alpha[i*MAX_FLUIDNUM+0],1);
		//if(buf.MFtype[i]==0)
		//	buf.mclr[i] = COLORA(1,1,1,1);
		//else
		if (buf.MFtype[i] == 2 || (_example == 2&&buf.MFtype[i] >= 2))
		{
			//buf.mclr[i] = COLORA(1, 1, 0, 0.6);
			int index = buf.elasticID[i];
			for (int k = 1; k < MAX_FLUIDNUM; ++k)
				beta[k] = buf.mf_beta[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k * MAX_SOLIDNUM + buf.MFtype[i] - 2];
			if (_example == 2) 
			{
				if (buf.MFtype[i] == 5)
					buf.mclr[i] =
					COLORA(1 / (1 + beta[2] + beta[3]), 1 / (1 + beta[1] + beta[3]), beta[3] / (1 + beta[3]), !simData.HideSolid);
						//COLORA(1 / (1+beta[2] + beta[3]), 1 / (1+beta[1] + beta[3]), beta[3]/(1+beta[3]), !simData.HideSolid);
				else
				{
					buf.mclr[i] =
						COLORA(0, 1, 0, !simData.HideSolid);
				}
			}
			else
				buf.mclr[i] =
					COLORA(1 - (beta[2] + beta[3]), 1 - (beta[1] + beta[3]), 1 - (beta[1] + beta[2]), !simData.HideSolid);
		}
		else
		{
			buf.mclr[i] = COLORA(buf.mf_alpha[i*MAX_FLUIDNUM + 1],buf.mf_alpha[i*MAX_FLUIDNUM + 2],
					buf.mf_alpha[i*MAX_FLUIDNUM + 3],!simData.HideFluid*0.55*
					(buf.mf_alpha[i*MAX_FLUIDNUM + 2]+ buf.mf_alpha[i*MAX_FLUIDNUM + 1]+
						buf.mf_alpha[i*MAX_FLUIDNUM + 3]));
			//buf.mclr[i] = COLORA(buf.mf_alpha[i*MAX_FLUIDNUM + 2] + buf.mf_beta[i*MAX_FLUIDNUM + 2], 
			//	buf.mf_alpha[i*MAX_FLUIDNUM + 1] + buf.mf_beta[i*MAX_FLUIDNUM + 1],
			//	buf.mf_alpha[i*MAX_FLUIDNUM + 0] + buf.mf_beta[i*MAX_FLUIDNUM + 0], !simData.HideFluid);
			//buf.mclr[i] = COLORA(buf.mf_alpha[i*MAX_FLUIDNUM + 2],
			//	buf.mf_alpha[i*MAX_FLUIDNUM + 1],
			//	 buf.mf_alpha[i*MAX_FLUIDNUM + 0], !simData.HideFluid*(buf.mf_alpha[i*MAX_FLUIDNUM + 1]+ buf.mf_alpha[i*MAX_FLUIDNUM + 2]));
		}
			//buf.mclr[i] = COLORA(buf.mf_alpha[i*MAX_FLUIDNUM + 2], buf.mf_alpha[i*MAX_FLUIDNUM + 1], buf.mf_alpha[i*MAX_FLUIDNUM + 0], 0);
	}
	else if (buf.misbound[i] == 1)
	{
		buf.mveleval[i] = make_float3(0,0,0);		// v(t+1) = [v(t-1/2) + v(t+1/2)] * 0.5			
		buf.maccel[i] = make_float3(0,0,0);
		buf.mforce[i] = make_float3(0,0,0);
		if (buf.MFtype[i] > 2) 
		{
			for (int k = 1; k < MAX_FLUIDNUM; ++k)
				beta[k] = buf.mf_beta[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k * MAX_SOLIDNUM + buf.MFtype[i] - 2];
			float sum = beta[1] + beta[2] + beta[3] + 1;
			buf.mclr[i] =
				COLORA(1 - (beta[2] + beta[3]), 1 - (beta[1] + beta[3]), 1 - (beta[1] + beta[2]), !simData.HideRigid);
			//buf.mclr[i] = COLORA((sqrt(beta[1]))/sum, (sqrt(beta[2]))/sum, (sqrt(beta[3]))/sum, !simData.HideRigid*(beta[1]+beta[2]+beta[3]));
			//buf.mclr[i] = COLORA((1+beta[1])/sum, (1+beta[2])/sum, (1+beta[3])/sum, !simData.HideRigid);
			//buf.mclr[i] = COLORA(1, 1, 1, !simData.HideBound);
		}
		else
		{
			buf.mclr[i] = COLORA(1, 1, 1, !simData.HideBound);
		}
	}
	buf.mforce[i] = make_float3(0, 0, 0);
}

__global__ void updateVelocity(float time, bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;
	//if (buf.MFtype[i] == 3)return;
	if ( buf.mgcell[i] == GRID_UNDEF ) {
		buf.mpos[i] = make_float3(-1000,-1000,-1000);
		buf.maccel[i] = make_float3(0,0,0);
		return;
	}
	// Get particle vars
	register float3 accel, accel1, accel2;
	register float speed;

	// Leapfrog integration						
	accel = buf.maccel[i];
	if (isnan(dot(accel, accel)))
		printf("particle %d's type is %d, accel is nan\n",
			i, buf.MFtype[i]);
	//if (buf.MFtype[i] == 0 && i % 10000 == 0)
	//	printf("particle %d's mixture vel is (%f,%f,%f), fluid vel is (%f,%f,%f)\n",
	//		i, buf.mveleval[i].x, buf.mveleval[i].y, buf.mveleval[i].z,
	//		buf.fluidVel[i*MAX_FLUIDNUM + 1].x, buf.fluidVel[i*MAX_FLUIDNUM + 1].y,
	//		buf.fluidVel[i*MAX_FLUIDNUM + 1].z);
	speed = accel.x*accel.x + accel.y*accel.y + accel.z*accel.z;
	if (speed > simData.AL2) {
		accel *= simData.AL / sqrt(speed);
	}
	
	accel += simData.pgravity;
	buf.maccel[i] = accel;
	////int index;
	//if(simData.example == 1 || simData.example == 2)
	//	if (buf.MFtype[i] == 1)
	//	{
	//		int index = buf.elasticID[i];
	//		if(buf.frame[index] > 1200 && buf.frame[index] < 1600)
	//				accel -= 3 * simData.pgravity;
	//		if (buf.frame[index] == 1600)
	//		{
	//			buf.mveleval[i] = make_float3(0, 0, 0);
	//			accel -= simData.pgravity;
	//		}
	//		if (buf.frame[index] >= 1600)
	//		{
	//			accel -= simData.pgravity;
	//			if (buf.isSurface[index] && buf.frame[index] <= 2000 && buf.frame[index] >= 1800 && simData.example == 1)
	//				accel += -300 * buf.normal[index];
	//		}
	//	}

	if (buf.misbound[i] != 1)
	{
		buf.mveleval[i] = buf.mveleval[i] + simData.mf_dt*accel;
		{
			//buf.mveleval[i] += (1-buf.fluidPercent[i])*simData.mf_dt*buf.poroForce[i];
			float vm = dot(buf.mveleval[i], buf.mveleval[i]);// .x*buf.mveleval[i].x + buf.mveleval[i].y*buf.mveleval[i].y + buf.mveleval[i].z*buf.mveleval[i].z;
			vm = sqrt(vm);
			if (vm > simData.VL)
			{
				buf.mveleval[i] *= simData.VL / vm;
			}
		}
	}
	else if (buf.misbound[i] == 1)
	{
		buf.mveleval[i] = make_float3(0,0,0);		// v(t+1) = [v(t-1/2) + v(t+1/2)] * 0.5			
		buf.maccel[i] = make_float3(0,0,0);
		buf.mforce[i] = make_float3(0,0,0);
		//buf.mclr[i] = COLORA(1,1,1,0.8);
	}
	//buf.vel_mid[i] = buf.mveleval[i];
}
__global__ void computeMidVel(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;
	//if (buf.MFtype[i] == 3)return;
	if (buf.mgcell[i] == GRID_UNDEF) {
		buf.mpos[i] = make_float3(-1000, -1000, -1000);
		buf.maccel[i] = make_float3(0, 0, 0);
		return;
	}

	// Get particle vars
	register float3 accel, norm, pos = buf.mpos[i];
	register float speed;
	//buf.vel_mid[i] = buf.mveleval[i];
	//if (dot(buf.vel_mid[i], buf.vel_mid[i])!=dot(buf.mveleval[i], buf.mveleval[i]))
	//	printf("particle %d's type is %d, vel is (%f,%f,%f), vel_mid is (%f,%f,%f)\n",
	//		i, buf.MFtype[i], buf.mveleval[i].x, buf.mveleval[i].y, buf.mveleval[i].z,
	//		buf.vel_mid[i].x, buf.vel_mid[i].y, buf.vel_mid[i].z);
	//	float scale_dens = 1000.0/buf.mf_restdensity[i];

	accel = buf.maccel[i];

	speed = accel.x*accel.x + accel.y*accel.y + accel.z*accel.z;
	if (speed > simData.AL2) {
		//printf("other accel is (%f,%f,%f), vel is (%f,%f,%f)\n",
		//	accel.x, accel.y, accel.z, buf.mveleval[i].x, buf.mveleval[i].y, buf.mveleval[i].z);
		accel *= simData.AL / sqrt(speed);
	}
	buf.mforce[i] = accel;
	buf.fluidForce[i] = accel;
	if (buf.misbound[i] != 1)
	{
		buf.vel_mid[i] = buf.mveleval[i] + simData.mf_dt*accel;
	}
	else 
	{
		buf.mveleval[i] = make_float3(0, 0, 0);		// v(t+1) = [v(t-1/2) + v(t+1/2)] * 0.5			
		buf.maccel[i] = make_float3(0, 0, 0);
		buf.mforce[i] = make_float3(0, 0, 0);
		buf.vel_mid[i] = make_float3(0, 0, 0);
		//buf.mclr[i] = COLORA(1,1,1,0.8);
	}
	//buf.maccel[i] = make_float3(0, 0, 0);
	//buf.mforce[i] = make_float3(0, 0, 0);
}

void LeapFrogIntegration(float time){
	updateVelocity<<<fcuda.numBlocks, fcuda.numThreads>>>(time, fbuf, fcuda.pnum);
	cudaThreadSynchronize();

	updatePosition << <fcuda.numBlocks, fcuda.numThreads >> >(time, fbuf, fcuda.pnum);
	cudaThreadSynchronize();
}

//****An Implicit SPH Formulation for Incompressible Linearly Elastic Solids*************
__global__ void ComputeMap(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;
	if (_example != 2)
	{
		if (buf.MFtype[i] != 2) return;
	}
	else
		if (buf.MFtype[i] < 2)
			return;
	int gc = buf.mgcell[i];
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	
	if (gc == GRID_UNDEF) {
		buf.mpos[i] = make_float3(-1000, -1000, -1000);
		buf.maccel[i] = make_float3(0, 0, 0);
		return;
	}
	int elasticIndex = buf.elasticID[i];
	int j = 0;
	
	for(int l=0;l<buf.neighborNum[elasticIndex];++l)
	{
		j = buf.neighborID[elasticIndex * simData.maxNeighborNum + l];
		for(int k=0;k<buf.neighborNum[j];++k)
			if(elasticIndex == buf.neighborID[j*simData.maxNeighborNum +k])
			{
				//if (elasticIndex == 1600) 
				//{
				//	printf("elastic id: %d,neighborID:%d\n", buf.elasticID[i], j);
				//}
				buf.neighborIndex[elasticIndex * simData.maxNeighborNum + l] = k;
				break;
			}
	}
	//if (elasticIndex == 1600)
	//	printf("particle %d's elasticID is %d\n", i, buf.elasticID[i]);
}
//compute only once
__global__ void ComputeCorrectL(bufList buf,int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;
	if (_example != 2)
	{
		if (buf.MFtype[i] != 2) return;
	}
	else
		if (buf.MFtype[i] < 2)
			return;
	int gc = buf.mgcell[i];
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	//if(i%100 == 0)
	//printf("particle %d's elasticID is %d\n", i, buf.elasticID[i]);
	if (gc == GRID_UNDEF) {
		buf.mpos[i] = make_float3(-1000, -1000, -1000);
		buf.maccel[i] = make_float3(0, 0, 0);
		return;
	}
	gc -= nadj;
	float correctL[9];
	for (int l = 0; l < 9; ++l)
		correctL[l] = 0;
	int index = 0;
	int jndex, j;
	int elasticIndex = buf.elasticID[i];
	float dsq, c;
	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2 / d2;
	float pmterm;
	float3 dist, vmr;
	//if(elasticIndex == 1600)
	//printf("particle %d's elasticIndex is %d\n", i, elasticIndex);
	//if (elasticIndex >= simData.numElasticPoints)
	//printf("elasticIndex = %d and limit %d\n", elasticIndex, simData.numElasticPoints);
	//fbuf.elasticID[elasticIndex] = elasticIndex;
	//buf.initialVolume[elasticIndex] = buf.mf_restmass[i] * buf.mdensity[i];
	
	for (int l = 0; l < buf.neighborNum[elasticIndex]; l++)
	{
		jndex = buf.neighborID[elasticIndex * simData.maxNeighborNum + l];
		j = buf.particleID[jndex];
		dist = (buf.mpos[i] - buf.mpos[j]);		// dist in cm
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
		dsq = sqrt(dsq*d2);
		c = simData.psmoothradius - dsq;
		dist *= simData.psimscale;
		
		pmterm = buf.initialVolume[jndex] * simData.spikykern * c * c / dsq;
		//pmterm = buf.initialVolume[jndex] * simData.spikykern * c * c;//v_j 0
		correctL[0] += -pmterm * dist.x*dist.x; correctL[1] += -pmterm * dist.x*dist.y; correctL[2] += -pmterm * dist.x*dist.z;
		correctL[3] += -pmterm * dist.y*dist.x; correctL[4] += -pmterm * dist.y*dist.y; correctL[5] += -pmterm * dist.y*dist.z;
		correctL[6] += -pmterm * dist.z*dist.x; correctL[7] += -pmterm * dist.z*dist.y; correctL[8] += -pmterm * dist.z*dist.z;
	}

	if (det(correctL) != 0) {
		/*if (i % 1000 == 0)
			printf("particle %d's L is (%f,%f,%f)(%f,%f,%f)(%f,%f,%f)\n", i,
				correctL[0], correctL[1], correctL[2],
				correctL[3], correctL[4], correctL[5],
				correctL[6], correctL[7], correctL[8]);*/
		InverseMatrix3(correctL);
		/*if (elasticIndex == 0)
			printf("particle %d's inverseL is (%f,%f,%f)(%f,%f,%f)(%f,%f,%f)\n", i,
				correctL[0], correctL[1], correctL[2],
				correctL[3], correctL[4], correctL[5],
				correctL[6], correctL[7], correctL[8]);*/
	}
	else
		printf("ERROR:particle %d's correctL cannot be inversed! neighbor num is %d, correctL is (%f,%f,%f)(%f,%f,%f)(%f,%f,%f)\n", 
			i, buf.neighborNum[elasticIndex], correctL[0], correctL[1],correctL[2],correctL[3],
			correctL[4],correctL[5],correctL[6],correctL[7],correctL[8]);
//	float3 dist;
//	float c;
//	int jndex;
	for(int l=0;l<buf.neighborNum[elasticIndex];++l)
	{
		dist = buf.neighborDistance[elasticIndex * simData.maxNeighborNum + l];
		dsq = sqrt(dot(dist, dist));
		c = simData.psmoothradius - dsq;
		buf.kernelGrad[elasticIndex * simData.maxNeighborNum + l].x = correctL[0] * dist.x + correctL[1] * dist.y + correctL[2] * dist.z;
		buf.kernelGrad[elasticIndex * simData.maxNeighborNum + l].y = correctL[3] * dist.x + correctL[4] * dist.y + correctL[5] * dist.z;
		buf.kernelGrad[elasticIndex * simData.maxNeighborNum + l].z = correctL[6] * dist.x + correctL[7] * dist.y + correctL[8] * dist.z;
		buf.kernelGrad[elasticIndex * simData.maxNeighborNum + l].x *= simData.spikykern *c *c/dsq;
		buf.kernelGrad[elasticIndex * simData.maxNeighborNum + l].y *= simData.spikykern *c *c/dsq;
		buf.kernelGrad[elasticIndex * simData.maxNeighborNum + l].z *= simData.spikykern *c *c/dsq;
	
		//jndex = buf.neighborID[elasticIndex];
		//buf.initialVolume[elasticIndex] += simData.poly6kern * pow(c, 3) * buf.mf_restmass[i] * buf.mdensity[buf.particleID[jndex]];
	}
	
	buf.frame[elasticIndex] = 0;
	//if (i % 1000 == 0)
	//	printf("initial volume is %f\n", 1000000*buf.initialVolume[elasticIndex]);
}
__global__ void CheckCorrectedKernelGradientError(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;
	if (_example != 2)
	{
		if (buf.MFtype[i] != 2) return;
	}
	else
		if (buf.MFtype[i] < 2)
			return;
	int gc = buf.mgcell[i];
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	//if(i%100 == 0)
	//printf("particle %d's elasticID is %d\n", i, buf.elasticID[i]);
	if (gc == GRID_UNDEF) {
		buf.mpos[i] = make_float3(-1000, -1000, -1000);
		buf.maccel[i] = make_float3(0, 0, 0);
		return;
	}
	int index = buf.elasticID[i];
	int jndex, j;
	float3 dist;
	float check[9] = {0,0,0,0,0,0,0,0,0};
	float temp[9];
	//printf("particle %d's elasticIndex is %d\n", i, index);
	//if(index == 1600)
	//	printf("initial neighbor num is %d\n", buf.neighborNum[index]);
	for(int l=0;l<buf.neighborNum[index];++l)
	{
		jndex = buf.neighborID[index * simData.maxNeighborNum + l];
		dist = -buf.neighborDistance[index * simData.maxNeighborNum + l];
		//if (index == 100)
		//	printf("initial dist with %d is (%f,%f,%f)\n", jndex,dist.x, dist.y, dist.z);
	/*	if (index == 100 && jndex == 99)
			printf("initial dist is %f,%f,%f\n", dist.x, dist.y, dist.z);*/
		dist *= buf.initialVolume[jndex];
		/*if (index == 100 && jndex == 99)
			printf("initial kernel is %f,%f,%f\n", elasticInfo.kernelGrad[index * 600 + l].x, elasticInfo.kernelGrad[index * 600 + l].y, elasticInfo.kernelGrad[index * 600 + l].z);
		*/
		/*if (index == 100 && elasticInfo.neighborID[index * 600 + l] == 99)
			printf("initial volume is %.15f\n", elasticInfo.initialVolume[jndex]);*/
		tensorProduct(dist, buf.kernelGrad[index * simData.maxNeighborNum + l], temp);
		for (int k = 0; k < 9; ++k)
			check[k] += temp[k];
	}
	if (index == 1600)
		printf("checkError is (%f,%f,%f)(%f,%f,%f)(%f,%f,%f)\n",
			check[0], check[1], check[2],
			check[3], check[4], check[5],
			check[6], check[7], check[8]);
}
__device__ void contributeVolume(int i, int cell, bufList buf, int& index, float& volume)
{
	float dsq, c;
	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2 / d2;
	float3 dist, vmr;
	int j, jndex;
	if (buf.mgridcnt[cell] == 0) return;

	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];

	int elasticIndex = buf.elasticID[i];
	for (int cndx = cfirst; cndx < clast; cndx++)
	{
		j = buf.mgrid[cndx];
		if (buf.MFtype[j] == 2 || (_example == 2 && buf.MFtype[j] >= 2))
		{
			dist = (buf.mpos[i] - buf.mpos[j]);		// dist in cm
			dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
			if (!(dsq < r2 && dsq > 0))
				continue;
			if (index >= simData.maxNeighborNum)
				return;
			dsq = sqrt(dsq*d2);
			c = simData.psmoothradius - dsq;
			jndex = buf.elasticID[j];
			buf.neighborID[elasticIndex * simData.maxNeighborNum + index] = jndex;
			dist *= simData.psimscale;
			buf.neighborDistance[elasticIndex * simData.maxNeighborNum + index] = dist;
			volume += pow(buf.mf_restmass[j] * buf.density_solid[j], 2) 
				* simData.poly6kern * pow((r2*d2 - dsq*dsq), 3);
			index++;
		}
	}
}
__global__ void ComputeInitialVolume(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;
	if (_example != 2)
	{
		if (buf.MFtype[i] != 2) return;
	}
	else
		if (buf.MFtype[i] < 2)
			return;
	int gc = buf.mgcell[i];
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	//if(i%100 == 0)
	//printf("particle %d's elasticID is %d\n", i, buf.elasticID[i]);
	if (gc == GRID_UNDEF) {
		buf.mpos[i] = make_float3(-1000, -1000, -1000);
		buf.maccel[i] = make_float3(0, 0, 0);
		return;
	}
	gc -= nadj;
	int index = 0;
	int elasticIndex = buf.elasticID[i];

	buf.initialVolume[elasticIndex] = 0;
	buf.particleID[elasticIndex] = i;
	for (int c = 0; c < simData.gridAdjCnt; c++)
	{
		contributeVolume(i, gc + simData.gridAdj[c], buf, index, buf.initialVolume[elasticIndex]);
		if (index >= simData.maxNeighborNum)
			printf("ERROR:Neighbor space is not enough!\n");
	}
	//buf.initialVolume[elasticIndex] = pow(simData.psmoothradius / 2, 3);
	//buf.initialVolume[elasticIndex] += 
	//	pow(buf.mf_restmass[i] * buf.density_solid[elasticIndex], 2)*pow(simData.r2, 3)*simData.poly6kern;
	//if(elasticIndex%1000==0)
	//printf("elastic particle %d's initial volume is %.10f\n", elasticIndex, buf.initialVolume[elasticIndex]);
	buf.neighborNum[elasticIndex] = index;
	//if (buf.mpos[i].y > 20)
	//	buf.isHead[elasticIndex] = 1;
	//else
	//	buf.isHead[elasticIndex] = 0;
	
	//if (elasticIndex % 1000 == 0)
	//	printf("elastic particle %d's rest mass is %f, solid density is %f\n", elasticIndex, buf.mf_restmass[i], buf.density_solid[elasticIndex]);
}
void ComputeCorrectLCUDA()
{
	ComputeInitialVolume << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: computeInitialVolume: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();

	ComputeCorrectL << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: computeCorrectL: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();

	ComputeMap << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: computeMap: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();

	CheckCorrectedKernelGradientError << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: checkCKGradError: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();

	//testFunc << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	//error = cudaGetLastError();
	//if (error != cudaSuccess) {
	//	fprintf(stderr, "CUDA ERROR: checkCKGradError: %s\n", cudaGetErrorString(error));
	//}
	//cudaThreadSynchronize();
}
__device__ float contributeTest(int i, int cell, bufList buf)
{
	float dsq, c;
	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2 / d2;

	float3 dist, vmr;
	float cmterm, cmterm1;
	//	float massj;
	float pmterm, vmterm;
	//	float q;
	int j, mulj;
	float aveDenij, cx, xterm;
	//if (i % 100 == 0)
	//	printf("particle %d's gridcnt is %d\n", i,buf.mgridcnt[cell]);
	if (buf.mgridcnt[cell] == 0) return 0;

	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];

	int elasticIndex = buf.elasticID[i];
	int jndex;
	float sum = 0;
	for (int cndx = cfirst; cndx < clast; cndx++)
	{
		j = buf.mgrid[cndx];
		dist = (buf.mpos[i] - buf.mpos[j]);		// dist in cm
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
		if (!(dsq < r2 && dsq > 0))
			continue;
		c = (r2 - dsq)*d2;
		sum += buf.mf_restmass[j] / buf.mf_restdensity[j]* simData.poly6kern * pow(c, 3);
	}
	return sum;
}
__global__ void testFunc(bufList buf,int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;
	//if (buf.MFtype[i] != 1) return;
	int gc = buf.mgcell[i];
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	if (gc == GRID_UNDEF) {
		//buf.mpos[i] = make_float3(-1000, -1000, -1000);
		//buf.mvel[i] = make_float3(0, 0, 0);
		return;
	}
	gc -= nadj;
	float sum = 0;
	for (int c = 0; c < simData.gridAdjCnt; c++)
	{
		sum += contributeTest(i, gc + simData.gridAdj[c], buf);
	}
	if (i % 1000 == 0)
		printf("test sum is %f\n", sum);
	//if (buf.MFtype[i] != 1) return;
	//printf("particle %d is an elastic particle,ID is %d\n", i,buf.elasticID[i]);
}

__global__ void ComputeDeformGrad(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;
	if (_example != 2)
	{
		if (buf.MFtype[i] != 2) return;
	}
	else
		if (buf.MFtype[i] < 2)
			return;
	int gc = buf.mgcell[i];
	//if(i%100 == 0)
	//printf("particle %d's elasticID is %d\n", i, buf.elasticID[i]);
	if (gc == GRID_UNDEF) {
		buf.mpos[i] = make_float3(-1000, -1000, -1000);
		buf.maccel[i] = make_float3(0, 0, 0);
		return;
	}
	for (int l = 0; l < 9; ++l)
		buf.gradDeform[i*9+l] = 0;
	float3 dist,grad;
	int elasticIndex = buf.elasticID[i];
	if (buf.particleID[elasticIndex] != i)
		printf("map error!id is %d, i is %d\n", buf.particleID[elasticIndex], i);
	//elasticInfo.particleID[elasticIndex] = i;
	float tempDG[9];
	int jndex, j;
	//if(elasticIndex == 100)
	//	printf("now neighbor num is %d\n", elasticInfo.neighborNum[elasticIndex]);
	for (int l = 0; l<buf.neighborNum[elasticIndex]; ++l)
	{
		jndex = buf.neighborID[elasticIndex * simData.maxNeighborNum + l];
		j = buf.particleID[jndex];
		if(buf.elasticID[j]!=jndex)
		{
			printf("map error!\n");
			continue;
		}
		dist = (buf.mpos[j] - buf.mpos[i]) * simData.psimscale;
		//if (elasticIndex == 100)
		//	printf("now dist with %d is (%f,%f,%f)\n", jndex, dist.x, dist.y, dist.z);
		dist *= buf.initialVolume[buf.neighborID[elasticIndex * simData.maxNeighborNum + l]];
	/*	if (elasticIndex == 100 && elasticInfo.neighborID[elasticIndex * 600 + l] == 99)
			printf("now dist is %f,%f,%f\n", dist.x, dist.y, dist.z);*/
		/*if (elasticIndex == 100 && elasticInfo.neighborID[elasticIndex * 600 + l] == 99)
			printf("now kernel is %f,%f,%f\n", elasticInfo.kernelGrad[elasticIndex * 600 + l].x, elasticInfo.kernelGrad[elasticIndex * 600 + l].y, elasticInfo.kernelGrad[elasticIndex * 600 + l].z);*/
		/*if (elasticIndex == 100 && elasticInfo.neighborID[elasticIndex * 600 + l] == 99)
			printf("now volume is %.15f\n", elasticInfo.initialVolume[jndex]);*/
		grad = buf.kernelGrad[elasticIndex * simData.maxNeighborNum + l];
		tensorProduct(dist, grad, tempDG);
		for (int k = 0; k < 9; ++k)
			buf.gradDeform[i*9+k] += tempDG[k];
	}
	//if (buf.elasticID[i] == 1600)
	//	printf("particle %d's deform grad is (%f,%f,%f)(%f,%f,%f)(%f,%f,%f)\n", elasticIndex,
	//		buf.gradDeform[i * 9],
	//		buf.gradDeform[i * 9 + 1], buf.gradDeform[i * 9 + 2], buf.gradDeform[i * 9 + 3],
	//		buf.gradDeform[i * 9 + 4], buf.gradDeform[i * 9 + 5], buf.gradDeform[i * 9 + 6],
	//		buf.gradDeform[i * 9 + 7], buf.gradDeform[i * 9 + 8]);
	float q[9] = { 1,0,0,0,1,0,0,0,1 };
	float error = 0;
	float3 t;
	extractRotation(&buf.gradDeform[i * 9], q, 100);
	//if (i == 37000)
	//	printf("q is (%f,%f,%f,%f)\n", q[0], q[1], q[2], q[3]);
	for (int l = 0; l < 9; ++l)
		buf.Rotation[i * 9 + l] = q[l];
	
	for (int l = 0; l<buf.neighborNum[elasticIndex]; ++l)
	{
		buf.kernelRotate[elasticIndex * simData.maxNeighborNum + l] = 
			multiply_mv3(&buf.Rotation[i * 9], buf.kernelGrad[elasticIndex * simData.maxNeighborNum + l]);
	}

	/*if (buf.elasticID[i] == 100)
		printf("delta error is %f\n", error);*/
	/*if (buf.elasticID[i] == 1600)
		printf("particle %d's rotation is (%f,%f,%f)(%f,%f,%f)(%f,%f,%f)\n", i,
			buf.Rotation[i * 9],
			buf.Rotation[i * 9 + 1], buf.Rotation[i * 9 + 2], buf.Rotation[i * 9 + 3],
			buf.Rotation[i * 9 + 4], buf.Rotation[i * 9 + 5], buf.Rotation[i * 9 + 6],
			buf.Rotation[i * 9 + 7], buf.Rotation[i * 9 + 8]);*/
}

__global__ void ComputeFinalDeformGrad(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;
	if (_example != 2)
	{
		if (buf.MFtype[i] != 2) return;
	}
	else
		if (buf.MFtype[i] < 2)
			return;
	int gc = buf.mgcell[i];
	//if(i%100 == 0)
	//printf("particle %d's elasticID is %d\n", i, buf.elasticID[i]);
	if (gc == GRID_UNDEF) {
		buf.mpos[i] = make_float3(-1000, -1000, -1000);
		buf.maccel[i] = make_float3(0, 0, 0);
		return;
	}
	float3 rotatedKernelGrad;
	//compute corotated deformation gradient
	int elasticIndex = buf.elasticID[i];
	if (elasticIndex < 0 || elasticIndex >= simData.numElasticPoints)
		printf("elasticIndex wrong!\n");
	float3 grad,dist;
	float deformGrad[9];
	for(int k=0;k<9;++k)
		buf.gradDeform[i * 9 + k] = 0;
	int j, jndex;
	for (int l = 0; l<buf.neighborNum[elasticIndex]; ++l)
	{
		jndex = buf.neighborID[elasticIndex * simData.maxNeighborNum + l];
		j = buf.particleID[jndex];
		grad = buf.kernelRotate[elasticIndex * simData.maxNeighborNum + l];
		dist = buf.mpos[j] - buf.mpos[i];
		dist *=  simData.psimscale;
		//dist -= multiply_mv3(&buf.Rotation[i * 9], -elasticInfo.neighborDistance[elasticIndex * 600 + l]);
		dist *= buf.initialVolume[jndex];
		tensorProduct(dist, grad, deformGrad);
		for (int k = 0; k < 9; ++k)
			buf.gradDeform[i * 9 + k] += deformGrad[k];
		
	}
	//if (elasticIndex == 1600)
	//	printf("final deform gradient is (%f,%f,%f)(%f,%f,%f)(%f,%f,%f)\n",
	//		buf.gradDeform[i * 9], buf.gradDeform[i * 9 + 1], buf.gradDeform[i * 9 + 2],
	//		buf.gradDeform[i * 9 + 3], buf.gradDeform[i * 9 + 4], buf.gradDeform[i * 9 + 5],
	//		buf.gradDeform[i * 9 + 6], buf.gradDeform[i * 9 + 7], buf.gradDeform[i * 9 + 8]);
	/*buf.gradDeform[i * 9] += 1;
	buf.gradDeform[i * 9 + 4] += 1;
	buf.gradDeform[i * 9 + 8] += 1;*/

	////看看是否满足那个条件！
	//float test[9] = { 0,0,0,0,0,0,0,0,0 };
	//for (int l = 0; l<buf.neighborNum[elasticIndex]; ++l)
	//{
	//	jndex = buf.neighborID[elasticIndex * simData.maxNeighborNum + l];
	//	j = buf.particleID[jndex];
	//	grad = buf.kernelRotate[elasticIndex * simData.maxNeighborNum + l];
	//	dist = multiply_mv3(&buf.Rotation[i * 9], -buf.neighborDistance[elasticIndex * simData.maxNeighborNum + l]);
	//	dist *= buf.initialVolume[buf.neighborID[elasticIndex * simData.maxNeighborNum + l]];
	//	tensorProduct(dist, grad, deformGrad);
	//	for (int k = 0; k < 9; ++k)
	//		test[k] += deformGrad[k];

	//}
	//if (elasticIndex == 100)
	//	printf("test matrix is (%f,%f,%f)(%f,%f,%f)(%f,%f,%f)\n",
	//		test[0], test[1], test[2],
	//		test[3], test[4], test[5],
	//		test[6], test[7], test[8]);
}
__global__ void ComputeStrainAndStress(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;
	if (_example != 2)
	{
		if (buf.MFtype[i] != 2) return;
	}
	else
		if (buf.MFtype[i] < 2)
			return;
	int gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) {
		//buf.mpos[i] = make_float3(-1000, -1000, -1000);
		//buf.mvel[i] = make_float3(0, 0, 0);
		return;
	}
	float3 rotatedKernelGrad;
	//compute corotated deformation gradient
	int index = buf.elasticID[i];
	if (index < 0 || index >= simData.numElasticPoints)
		printf("elasticIndex wrong!\n");
	float3 grad, dist;
	float deformGrad[9];
	for (int k = 0; k<9; ++k)
		buf.gradDeform[i * 9 + k] = 0;
	int j, jndex;
	for (int l = 0; l<buf.neighborNum[index]; ++l)
	{
		jndex = buf.neighborID[index * simData.maxNeighborNum + l];
		j = buf.particleID[jndex];
		grad = buf.kernelRotate[index * simData.maxNeighborNum + l];
		dist = buf.mpos[j] - buf.mpos[i];
		dist *= simData.psimscale;
		dist *= buf.initialVolume[jndex];
		tensorProduct(dist, grad, deformGrad);
		for (int k = 0; k < 9; ++k)
			buf.gradDeform[i * 9 + k] += deformGrad[k];
	}

	//strain and stress
	float strain[9], stress[9];
	float alpha;
	transmit3(&buf.gradDeform[i * 9], stress);
	for (int l = 0; l < 9; ++l)
		strain[l] = 0.5*(buf.gradDeform[i * 9 + l] + stress[l]);

	strain[0] -= 1; strain[4] -= 1; strain[8] -= 1;
	buf.volumetricStrain[index] = strain[0] + strain[4] + strain[8];

	float lambda = simData.lambda;
	float tr_strain = strain[0] + strain[4] + strain[8];
	for (int l = 0; l < 9; ++l)
		stress[l] = 2 * simData.miu * strain[l];
	stress[0] += lambda * tr_strain; stress[4] += lambda * tr_strain; stress[8] += lambda * tr_strain;
	alpha = simData.poroDeformStrength*(1 - simData.bulkModulus_porous / simData.bulkModulus_grains) * buf.pressure_water[i*MAX_FLUIDNUM];

	stress[0] -= alpha;
	stress[4] -= alpha;
	stress[8] -= alpha;

	for (int l = 0; l < 9; ++l)
		buf.gradDeform[i * 9 + l] = stress[l];

}

__global__ void ComputeElasticForce(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;
	if (_example != 2)
	{
		if (buf.MFtype[i] != 2) return;
	}
	else
		if (buf.MFtype[i] < 2)
			return;
	int gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) 
	{
		//buf.mpos[i] = make_float3(-1000, -1000, -1000);
		//buf.mvel[i] = make_float3(0, 0, 0);
		return;
	}
	int index = buf.elasticID[i];
	if (index < 0 || index >= simData.numElasticPoints)
		printf("elasticIndex wrong!\n");
	int j, jndex, k;
	float3 force = make_float3(0, 0, 0);
	float3 t1, t2;
	for (int l = 0; l<buf.neighborNum[index]; ++l)
	{
		jndex = buf.neighborID[index * simData.maxNeighborNum + l];
		j = buf.particleID[jndex];
		k = buf.neighborIndex[index * simData.maxNeighborNum + l];
		t1 = multiply_mv3(&buf.gradDeform[i * 9], buf.kernelRotate[index * simData.maxNeighborNum + l]);
		t1 -= multiply_mv3(&buf.gradDeform[j * 9], buf.kernelRotate[jndex * simData.maxNeighborNum + k]);
		t1 *= buf.initialVolume[index];
		t1 *= buf.initialVolume[jndex];
		force += t1;
	}
	//if (index % 30000 == 0) 
	//	printf("solid particle %d's elastic force is (%f,%f,%f)\n", index, force.x, force.y, force.z);

	//buf.mforce[i] += force;
	//buf.maccel[i] += force;
	buf.bx[index] = buf.mveleval[i].x + simData.mf_dt*force.x / buf.mf_restmass[i];
	buf.by[index] = buf.mveleval[i].y + simData.mf_dt*force.y / buf.mf_restmass[i];
	buf.bz[index] = buf.mveleval[i].z + simData.mf_dt*force.z / buf.mf_restmass[i];

	//if (index % 10000 == 0)
	//	printf("b is (%f,%f,%f)\n", buf.bx[index], buf.by[index], buf.bz[index]);
	buf.vx[index] = buf.mveleval[i].x; buf.vy[index] = buf.mveleval[i].y; buf.vz[index] = buf.mveleval[i].z;
}

__global__ void ComputeIterationStrainAndStress(bufList buf, int pnum, float* px, float*py, float*pz)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;
	if (_example != 2)
	{
		if (buf.MFtype[i] != 2) return;
	}
	else
		if (buf.MFtype[i] < 2)
			return;
	int gc = buf.mgcell[i];
	//if(i%100 == 0)
	//printf("particle %d's elasticID is %d\n", i, buf.elasticID[i]);
	if (gc == GRID_UNDEF) {
		buf.mpos[i] = make_float3(-1000, -1000, -1000);
		buf.maccel[i] = make_float3(0, 0, 0);
		return;
	}
	float3 rotatedKernelGrad;
	//compute corotated deformation gradient
	int elasticIndex = buf.elasticID[i];
	if (elasticIndex < 0 || elasticIndex >= simData.numElasticPoints)
		printf("elasticIndex wrong!\n");
	float3 grad, dist;
	float deformGrad[9];
	for (int k = 0; k<9; ++k)
		buf.gradDeform[i * 9 + k] = 0;
	int j, jndex;
	int index = buf.elasticID[i];
	for (int l = 0; l<buf.neighborNum[elasticIndex]; ++l)
	{
		jndex = buf.neighborID[elasticIndex * simData.maxNeighborNum + l];
		j = buf.particleID[jndex];
		grad = buf.kernelRotate[elasticIndex * simData.maxNeighborNum + l];
		//dist = buf.mpos[j] - buf.mpos[i];
		//dist *= simData.psimscale;
		dist = make_float3(px[jndex] - px[elasticIndex], py[jndex] - py[elasticIndex], pz[jndex] - pz[elasticIndex]) * simData.mf_dt;
		//dist -= multiply_mv3(&buf.Rotation[i * 9], -elasticInfo.neighborDistance[elasticIndex * 600 + l]);
		dist *= buf.initialVolume[jndex];
		tensorProduct(dist, grad, deformGrad);
		for (int k = 0; k < 9; ++k)
			buf.gradDeform[i * 9 + k] += deformGrad[k];
	}


	//strain and stress
	float strain[9], stress[9];
	float alpha;
	transmit3(&buf.gradDeform[i * 9], stress);
	for (int l = 0; l < 9; ++l)
		strain[l] = 0.5*(buf.gradDeform[i * 9 + l] + stress[l]);

	//strain[0] -= 1; strain[4] -= 1; strain[8] -= 1;
	buf.volumetricStrain[index] = strain[0] + strain[4] + strain[8];

	float lambda = simData.lambda;
	float tr_strain = strain[0] + strain[4] + strain[8];
	for (int l = 0; l < 9; ++l)
		stress[l] = 2 * simData.miu * strain[l];
	stress[0] += lambda * tr_strain; stress[4] += lambda * tr_strain; stress[8] += lambda * tr_strain;
	alpha = simData.poroDeformStrength*(1 - simData.bulkModulus_porous / simData.bulkModulus_grains) * buf.pressure_water[i*MAX_FLUIDNUM];

	stress[0] -= alpha;
	stress[4] -= alpha;
	stress[8] -= alpha;

	for (int l = 0; l < 9; ++l)
		buf.gradDeform[i * 9 + l] = stress[l];
}
__global__ void ComputeIterationElasticForce(bufList buf, int pnum, float* px, float*py, float*pz)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;
	if (_example != 2)
	{
		if (buf.MFtype[i] != 2) return;
	}
	else
		if (buf.MFtype[i] < 2)
			return;

	int gc = buf.mgcell[i];
	if (gc == GRID_UNDEF)
	{
		return;
	}
	int index = buf.elasticID[i];
	if (index < 0 || index >= simData.numElasticPoints)
		printf("elasticIndex wrong!\n");
	int j, jndex, k;
	float3 force = make_float3(0, 0, 0);
	float3 t1, t2;
	for (int l = 0; l<buf.neighborNum[index]; ++l)
	{
		jndex = buf.neighborID[index * simData.maxNeighborNum + l];
		j = buf.particleID[jndex];
		k = buf.neighborIndex[index * simData.maxNeighborNum + l];
		t1 = multiply_mv3(&buf.gradDeform[i * 9], buf.kernelRotate[index * simData.maxNeighborNum + l]);
		t1 -= multiply_mv3(&buf.gradDeform[j * 9], buf.kernelRotate[jndex * simData.maxNeighborNum + k]);
		t1 *= buf.initialVolume[index];
		t1 *= buf.initialVolume[jndex];
		force += t1;
	}

	buf.Apx[index] = px[index] - simData.mf_dt*force.x / buf.mf_restmass[i];
	buf.Apy[index] = py[index] - simData.mf_dt*force.y / buf.mf_restmass[i];
	buf.Apz[index] = pz[index] - simData.mf_dt*force.z / buf.mf_restmass[i];
}
__global__ void initElasticIteration(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;
	if (_example != 2)
	{
		if (buf.MFtype[i] != 2) return;
	}
	else
		if (buf.MFtype[i] < 2)
			return;
	int gc = buf.mgcell[i];
	if (gc == GRID_UNDEF)
	{
		return;
	}
	int index = buf.elasticID[i];
	buf.px[index] = buf.rx[index] = buf.bx[index] - buf.Apx[index];
	buf.py[index] = buf.ry[index] = buf.by[index] - buf.Apy[index];
	buf.pz[index] = buf.rz[index] = buf.bz[index] - buf.Apz[index];
}
__global__ void updateV(bufList buf, int pnum, float3 alpha)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;
	if (_example != 2)
	{
		if (buf.MFtype[i] != 2) return;
	}
	else
		if (buf.MFtype[i] < 2)
			return;
	int gc = buf.mgcell[i];
	if (gc == GRID_UNDEF)
	{
		return;
	}
	int index = buf.elasticID[i];
	buf.vx[index] += alpha.x * buf.px[index]; buf.vy[index] += alpha.y * buf.py[index]; buf.vz[index] += alpha.z * buf.pz[index];
	buf.r2x[index] = buf.rx[index] - alpha.x*buf.Apx[index];
	buf.r2y[index] = buf.ry[index] - alpha.y*buf.Apy[index];
	buf.r2z[index] = buf.rz[index] - alpha.z*buf.Apz[index];
}
__global__ void updateP(bufList buf, int pnum, float3 beta)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;
	if (_example != 2)
	{
		if (buf.MFtype[i] != 2) return;
	}
	else
		if (buf.MFtype[i] < 2)
			return;
	int gc = buf.mgcell[i];
	if (gc == GRID_UNDEF)
	{
		return;
	}
	int index = buf.elasticID[i];
	buf.px[index] = buf.r2x[index] + beta.x*buf.px[index];
	buf.py[index] = buf.r2y[index] + beta.y*buf.py[index];
	buf.pz[index] = buf.r2z[index] + beta.z*buf.pz[index];

	buf.rx[index] = buf.r2x[index]; buf.ry[index] = buf.r2y[index]; buf.rz[index] = buf.r2z[index];
}
__global__ void ApplyElasticForce(bufList buf, int pnum, float* vx, float*vy, float*vz)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;
	
	if (_example != 2)
	{
		if (buf.MFtype[i] != 2) return;
	}
	else
		if (buf.MFtype[i] < 2)
			return;
	int gc = buf.mgcell[i];
	if (gc == GRID_UNDEF)
	{
		return;
	}
	int index = buf.elasticID[i];
	if (index < 0 || index >= simData.numElasticPoints)
		printf("elasticIndex wrong!\n");

	float3 force;
	force.x = (vx[index] - buf.mveleval[i].x) / simData.mf_dt;
	force.y = (vy[index] - buf.mveleval[i].y) / simData.mf_dt;
	force.z = (vz[index] - buf.mveleval[i].z) / simData.mf_dt;

	buf.pressForce[i] = force;
	buf.mforce[i] += force;
	buf.maccel[i] += force;
}
__device__ float contributeColorField(int i, int cell, bufList buf, int& count)
{
	float dsq, c, sum=0;
	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2 / d2;
	float3 dist, vmr;
	float cmterm;
	int j, jndex;
	if (buf.mgridcnt[cell] == 0) return sum;

	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];

	int index = buf.elasticID[i];
	for (int cndx = cfirst; cndx < clast; cndx++)
	{
		j = buf.mgrid[cndx];
		if (buf.MFtype[j] == 2 || (_example == 2 && buf.MFtype[j] >= 2))
		{
			dist = (buf.mpos[i] - buf.mpos[j]);		// dist in cm
			dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
			if (!(dsq < r2 && dsq > 0))
				continue;
			dsq = sqrt(dsq*d2);
			jndex = buf.elasticID[j];
			c = pow(simData.r2 - dsq*dsq, 3);
			cmterm = buf.mf_restmass[j] * buf.density_solid[j]*c*simData.poly6kern;
			sum += cmterm;
			count++;
		}
	}
	return sum;
}
__global__ void ComputeElasticColorField(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;
	if (_example != 2)
	{
		if (buf.MFtype[i] != 2) return;
	}
	else
		if (buf.MFtype[i] < 2)
			return;
	int gc = buf.mgcell[i];
	if (gc == GRID_UNDEF)
	{
		return;
	}
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	gc -= nadj;
	int index = buf.elasticID[i];
	if (index < 0 || index >= simData.numElasticPoints)
		printf("elasticIndex wrong!\n");
	buf.colorValue[i] = 0;
	int count = 0;
	for (int c = 0; c < simData.gridAdjCnt; c++)
	{
		buf.colorValue[i] += contributeColorField(i, gc + simData.gridAdj[c], buf, count);
	}
	if (count <= 25)
	//if(count<=20)
		buf.isSurface[index] = 1;
	else
		buf.isSurface[index] = 0;
}
__device__ float3 contributeElasticNormal(int i, int cell, bufList buf)
{
	float dsq, c;
	float3 sum = make_float3(0,0,0);
	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2 / d2;
	float3 dist, vmr;
	float cmterm;
	int j, jndex;
	if (buf.mgridcnt[cell] == 0) return sum;

	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];

	int index = buf.elasticID[i];
	for (int cndx = cfirst; cndx < clast; cndx++)
	{
		j = buf.mgrid[cndx];
		if (buf.MFtype[j] == 2 || (_example == 2 && buf.MFtype[j] >= 2))
		{
			dist = (buf.mpos[i] - buf.mpos[j]);		// dist in cm
			dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
			if (!(dsq < r2 && dsq > 0))
				continue;
			jndex = buf.elasticID[j];
			dsq = sqrt(dsq*d2);
			dist *= simData.psimscale;
			jndex = buf.elasticID[j];
			c = simData.psmoothradius - dsq;
			cmterm = buf.mf_restmass[j] * buf.density_solid[j] * c*c / dsq*simData.spikykern;
			sum += cmterm * (buf.colorValue[j] - buf.colorValue[i])*dist;
		
		}
	}
	return sum;
}
__global__ void ComputeElasticNormal(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;
	if (_example != 2)
	{
		if (buf.MFtype[i] != 2) return;
	}
	else
		if (buf.MFtype[i] < 2)
			return;
	int gc = buf.mgcell[i];
	if (gc == GRID_UNDEF)
	{
		return;
	}
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	gc -= nadj;
	int index = buf.elasticID[i];
	if (index < 0 || index >= simData.numElasticPoints)
		printf("elasticIndex wrong!\n");
	buf.normal[index] = make_float3(0,0,0);
	for (int c = 0; c < simData.gridAdjCnt; c++)
	{
		buf.normal[index] -= contributeElasticNormal(i, gc + simData.gridAdj[c], buf);
	}
	float d = dot(buf.normal[index], buf.normal[index]);
	if (d != 0)
		buf.normal[index] /= sqrt(d);
	
}
void ComputeElasticForceCUDA()
{
	ComputeDeformGrad << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: computeCorrectL: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();

	//ComputeFinalDeformGrad << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	//error = cudaGetLastError();
	//if (error != cudaSuccess) {
	//	fprintf(stderr, "CUDA ERROR: compute final deformable gradient: %s\n", cudaGetErrorString(error));
	//}
	//cudaThreadSynchronize();

	ComputeStrainAndStress << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: compute strain and stress: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();

	ComputeElasticForce << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: compute elastic force: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();

	int countNum = 0;
	float errorIter, precision = 0.01;
	float3 alpha, beta;
	cublasHandle_t handle;
	cublasCreate(&handle);

	ComputeIterationStrainAndStress << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum, fbuf.vx, fbuf.vy, fbuf.vz);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: compute iteration strain and stress: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();

	ComputeIterationElasticForce << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum, fbuf.vx, fbuf.vy, fbuf.vz);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: compute iteration elastic force: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();

	initElasticIteration << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: init elastic iteration: %s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
	float al = -1, t1, t2, t3;
	do {
		countNum++;
		ComputeIterationStrainAndStress << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum, fbuf.px, fbuf.py, fbuf.pz);
		error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf(stderr, "CUDA ERROR: compute iteration strain and stress: %s\n", cudaGetErrorString(error));
		}
		cudaDeviceSynchronize();

		ComputeIterationElasticForce << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum, fbuf.px, fbuf.py, fbuf.pz);
		error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf(stderr, "CUDA ERROR: compute iteration elastic force: %s\n", cudaGetErrorString(error));
		}
		cudaDeviceSynchronize();

		cublasSdot(handle, fcuda.numElasticPoints, fbuf.rx, 1, fbuf.rx, 1, &(alpha.x));
		cublasSdot(handle, fcuda.numElasticPoints, fbuf.ry, 1, fbuf.ry, 1, &(alpha.y));
		cublasSdot(handle, fcuda.numElasticPoints, fbuf.rz, 1, fbuf.rz, 1, &(alpha.z));

		cublasSdot(handle, fcuda.numElasticPoints, fbuf.px, 1, fbuf.Apx, 1, &(beta.x));
		cublasSdot(handle, fcuda.numElasticPoints, fbuf.py, 1, fbuf.Apy, 1, &(beta.y));
		cublasSdot(handle, fcuda.numElasticPoints, fbuf.pz, 1, fbuf.Apz, 1, &(beta.z));
		cudaDeviceSynchronize();
		alpha /= beta;

		updateV << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum, alpha);
		error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf(stderr, "CUDA ERROR: compute update V: %s\n", cudaGetErrorString(error));
		}
		cudaDeviceSynchronize();

		//t1 = 0; t2 = 0; t3 = 0;
		cublasSasum(handle, fcuda.numElasticPoints, fbuf.r2x, 1, &t1);
		cublasSasum(handle, fcuda.numElasticPoints, fbuf.r2y, 1, &t2);
		cublasSasum(handle, fcuda.numElasticPoints, fbuf.r2z, 1, &t3);
		cudaDeviceSynchronize();
		errorIter = t1 + t2 + t3;
		if (errorIter < precision)
			break;
		//printf("iter num is %d, error is %f\n", countNum, errorIter);
		cublasSdot(handle, fcuda.numElasticPoints, fbuf.r2x, 1, fbuf.r2x, 1, &(beta.x));
		cublasSdot(handle, fcuda.numElasticPoints, fbuf.r2y, 1, fbuf.r2y, 1, &(beta.y));
		cublasSdot(handle, fcuda.numElasticPoints, fbuf.r2z, 1, fbuf.r2z, 1, &(beta.z));

		cublasSdot(handle, fcuda.numElasticPoints, fbuf.rx, 1, fbuf.rx, 1, &(alpha.x));
		cublasSdot(handle, fcuda.numElasticPoints, fbuf.ry, 1, fbuf.ry, 1, &(alpha.y));
		cublasSdot(handle, fcuda.numElasticPoints, fbuf.rz, 1, fbuf.rz, 1, &(alpha.z));
		cudaDeviceSynchronize();

		beta /= alpha;
		updateP << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum, beta);
		error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf(stderr, "CUDA ERROR: compute update V: %s\n", cudaGetErrorString(error));
		}
		cudaDeviceSynchronize();

	} while (countNum < 5);
	//ex1 for 5, ex2 for 5
	//printf("\n");

	ApplyElasticForce << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum, fbuf.vx, fbuf.vy, fbuf.vz);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: apply elastic force: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();

	ComputeElasticColorField << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: compute elastic color field: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();

	ComputeElasticNormal << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: compute elastic normal: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();
	cublasDestroy(handle);
}

__device__ float contributeDivDarcyFlux(int i, int cell, bufList buf, float&normalize)
{
	float dsq, c;
	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2;
	//int jndex;
	float3 dist, vmr;
	float cmterm, cmterm1;
	//	float massj;
	float pmterm, vmterm;
	//	float q;
	int j, mulj;
	float aveDenij, cx, xterm;
	float sum = 0;
	//if (i % 100 == 0)
	//	printf("particle %d's gridcnt is %d\n", i,buf.mgridcnt[cell]);
	if (buf.mgridcnt[cell] == 0) return sum;

	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];
	//int index = buf.elasticID[i];
	//int jndex,index = buf.elasticID[i];
	for (int cndx = cfirst; cndx < clast; cndx++)
	{
		j = buf.mgrid[cndx];
		if (buf.misbound[j])
			continue;
		if (buf.MFtype[i] == buf.MFtype[j] && buf.MFtype[i] == 0)
			continue;
		//jndex = buf.elasticID[j];
		dist = (buf.mpos[i] - buf.mpos[j])*simData.psimscale;		// dist in cm
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
		if (!(dsq < r2 && dsq > 0))
			continue;
		dsq = sqrt(dsq);
		c = simData.psmoothradius - dsq;
		//cmterm = c*c*simData.spikykern * simData.pmass / buf.mf_restdensity[j] / dsq;
		//cmterm = -1 / dsq;
		cmterm = c*c*simData.spikykern * simData.pmass / simData.mf_dens[1];
		//cmterm = c*c*simData.spikykern * simData.pmass * buf.density_solid[buf.elasticID[j]] / dsq;
		//if (buf.MFtype[i] == buf.MFtype[j])
		sum += dot((buf.gradPressure[j]+ buf.gradPressure[i])*0.5, dist/dsq)*cmterm;
		normalize += cmterm;
		//else
		//	sum += dot(buf.gradPressure[i], dist)*cmterm;
	}
	return sum;
}

__device__ void contributePorePressure(int i, int cell, bufList buf,float* beta, float &sum, float&b)
{
	float dsq, c;
	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2;

	float3 dist, vmr;
	float cmterm, cmterm1;
	//	float massj;
	float3 pmterm, vmterm;
	//	float q;
	int j, mulj;
	float aveDenij, cx, xterm;
	
	if (buf.mgridcnt[cell] == 0) return;

	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];
	//int index = buf.elasticID[i];
	float q;
	for (int cndx = cfirst; cndx < clast; cndx++)
	{
		j = buf.mgrid[cndx];
		if (buf.MFtype[j] != 0)
			continue;
		dist = (buf.mpos[i] - buf.mpos[j])*simData.psimscale;		// dist in cm
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
		q = sqrt(dsq / r2);

		if (q <= 0 || q >= 1)
			continue;
		if (q < 0.5)
			cmterm = 6 * (q*q*q - q*q) + 1;
		else
			cmterm = 2 * pow(1 - q, 3);

		//if (q >= 1)
		//	continue;
		//if (q >= 0 && q <= 0.5)
		//	cmterm = buf.density_solid[i] * (6 * (q*q*q - q*q) + 1);
		//else
		//	cmterm = buf.density_solid[i] * 2*pow(1-q,3);
		if (buf.totalDis[j*MAX_SOLIDNUM + buf.MFtype[i] - 2] <= 0.000001)
			b = 1;
		cmterm *= buf.mf_restmass[j] / buf.mf_restdensity[j];
		//cmterm = pow((r2 - dsq), 3)*simData.poly6kern*buf.mf_restmass[j] * buf.mdensity[j] / buf.totalDis[j];
		/*if (isnan(cmterm))
			continue;*/
		//cmterm *= buf.mf_restmass[j] / buf.mf_restdensity[j];
		//if (buf.totalDis[j*MAX_SOLIDNUM + buf.MFtype[i] - 2] == 0)
		//	continue;
		cmterm /= buf.totalDis[j*MAX_SOLIDNUM + buf.MFtype[i] - 2];
		
		for (int k = 1; k < MAX_FLUIDNUM; ++k)
		{
			
			//for (int l = 0; l < MAX_SOLIDNUM; ++l)
			{
				sum += (buf.mf_beta[j*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM+buf.MFtype[i]-2] * simData.mf_dens[k] / simData.mf_mass[k]) * cmterm;
				beta[k] += (buf.mf_beta[j*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + buf.MFtype[i] - 2] * simData.mf_dens[k] / simData.mf_mass[k]) * cmterm;
				if (isnan(sum))
				{
					b = buf.mf_restdensity[j];
					return;
				}
			}
	/*		sum += buf.mf_beta[j*MAX_FLUIDNUM + k] * cmterm;
			beta[k] += buf.mf_beta[j*MAX_FLUIDNUM + k] * cmterm;*/
		}
	}
}


__global__ void ComputeSolidPorePressure(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;
	if (buf.MFtype[i]==1)return;
	int gc = buf.mgcell[i];
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	if (gc == GRID_UNDEF) {
		return;
	}
	gc -= nadj;
	float fluidSum = 0;
	float beta[MAX_FLUIDNUM];
	float normalize = 0;
	//if (i % 10000 == 0)
	//	printf("pressure ratio is (%f,%f,%f,%f),(%f,%f,%f,%f),(%f,%f,%f,%f) \n",
	//		simData.pressRatio[4], simData.pressRatio[5], simData.pressRatio[6], simData.pressRatio[7]
	//		, simData.pressRatio[8], simData.pressRatio[9], simData.pressRatio[10], simData.pressRatio[11]
	//		, simData.pressRatio[12], simData.pressRatio[13], simData.pressRatio[14], simData.pressRatio[15]);
	//if(buf.MFtype[i] == 0)
	//printf("%d's type is %d, beta is (%f,%f,%f)\n", i, buf.MFtype[i], beta[0], beta[1],
	//	beta[2]);
	float b = 10;
	if (buf.MFtype[i] > 1)
	{
		for (int k = 0; k < MAX_FLUIDNUM; ++k)
			beta[k] = 0;
		for (int c = 0; c < simData.gridAdjCnt; c++)
		{
			contributePorePressure(i, gc + simData.gridAdj[c], buf, beta, fluidSum, b);
		}
		/*if (fluidSum > 0.1)
			printf("fluid sum is %f, beta is (%f,%f,%f,%f)\n",
				fluidSum, beta[0], beta[1], beta[2], beta[3]);*/
		for (int k = 0; k < MAX_FLUIDNUM; ++k)
		{
			//if (buf.MFtype[i] == 2||(_example==2))
			if (buf.MFtype[i] == 2)
				buf.pressure_water[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM+buf.MFtype[i]-2] = 
					simData.CoCompressibility*(fluidSum - (1 - simData.bulkModulus_porous / simData.bulkModulus_grains)*buf.volumetricStrain[buf.elasticID[i]]);
			else
				buf.pressure_water[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + buf.MFtype[i] - 2] = simData.CoCompressibility*fluidSum;
			if (isnan(buf.pressure_water[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + buf.MFtype[i] - 2]))
				printf("solid %d's pore pressure is nan.beta is (%f,%f,%f)  density solid is %f, b is %.10f\n", 
					i, beta[1], beta[2], beta[3], buf.density_solid[i], b);
			//if(buf.mpos[i].y>60&&i%10==0)
			//	printf("press water is %f\n", buf.pressure_water[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + buf.MFtype[i] - 2]);
			buf.mf_beta[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + buf.MFtype[i] - 2] = beta[k];
		}

		//float mass = simData.mf_mass[0];
		//if (_example == 2 && buf.MFtype[i]>1)
		//	for (int k = 1; k < MAX_FLUIDNUM; ++k)
		//		mass += 0.05*buf.mf_beta[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + buf.MFtype[i] - 2] * simData.mf_mass[buf.MFtype[i] - 2];
		//buf.mf_restmass[i] = mass;

		/*if(buf.elasticID[i]%1000==0&& abs(buf.volumetricStrain[buf.elasticID[i]])>0.001)
			printf("elastic %d's volume strain is %f\n", buf.elasticID[i],
				buf.volumetricStrain[buf.elasticID[i]]);*/
	}
	else
	{
		for (int k = 1; k < MAX_FLUIDNUM; ++k)
			for (int l = 0; l < MAX_SOLIDNUM; ++l)
			{
				buf.pressure_water[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l] = simData.pressRatio[k*MAX_SOLIDNUM + l] * simData.rest_porosity*simData.CoCompressibility;// *buf.mf_alpha[i*MAX_FLUIDNUM + k];
				//if (i % 10000 == 0)
				//	printf("%d's press water is %f\n", i, buf.pressure_water[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l]);
			}//buf.pressure_water[i*MAX_FLUIDNUM*MAX_SOLIDNUM + 0] = (simData.pressRatio[1]*buf.mf_beta[i*MAX_FLUIDNUM+1]+simData.pressRatio[2]*buf.mf_beta[i*MAX_FLUIDNUM + 2]) * simData.rest_porosity*simData.CoCompressibility;
		
	}
	
}


__device__ void findNearbySolid(int i, int cell, bufList buf) 
{
	float dsq, c, dsq2;
	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2;
	
	float3 dist;
	float cmterm;
	float pmterm;
	int j, jndex;
	//int t = -1;
	if (buf.mgridcnt[cell] == 0) return;

	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];

	float q;
	for (int cndx = cfirst; cndx < clast; cndx++) 
	{
		j = buf.mgrid[cndx];
		if (buf.MFtype[j] == 1|| buf.MFtype[j] == 0)
			continue;
		//if (buf.isSurface[buf.elasticID[j]] == 0)
		//	continue;
		dist = (buf.mpos[i] - buf.mpos[j])*simData.psimscale;		// dist in cm
		dsq2 = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
		if (!(dsq2 < r2 && dsq2 > 0))
			continue;
		q = sqrt(dsq2 / r2);
		if (q >= 0 && q <= 0.5)
			buf.totalDis[i*MAX_SOLIDNUM + buf.MFtype[j] - 2] += (6 * (pow(q, 3) - pow(q, 2)) + 1);
		else
			buf.totalDis[i*MAX_SOLIDNUM + buf.MFtype[j] - 2] += 2 * pow(1 - q, 3);
		buf.solidCount[i*MAX_SOLIDNUM + buf.MFtype[j] - 2] += 1;


		//if (q > 2)
		//	continue;
		//if (q >= 0 && q <= 1)
		//	buf.totalDis[i*MAX_SOLIDNUM+buf.MFtype[j]-2] += simData.CubicSplineKern2*(1 - 1.5*q*q*(1 - q / 2));
		//else
		//	buf.totalDis[i*MAX_SOLIDNUM + buf.MFtype[j] - 2] += simData.CubicSplineKern1*pow(2 - q, 3);

		//total_dist += pow((r2 - dsq2), 3)*simData.poly6kern*buf.mf_restmass[i] * buf.mdensity[i];
		//total_dist += sqrt(dsq2);
		
	}
}
__global__ void FindNearbySolid(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum)
		return;
	if (buf.MFtype[i] != 0)
		return;
	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) return;						// particle out-of-range
	gc -= nadj;
	for (int k = 0; k < MAX_SOLIDNUM; ++k) {
		buf.totalDis[i*MAX_SOLIDNUM + k] = 0;
		buf.solidCount[i*MAX_SOLIDNUM + k] = 0;
	}
	for (int c = 0; c < simData.gridAdjCnt; c++)
	{
		findNearbySolid(i, gc + simData.gridAdj[c], buf);
	}
	//for (int k = 0; k < MAX_SOLIDNUM; ++k)
	//	buf.totalDis[i*MAX_SOLIDNUM + k] *= buf.mf_restmass[i] * buf.mdensity[i];
	//if (buf.solidCount[i] >= 25)
	//	buf.isInside[i] = true;
	//else
	//	buf.isInside[i] = false;
	float step;
	float betasum = 0;
	for (int l = 0; l < MAX_SOLIDNUM; ++l)
	{
		
		if (buf.solidCount[i*MAX_SOLIDNUM + l] == 0)
		{
			for (int k = 1; k < simData.mf_catnum; ++k)
			{
				step = (-buf.mf_beta[i*simData.mf_catnum*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l]);
				buf.mf_beta[i*simData.mf_catnum*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l] = 0;
				buf.mf_beta_next[i*simData.mf_catnum*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l] = 0;

				buf.mf_alpha[i*simData.mf_catnum + k] -= step;
				buf.mf_alpha_next[i*simData.mf_catnum + k] -= step;
			}
		}
		for (int k = 1; k < simData.mf_catnum; ++k)
			betasum += buf.mf_beta[i*simData.mf_catnum*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l];
	}
	buf.mf_alpha_sum[i] = 0;
	buf.mf_restdensity_out[i] = 0;
	//buf.rest_colorValue[i] = 0;
	for (int k = 1; k < simData.mf_catnum; ++k)
	{
		buf.mf_alpha_sum[i] += buf.mf_alpha[i*simData.mf_catnum + k];
		buf.mf_restdensity_out[i] += buf.mf_alpha[i*simData.mf_catnum + k] * simData.mf_dens[k];
		//buf.rest_colorValue[i] += buf.mf_alpha[i*simData.mf_catnum + k] * simData.colorValue[k];
	}
	if (abs(betasum + buf.mf_alpha_sum[i] - 1) > 0.01 || isnan(betasum))
		printf("alphasum is %f, betasum is %f\n", buf.mf_alpha_sum[i], betasum);
	if (buf.mf_alpha_sum[i] > 0.0001)
		buf.mf_restdensity_out[i] /= buf.mf_alpha_sum[i];
	else
	{
		buf.mf_restdensity_out[i] = 1;
		buf.mf_alpha_sum[i] = 0;
	}
	////if (i % 10000 == 0)
	//if(buf.mf_alpha_sum[i] < 0.99)
	//	printf("mf_dens is (%f,%f,%f,%f), alpha sum is %f, densityout is %f, alpha is (%f,%f,%f), solid count is (%d,%d,%d,%d),beta is (%f,%f,%f,%f)(%f,%f,%f,%f)(%f,%f,%f,%f)(%f,%f,%f,%f)\n",
	//		simData.mf_dens[0], simData.mf_dens[1], simData.mf_dens[2], simData.mf_dens[3], buf.mf_alpha_sum[i], buf.mf_restdensity_out[i],
	//		buf.mf_alpha[i*simData.mf_catnum + 1], buf.mf_alpha[i*simData.mf_catnum + 2], buf.mf_alpha[i*simData.mf_catnum + 3],
	//		buf.solidCount[i*MAX_SOLIDNUM + 0], buf.solidCount[i*MAX_SOLIDNUM + 1], buf.solidCount[i*MAX_SOLIDNUM + 2], buf.solidCount[i*MAX_SOLIDNUM + 3],
	//		buf.mf_beta[i*simData.mf_catnum*MAX_SOLIDNUM + 0 * MAX_SOLIDNUM + 0], buf.mf_beta[i*simData.mf_catnum*MAX_SOLIDNUM + 1 * MAX_SOLIDNUM + 0],
	//		buf.mf_beta[i*simData.mf_catnum*MAX_SOLIDNUM + 2 * MAX_SOLIDNUM + 0], buf.mf_beta[i*simData.mf_catnum*MAX_SOLIDNUM + 3 * MAX_SOLIDNUM + 0],
	//		buf.mf_beta[i*simData.mf_catnum*MAX_SOLIDNUM + 0 * MAX_SOLIDNUM + 1],buf.mf_beta[i*simData.mf_catnum*MAX_SOLIDNUM + 1*MAX_SOLIDNUM + 1], 
	//		buf.mf_beta[i*simData.mf_catnum*MAX_SOLIDNUM + 2*MAX_SOLIDNUM + 1],buf.mf_beta[i*simData.mf_catnum*MAX_SOLIDNUM + 3*MAX_SOLIDNUM + 1],
	//		buf.mf_beta[i*simData.mf_catnum*MAX_SOLIDNUM + 0 * MAX_SOLIDNUM + 2], buf.mf_beta[i*simData.mf_catnum*MAX_SOLIDNUM + 1 * MAX_SOLIDNUM + 2],
	//		buf.mf_beta[i*simData.mf_catnum*MAX_SOLIDNUM + 2 * MAX_SOLIDNUM + 2], buf.mf_beta[i*simData.mf_catnum*MAX_SOLIDNUM + 3 * MAX_SOLIDNUM + 2], 
	//		buf.mf_beta[i*simData.mf_catnum*MAX_SOLIDNUM + 0 * MAX_SOLIDNUM + 3], buf.mf_beta[i*simData.mf_catnum*MAX_SOLIDNUM + 1 * MAX_SOLIDNUM + 3],
	//		buf.mf_beta[i*simData.mf_catnum*MAX_SOLIDNUM + 2 * MAX_SOLIDNUM + 3], buf.mf_beta[i*simData.mf_catnum*MAX_SOLIDNUM + 3 * MAX_SOLIDNUM + 3]);

}
__device__ int findNearestSolid(int i, int cell, bufList buf, float*distance) {
	float dsq, c, dsq2;
	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2;

	float3 dist;
	float cmterm;
	float pmterm;
	int j, jndex;
	int t = -1;
	if (buf.mgridcnt[cell] == 0) return -1;

	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];

	for (int cndx = cfirst; cndx < clast; cndx++)
	{
		j = buf.mgrid[cndx];
		if (buf.MFtype[j] <= 1)
			continue;
		dist = (buf.mpos[i] - buf.mpos[j])*simData.psimscale;		// dist in cm
		dsq2 = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
		if (!(dsq2 < r2 && dsq2 > 0))
			continue;
		if (dsq2 < distance[buf.MFtype[j] - 2])
		{
			distance[buf.MFtype[j] - 2] = dsq2;
			t = j;
		}
	}
	return t;
}
__global__ void ComputeFPCorrection(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum)
		return;

	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) return;						// particle out-of-range
	buf.rest_colorValue[i] = simData.colorValue[0];
	if (buf.MFtype[i] != 0)
		return;
	gc -= nadj;
	float distance[MAX_SOLIDNUM];
	for (int k = 0; k < MAX_SOLIDNUM; ++k) {
		distance[k] = simData.r2;
	}
	int j = -1, t;
	//buf.fluidPercent[i] = buf.nextFluidPercent[i];
	float step;
	for (int c = 0; c < simData.gridAdjCnt; c++)
	{
		t = findNearestSolid(i, gc + simData.gridAdj[c], buf, distance);
		/*if (t != -1)
			j = t;*/
	}

	float oldFP;
	

}
__device__ void contributePoroVelocity(int i, int cell, bufList buf, float3* poroVel, float* normalize, float3* advectVel, int &count)
{
	float dsq, c, dsq2;
	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2;

	float3 dist;
	float cmterm;
	float pmterm;
	int j, jndex;
	
	if (buf.mgridcnt[cell] == 0) return;

	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];
	float q = 0;
	for (int cndx = cfirst; cndx < clast; cndx++)
	{
		j = buf.mgrid[cndx];
		if (buf.MFtype[j] <= 1)
			continue;
		dist = (buf.mpos[i] - buf.mpos[j])*simData.psimscale;		// dist in cm
		dsq2 = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
		q = sqrt(dsq2 / r2);

		if (q >= 1 || q <= 0)
			continue;
		if (q <= 0.5)
			pmterm = 6 * (q*q*q - q*q) + 1;
		else
			pmterm = 2 * pow(1 - q, 3);

		//if (q >= 2 || q <= 0)
		//	continue;
		//if (q > 1)
		//	pmterm = 0.25*pow(2 - q, 3);
		//else
		//	pmterm = 1 - 1.5*q*q*(1 - 0.5*q);

		//pmterm *= buf.density_solid[j];
		//pmterm *= simData.CubicSplineKern2;
		for (int k = 1; k < simData.mf_catnum; k++)
		{
			if (isnan(dot(buf.gradPressure[j*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + buf.MFtype[j] - 2], buf.gradPressure[j*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + buf.MFtype[j] - 2])))
			{
				count++;
				continue;
			}
			poroVel[k*MAX_SOLIDNUM + buf.MFtype[j] - 2] += pmterm * buf.gradPressure[j*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM+ buf.MFtype[j] - 2];
			advectVel[k*MAX_SOLIDNUM + buf.MFtype[j] - 2] += pmterm * buf.mveleval[j];
		}
		normalize[buf.MFtype[j] - 2] += pmterm;
	}
	return;
}
__global__ void ComputePoroVelocity(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum)
		return;
	if (buf.MFtype[i] != 0)
		return;
	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) return;						// particle out-of-range
	gc -= nadj;
	float normalize[MAX_SOLIDNUM];// = 0;
	float3 poroVel[MAX_FLUIDNUM * MAX_SOLIDNUM];
	float3 advectVel[MAX_FLUIDNUM * MAX_SOLIDNUM];
	float3 force, forcesum = make_float3(0,0,0);
	float betadensity = 0;
	float betasum = 0;
	//buf.poroForce[i] = make_float3(0, 0, 0);
	int count = 0;
	for (int k = 1; k < simData.mf_catnum*MAX_SOLIDNUM; ++k)
	{
		for (int l = 0; l < MAX_SOLIDNUM; ++l) 
		{
			poroVel[k*MAX_SOLIDNUM+l] = make_float3(0, 0, 0);
			advectVel[k*MAX_SOLIDNUM + l] = make_float3(0, 0, 0);
			betadensity += buf.mf_beta[i*simData.mf_catnum*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l] * simData.mf_dens[k];
			betasum += buf.mf_beta[i*simData.mf_catnum*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l];
			normalize[l] = 0;
		}
	}
	if (buf.mf_restdensity[i] <= 10)
		printf("rest den222 is %f, alpha is (%f,%f,%f), betasum is %f\n",
			buf.mf_restdensity[i], buf.mf_alpha[i*MAX_FLUIDNUM + 1], buf.mf_alpha[i*MAX_FLUIDNUM + 2], buf.mf_alpha[i*MAX_FLUIDNUM + 3],
			betasum);

	if (betadensity > 1)
		betadensity /= betasum;
	//int count = 0;
	for (int c = 0; c < simData.gridAdjCnt; c++)
	{
		contributePoroVelocity(i, gc + simData.gridAdj[c], buf, poroVel, normalize, advectVel, count);
	}
	buf.poroForce[i] = make_float3(0, 0, 0);
	float3 porevel, advectV;
	for (int l = 0; l < MAX_SOLIDNUM; ++l)
	{
		if (normalize[l] != 0)
		{
			for (int k = 1; k < simData.mf_catnum; ++k)
			{
				porevel = poroVel[k*MAX_SOLIDNUM + l];
				advectV = advectVel[k*MAX_SOLIDNUM + l];
				poroVel[k*MAX_SOLIDNUM + l] /= buf.totalDis[i*MAX_SOLIDNUM + l];
				advectVel[k*MAX_SOLIDNUM + l] /= buf.totalDis[i*MAX_SOLIDNUM + l];
				//poroVel[k] /= abs(normalize);
				//advectVel[k] /= abs(normalize);
				buf.poroVel[i*simData.mf_catnum*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l] = poroVel[k*MAX_SOLIDNUM + l] + advectVel[k*MAX_SOLIDNUM + l];

				//force = buf.mf_beta[i*MAX_FLUIDNUM + k]*(poroVel[k] - buf.mveleval[i])/simData.mf_dt;
				force = simData.mf_dens[k] * buf.poroVel[i*simData.mf_catnum*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l] - betadensity * buf.mveleval[i];
				force *= buf.mf_beta[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM+l] / (simData.mf_dt*buf.mf_restdensity[i]);
				//buf.mforce[i] += force;
				forcesum += force;
				//buf.poroForce[i] += force;
				if (isnan(dot(force, force)))
					printf("phase %d's pore force is nan,poro vel is (%f,%f,%f), advect vel is (%f,%f,%f), total dis is %f, count is %d\n", k,
						k, porevel.x, porevel.y, porevel.z,
						advectV.x, advectV.y, advectV.z,
						buf.totalDis[i*MAX_SOLIDNUM + l], count);
			}
			//if (buf.mf_alpha[i*MAX_FLUIDNUM + 1] > 0.99 && dot(buf.poroForce[i], buf.poroForce[i]) > 1)
			//	printf("%d's alpha is (%f,%f), beta is (%f,%f), vel is (%f,%f,%f), poro force is (%f,%f,%f)\n",
			//		i, buf.mf_alpha[i*MAX_FLUIDNUM + 1], buf.mf_alpha[i*MAX_FLUIDNUM + 2],
			//		buf.mf_beta[i*MAX_FLUIDNUM + 1], buf.mf_beta[i*MAX_FLUIDNUM + 2], buf.mveleval[i].x, buf.mveleval[i].y,
			//		buf.mveleval[i].z, buf.poroForce[i].x, buf.poroForce[i].y, buf.poroForce[i].z);
		}
		else
		{
			for (int k = 1; k < simData.mf_catnum; ++k)
				buf.poroVel[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l] = make_float3(0, 0, 0);
		}
	}

	//if (isnan(dot(forcesum, forcesum)))
	//	printf("particle %d's type is %d, poro accel is nan, total distance is %f\n",
	//		i, buf.MFtype[i], buf.totalDis[i*MAX_SOLIDNUM + 3]);

	//if (buf.MFtype[i] == 0 && i % 10000 == 0)
	//	printf("particle %d's mixture vel is (%f,%f,%f), fluid vel is (%f,%f,%f)\n",
	//		i, buf.mveleval[i].x, buf.mveleval[i].y, buf.mveleval[i].z,
	//		buf.fluidVel[i*MAX_FLUIDNUM + 1].x, buf.fluidVel[i*MAX_FLUIDNUM + 1].y,
	//		buf.fluidVel[i*MAX_FLUIDNUM + 1].z);
	//betasum = forcesum.x*forcesum.x + forcesum.y*forcesum.y + forcesum.z*forcesum.z;
	//if (betasum > simData.AL2) {
	//	forcesum *= simData.AL / sqrt(betasum);
	//}
	buf.mforce[i] += forcesum;
}

__device__ void contributeFluidFlux(int i, int cell, bufList buf, float&normalize)
{
	float dsq, c;
	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2 / d2;

	float3 dist, vmr;
	float cmterm, cmterm1;
	//	float massj;
	float3 pmterm, vmterm;
	//	float q;
	int j, mulj;
	float aveDenij, cx, xterm;

	if (buf.mgridcnt[cell] == 0) return;

	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];
	int index = buf.elasticID[i];
	for (int cndx = cfirst; cndx < clast; cndx++)
	{
		j = buf.mgrid[cndx];
		//if (buf.MFtype[j] == buf.MFtype[i])
		if(buf.MFtype[j] <= 1)
			continue;
		dist = (buf.mpos[i] - buf.mpos[j]);		// dist in cm
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
		if (!(dsq < r2 && dsq > 0))
			continue;
		dist *= simData.psimscale;
		dsq = sqrt(dsq*d2);
		c = (simData.psmoothradius - dsq);
		cmterm = c*c*simData.spikykern*buf.mf_restmass[j] * buf.density_solid[j];
		pmterm = dist / dsq*cmterm;
		for (int k = 1; k < simData.mf_catnum; ++k)
		{
			//cmterm1 = simData.CoCompressibility * simData.rest_porosity - buf.pressure_water[j*MAX_FLUIDNUM + k];
			cmterm1 = buf.pressure_water[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + buf.MFtype[j] - 2] 
				- buf.pressure_water[j*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + buf.MFtype[j] - 2];

			buf.gradPressure[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + buf.MFtype[j] - 2] += 
				(buf.mf_alpha[i*MAX_FLUIDNUM+k] + 
					buf.mf_beta[j*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + buf.MFtype[j] - 2])
				*cmterm1*pmterm;
		}
		normalize += cmterm;
	}
}
__global__ void ComputeFluidFlux(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum)
		return;
	//if (buf.misbound[i])
	//	return;
	if (buf.MFtype[i] != 0)
		return;
	
	//if (buf.MFtype[i] == 1 && buf.isSurface[buf.elasticID[i]]!=1)
	//	return;
	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) return;						// particle out-of-range
	gc -= nadj;
	float normalize = 0;
	
	for(int k=1;k<simData.mf_catnum;++k)
		for (int l = 0; l<MAX_SOLIDNUM; ++l)
			buf.gradPressure[i*MAX_FLUIDNUM*MAX_SOLIDNUM+k*MAX_SOLIDNUM+l] = make_float3(0, 0, 0);
	
	for (int c = 0; c < simData.gridAdjCnt; c++)
	{
		contributeFluidFlux(i, gc + simData.gridAdj[c], buf, normalize);
	}
	//if(normalize !=0)
	for (int k = 1; k<simData.mf_catnum; ++k)
		for (int l = 0; l < MAX_SOLIDNUM; ++l)
			//buf.gradPressure[i*MAX_FLUIDNUM + k] *= simData.mf_permeability[k] / (simData.mf_visc[k]*abs(normalize));
		{
			buf.gradPressure[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l]
				*= simData.capillary*simData.mf_permeability[k*MAX_SOLIDNUM + l] / simData.mf_visc[k];
			//if (dot(buf.gradPressure[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l], buf.gradPressure[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l]) != 0)
			//	printf("%d's phase %d  %d's grad pressure is (%f,%f,%f)\n", i, k, l,
			//		buf.gradPressure[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l].x, buf.gradPressure[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l].y,
			//		buf.gradPressure[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l].z);
		}
	//if (isnan(dot(buf.gradPressure[i], buf.gradPressure[i])))
	//if(dot(buf.gradPressure[i*MAX_FLUIDNUM + 1], buf.gradPressure[i*MAX_FLUIDNUM + 1])!=0&&i%100==0)
	//	printf("particle %d's type is %d, grad pressure is (%f,%f,%f)\n",
	//		i, buf.MFtype[i], buf.gradPressure[i*MAX_FLUIDNUM + 1].x, buf.gradPressure[i*MAX_FLUIDNUM + 1].y, buf.gradPressure[i*MAX_FLUIDNUM + 1].z
	//		);

}
__device__ void contributeSolidDarcyFlux(int i, int cell, bufList buf)
{
	float dsq, c;
	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2 / d2;

	float3 dist, vmr;
	float cmterm, cmterm1;
	//	float massj;
	float3 pmterm, vmterm;
	//	float q;
	int j, mulj;
	float aveDenij, cx, xterm;

	if (buf.mgridcnt[cell] == 0) return;

	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];
	
	for (int cndx = cfirst; cndx < clast; cndx++)
	{
		j = buf.mgrid[cndx];
		//buf.MFtype[j]<=1
		if (buf.MFtype[j] != buf.MFtype[i])
			continue;
		dist = (buf.mpos[i] - buf.mpos[j]);		// dist in cm
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
		if (!(dsq < r2 && dsq > 0))
			continue;
		dist *= simData.psimscale;
		dsq = sqrt(dsq*d2);
		c = (simData.psmoothradius - dsq);
		//if (buf.MFtype[i] == 1)
		cmterm = c*c*simData.spikykern*buf.mf_restmass[j] * buf.density_solid[i];
		//else
		//	cmterm = c*c*simData.spikykern*buf.mf_restmass[j] * buf.mdensity[i];
		pmterm = dist / dsq*cmterm;
		for (int k = 1; k<simData.mf_catnum; ++k)
			buf.gradPressure[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + buf.MFtype[i]-2] -= 
				(buf.pressure_water[j*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + buf.MFtype[i] - 2]
				- buf.pressure_water[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + buf.MFtype[i] - 2])*pmterm;
		//normalize += cmterm;
	}
}
__global__ void ComputeSolidDarcyFlux(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum)
		return;
	if (buf.MFtype[i] <= 1)
		return;
	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) return;						// particle out-of-range
	gc -= nadj;
	for (int k = 1; k<simData.mf_catnum; ++k)
		buf.gradPressure[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + buf.MFtype[i] - 2] = make_float3(0, 0, 0);
	for (int c = 0; c < simData.gridAdjCnt; c++)
	{
		contributeSolidDarcyFlux(i, gc + simData.gridAdj[c], buf);
	}

	for (int k = 1; k < simData.mf_catnum; ++k)
	{
		//poro velocity
		buf.gradPressure[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM+buf.MFtype[i]-2] 
			*= simData.mf_permeability[k*MAX_SOLIDNUM+ buf.MFtype[i] - 2] / (simData.mf_visc[k] * simData.rest_porosity);
		//buf.gradPressure[i*MAX_FLUIDNUM + k] += buf.mveleval[i];
	}
}
__device__ void contributeFluidChange(int i, int cell, bufList buf)
{
	float dsq, c;
	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2;
	//int jndex;
	float3 dist, vmr;
	float cmterm, cmterm1;
	//	float massj;
	float pmterm, vmterm;
	//	float q;
	int j, mulj;
	float aveDenij, cx, xterm;
	float sum = 0;
	//if (i % 100 == 0)
	//	printf("particle %d's gridcnt is %d\n", i,buf.mgridcnt[cell]);
	if (buf.mgridcnt[cell] == 0) return;

	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];
	//int index = buf.elasticID[i];
	//int jndex,index = buf.elasticID[i];
	for (int cndx = cfirst; cndx < clast; cndx++)
	{
		j = buf.mgrid[cndx];
		if (buf.MFtype[j] <= 1)
			continue;
		//jndex = buf.elasticID[j];
		dist = (buf.mpos[i] - buf.mpos[j])*simData.psimscale;		// dist in cm
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
		if (!(dsq < r2 && dsq > 0))
			continue;
		dsq = sqrt(dsq);
		c = simData.psmoothradius - dsq;
		cmterm = c*c*simData.spikykern * buf.mf_restmass[j] * buf.density_solid[j];
		for(int k=1;k<simData.mf_catnum;++k)
			buf.divDarcyFlux[i*MAX_FLUIDNUM*MAX_SOLIDNUM+k*MAX_SOLIDNUM+buf.MFtype[j]-2] += 
				dot(buf.gradPressure[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + buf.MFtype[j] - 2], dist / dsq)*cmterm;
		//normalize += cmterm;
	}
	return;
}
__global__ void ComputeFluidChange(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;

	if (buf.MFtype[i] != 0)
		return;

	int gc = buf.mgcell[i];
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	if (gc == GRID_UNDEF) return;
	gc -= nadj;
	for (int k = 0; k<simData.mf_catnum; ++k)
		for(int l=0;l<MAX_SOLIDNUM;++l)
			buf.divDarcyFlux[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l] = 0;
	for (int c = 0; c < simData.gridAdjCnt; c++)
	{
		contributeFluidChange(i, gc + simData.gridAdj[c], buf);
	}

	for (int k = 1; k < simData.mf_catnum; ++k) 
	{
		for (int l = 0; l < MAX_SOLIDNUM; ++l) 
		{
			buf.divDarcyFlux[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l] *= simData.mf_dt;
			if (buf.mf_beta[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l] == 0)
				buf.poroVel[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l] = make_float3(0, 0, 0);

			if (buf.divDarcyFlux[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l] > 0)
			{
				if (buf.mf_alpha[i*MAX_FLUIDNUM + k] - buf.divDarcyFlux[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l] < 0.001)
				{
					buf.divDarcyFlux[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l] = buf.mf_alpha[i*MAX_FLUIDNUM + k];
				}
			}
			if (buf.divDarcyFlux[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l] < 0)
			{
				if (buf.mf_beta[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l] + buf.divDarcyFlux[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l] < 0.001)
					buf.divDarcyFlux[i*MAX_FLUIDNUM *MAX_SOLIDNUM + k*MAX_SOLIDNUM + l] = -buf.mf_beta[i*MAX_FLUIDNUM *MAX_SOLIDNUM + k*MAX_SOLIDNUM + l];
			}
			/*if (buf.divDarcyFlux[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l] < 0)
				continue;*/
			buf.mf_beta_next[i*MAX_FLUIDNUM *MAX_SOLIDNUM + k*MAX_SOLIDNUM + l] += buf.divDarcyFlux[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l];
			buf.mf_alpha_next[i*MAX_FLUIDNUM + k] -= buf.divDarcyFlux[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l];
			//if (isnan(buf.divDarcyFlux[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l]))
			//if(buf.divDarcyFlux[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l]!=0)
			//	printf("particle %d's phase %d's div Darcy flux is %f, darcy flux is (%f,%f,%f)\n",
			//		i, k, buf.divDarcyFlux[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l],
			//		buf.gradPressure[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l].x,
			//		buf.gradPressure[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l].y, 
			//		buf.gradPressure[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l].z);
		}
	}

	//if (buf.mf_alpha[i*MAX_FLUIDNUM + 1] < buf.mf_alpha[i*MAX_FLUIDNUM + 2]-0.1&&!buf.isInside[i])
	//	printf("particle %d's alpha is (%f,%f), beta is (%f,%f), divDarcyFlux is (%f,%f)\n",
	//		i, buf.mf_alpha[i*MAX_FLUIDNUM + 1], buf.mf_alpha[i*MAX_FLUIDNUM + 2],
	//		buf.mf_beta[i*MAX_FLUIDNUM + 1], buf.mf_beta[i*MAX_FLUIDNUM + 2],
	//		buf.divDarcyFlux[i*MAX_FLUIDNUM + 1], buf.divDarcyFlux[i*MAX_FLUIDNUM + 2]);
}
__device__ void contributeFluidAdvance(int i, int cell, bufList buf, float3*gradBeta, float*DivVelocity)
{
	float dsq, c, dsq2;
	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2;

	float3 dist;
	float cmterm;
	float pmterm;
	int j;
	if (buf.mgridcnt[cell] == 0) return;

	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];
	
	for (int cndx = cfirst; cndx < clast; cndx++) {
		j = buf.mgrid[cndx];
		if (buf.MFtype[j] != 0)
			continue;
		dist = (buf.mpos[i] - buf.mpos[j])*simData.psimscale;		// dist in cm
		dsq = dot(dist, dist);
		if (!(dsq < r2 && dsq > 0))
			continue;
		dsq = sqrt(dsq);
		c = simData.psmoothradius - dsq;
		for (int k = 1; k < simData.mf_catnum; ++k) 
		{
			cmterm = c*c*simData.spikykern * buf.mf_restmass[j] * buf.mdensity[j];
			for (int l = 0; l < MAX_SOLIDNUM; ++l) 
			{
				DivVelocity[k*MAX_SOLIDNUM + l] += cmterm *
					dot((buf.mf_beta[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM+l] * buf.poroVel[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM+l] +
						buf.mf_beta[j*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM+l] * buf.poroVel[j*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM+l]), dist);
				gradBeta[k*MAX_SOLIDNUM+l] += cmterm * (buf.mf_beta[j*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM+l] - buf.mf_beta[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM+l])*dist;
			}
		}
	}
	return;
}
__global__ void ComputeFluidAdvance(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum)
		return;
	if (buf.MFtype[i] != 0)
		return;
	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) return;						// particle out-of-range
	gc -= nadj;
	float3 gradBeta[MAX_FLUIDNUM*MAX_SOLIDNUM];
	float DivVelocity[MAX_FLUIDNUM*MAX_SOLIDNUM],betachange[MAX_FLUIDNUM*MAX_SOLIDNUM];
	float sigma = 1;
	for (int k = 1; k < simData.mf_catnum; ++k)
	{
		for (int l = 0; l < MAX_SOLIDNUM; ++l) 
		{
			gradBeta[k*MAX_SOLIDNUM + l] = make_float3(0, 0, 0);
			DivVelocity[k*MAX_SOLIDNUM + l] = 0;
		}
	}
	for (int c = 0; c < simData.gridAdjCnt; c++)
	{
		contributeFluidAdvance(i, gc + simData.gridAdj[c], buf, gradBeta, DivVelocity);
	}
	//float betasum = 0;
	for (int k = 1; k < simData.mf_catnum; ++k) 
	{
		for (int l = 0; l < MAX_SOLIDNUM; ++l) 
		{
			betachange[k*MAX_SOLIDNUM+l] = sigma*simData.mf_dt*(-DivVelocity[k*MAX_SOLIDNUM + l] + dot(buf.mveleval[i], gradBeta[k*MAX_SOLIDNUM + l]));
			/*if (abs(betachange[k]) >= 0.0001)
				printf("error! particle %d's beta change is (%f,%f)\n",
					i, betachange[1], betachange[2]);*/
					//betachange limit
			if (betachange[k*MAX_SOLIDNUM + l] < -0.99)
			{
				betachange[k*MAX_SOLIDNUM + l] = -0.99;// * ((int)(buf.mf_alpha[muloffseti+fcount]>0)-(int)(buf.mf_alpha[muloffseti+fcount]<0));
			}
			//betasum += buf.mf_beta_next[i*MAX_FLUIDNUM + k];
			buf.mf_beta_next[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l] += betachange[k*MAX_SOLIDNUM + l];
		}
	}

	//if (i % 10000 == 0 && buf.solidCount[i]!=0)
	//	printf("particle %d's beta change is (%f,%f)\n",
	//		i, betachange[1], betachange[2]);
}

__device__ float3 contributeCapillaryForce(int i, int cell, bufList buf)
{
	float dsq, c, dsq2;
	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2;

	float3 dist;
	float cmterm;
	float3 pmterm;
	int j, jndex;
	float3 sum = make_float3(0,0,0);
	if (buf.mgridcnt[cell] == 0) return sum;

	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];

	//float kernel, betasum, kparm = 0.007 / pow(simData.psmoothradius, (float)(3.25));
	float kernel, betasum, kparm = 8 / (3.1415926*pow(simData.psmoothradius, 3));
	float q;
	for (int cndx = cfirst; cndx < clast; cndx++)
	{
		j = buf.mgrid[cndx];
		if (buf.MFtype[j] > 1)
		{
			dist = (buf.mpos[i] - buf.mpos[j])*simData.psimscale;		// dist in cm
			dsq2 = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
			if (!(dsq2 < r2 && dsq2 > 0))
				continue;
			q = sqrt(dsq2 / r2);
			//if (q > 1||q==0)
			//	continue;
			if (q <= 0.5)
				kernel = 3*q*q-2*q;
			else
				kernel = -pow(1-q,2);
			//kernel *= kparm;
			dsq = sqrt(dsq2);
			//c = simData.psmoothradius - dsq;
			betasum = 0;
			for (int k = 1; k < simData.mf_catnum; ++k)
				betasum += buf.mf_beta[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + buf.MFtype[j] - 2];
			sum += betasum*buf.mf_restmass[j] * buf.density_solid[j] * kernel *simData.gradCubicSplineKern * dist / dsq;
			//betasum = 1;
			//sum += -betasum * buf.mf_restdensity[i] * buf.density_solid[j] * dist / dsq * kernel;
			//sum += betasum*buf.mf_restmass[j] * buf.density_solid[j] * c*c *simData.spikykern * dist / dsq;

			//dsq = sqrt(dsq2);

			//if (2 * dsq > simData.psmoothradius)
			//	continue;
			//c = simData.psmoothradius - dsq;
			//betasum = 0;
			//for (int k = 1; k < simData.mf_catnum; ++k)
			//	betasum += buf.mf_beta[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + buf.MFtype[j] - 2];
			////sum += betasum * buf.mf_restmass[j] * c*c *simData.spikykern * dist / dsq;
			//	//betasum += buf.mf_alpha[i*MAX_FLUIDNUM + k] * simData.mf_dens[k];
			//betasum = 1;

			//kernel = pow((float)(2 * dsq - 4 * dsq*dsq / simData.psmoothradius), (float)0.25);
			////kernel = sqrt(sqrt(6 * dsq - 2 * simData.psmoothradius - 4 * dsq*dsq / simData.psmoothradius));
			//kernel *= kparm;
			//sum += -betasum * buf.mf_restdensity[i] * buf.density_solid[j] * dist/dsq * kernel;


		}
	}
	return sum;
}
__global__ void ComputeCapillaryForce(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum)
		return;
	if (buf.MFtype[i] != 0)
		return;

	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) return;						// particle out-of-range
	gc -= nadj;
	float colorField = 0;
	float3 normal = make_float3(0, 0, 0);
	for (int c = 0; c < simData.gridAdjCnt; c++)
	{
		normal += simData.capillaryForceRatio * contributeCapillaryForce(i, gc + simData.gridAdj[c], buf);
	}

	if ( isnan(dot(normal, normal)))
		printf("capillary force is (%f,%f,%f)\n", normal.x, normal.y, normal.z);
	//colorField = dot(normal, normal);
	//if (colorField > simData.AL2) {
	//	normal *= simData.AL / sqrt(colorField);
	//}
	buf.mforce[i] += normal;
	buf.poroForce[i] += normal;
	buf.maccel[i] = buf.mforce[i];
	
}
__device__ float3 contributeInnerBoundaryForce(int i, int cell, bufList buf, float betasum, float kparm)
{
	float dsq, c, dsq2;
	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2;

	float3 dist;
	float cmterm;
	float3 pmterm;
	int j, jndex;
	float3 sum = make_float3(0, 0, 0);
	if (buf.mgridcnt[cell] == 0) return sum;

	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];

	for (int cndx = cfirst; cndx < clast; cndx++)
	{
		j = buf.mgrid[cndx];
		if (buf.MFtype[j] == 1)
		{
			dist = (buf.mpos[i] - buf.mpos[j])*simData.psimscale;		// dist in cm
			dsq2 = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
			//if (!(dsq2 < r2 && dsq2 > 0))
			//	continue;
			dsq = sqrt(dsq2);
			if (2 * dsq >= simData.psmoothradius)
				continue;
			cmterm = 0.5*buf.mf_visc[i]*simData.psmoothradius*buf.mdensity[i];
			cmterm *= (max((float)0, dot(dist, -buf.mveleval[i]))) / (0.01*r2 + dsq2)*buf.density_solid[j];

			//c = (simData.psmoothradius - dsq);
			//if (buf.MFtype[i] == 1)
			//cmterm *= c*c*simData.spikykern;
			//if (2 * dsq - 4 * dsq2 / simData.psmoothradius < 0)
			//	continue;
			cmterm *= kparm*pow(2 * dsq - 4 * dsq2 / simData.psmoothradius, (float)0.25);
			//if (isnan(cmterm))
			//	continue;
			sum += betasum*cmterm * dist / dsq;

		}
	}
	return sum;
}
__global__ void ComputeInnerBoundaryForce(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum)
		return;
	if (buf.MFtype[i] == 0 || buf.misbound[i])
		return;

	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) return;						// particle out-of-range
	gc -= nadj;
	float colorField = 0;
	float betasum = 0;
	for (int k = 1; k < simData.mf_catnum; ++k)
		for(int l=1;l<=3;++l)
			betasum += buf.mf_beta[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + l];
	//betasum = 1;
	if (betasum < 0.001)
		return;
	float kparm = 0.007 / pow(simData.psmoothradius, (float)(3.25));
	//printf("beta sum%f\n", betasum);
	float3 normal = make_float3(0, 0, 0);
	for (int c = 0; c < simData.gridAdjCnt; c++)
	{
		normal +=  simData.stRatio*contributeInnerBoundaryForce(i, gc + simData.gridAdj[c], buf, betasum, kparm);
	}

	if (isnan(dot(normal,normal)))
		printf("inner boundary force is (%f,%f,%f)\n", normal.x, normal.y, normal.z);
	//colorField = dot(normal, normal);
	//if (colorField > simData.AL2) {
	//	normal *= simData.AL / sqrt(colorField);
	//}
	
	buf.mforce[i] += normal;
	buf.poroForce[i] += normal;
	buf.maccel[i] = buf.mforce[i];

}
__device__ float3 contributeSurfaceTension2(int i, int cell, bufList buf)
{
	float dsq, c, dsq2;
	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2;

	float3 dist;
	float cmterm;
	int j, jndex;
	float3 sum = make_float3(0, 0, 0);
	if (buf.mgridcnt[cell] == 0) return sum;

	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];

	for (int cndx = cfirst; cndx < clast; cndx++)
	{
		j = buf.mgrid[cndx];
		if (buf.MFtype[j] > 1)
		{
			dist = (buf.mpos[i] - buf.mpos[j])*simData.psimscale;		// dist in cm
			dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
			if (!(dsq < r2 && dsq > 0))
				continue;
			dsq = sqrt(dsq);
			c = simData.psmoothradius - dsq;
			cmterm = buf.mf_restmass[j] * buf.density_solid[j] * c*c / dsq*simData.spikykern;

			//sum += (buf.pressure_water[i*MAX_FLUIDNUM] - buf.pressure_water[j*MAX_FLUIDNUM])*cmterm;
			sum += (buf.pressure_water[i*MAX_FLUIDNUM])*cmterm;
		}
	}
	return sum;
}
__global__ void ComputeSurfaceTension2(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum)
		return;
	if (buf.MFtype[i] != 0)
		return;

	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) return;						// particle out-of-range
	gc -= nadj;
	float colorField = 0;
	float3 normal = make_float3(0, 0, 0);
	float mor = 2/simData.CoCompressibility;
	//float mor = 0.002;
	for (int c = 0; c < simData.gridAdjCnt; c++)
	{
		normal += mor * contributeSurfaceTension2(i, gc + simData.gridAdj[c], buf);
	}
	buf.mforce[i] += normal / buf.mf_restdensity[i];
	//buf.poroForce[i] += (buf.mf_beta[i*MAX_FLUIDNUM + 1] + buf.mf_beta[i*MAX_FLUIDNUM + 2])*normal;
	buf.maccel[i] = buf.mforce[i];

}

//capillary force exert on fluid particles
void ComputePorousForceCUDA()
{
	cudaError_t error;
	ComputeSolidDarcyFlux << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: compute poro velocity CUDA: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();

	ComputePoroVelocity << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: compute poro velocity CUDA: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();

	//if(fcuda.example == 11)
	//	ComputeSurfaceTension2 << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	//else
	ComputeCapillaryForce << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: compute surface tension CUDA: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();

	//if (fcuda.example != 6)
	//{
		ComputeInnerBoundaryForce << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
		error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf(stderr, "CUDA ERROR: compute surface tension CUDA: %s\n", cudaGetErrorString(error));
		}
		cudaThreadSynchronize();
	//}

	//fluid flow between fluids and solid surface
	ComputeFluidFlux << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: compute fluid flux CUDA: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();

	ComputeFluidChange << < fcuda.numBlocks, fcuda.numThreads >> > (fbuf, fcuda.pnum);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: compute fluid flux CUDA: %s\n", cudaGetErrorString(error));
	}
	cudaThreadSynchronize();
}

//**************************************************************************************************
//implicit incompressible SPH
__device__ float3 contributePressureForce(int i,float3 pos,int cell, bufList buf, int& count)
{
	float3 force = make_float3(0, 0, 0);
	if (buf.mgridcnt[cell] == 0)return force;
	
	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2;
	float3 dist;
	float c, dsq2, dsq;
	int j;
	float3 vmr;
	float cmterm;
	float3 vmterm;
	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];
	float q;
	for(int cndx = cfirst;cndx < clast;cndx++)
	{
		j = buf.mgrid[cndx];
		//if (buf.MFtype[i] != buf.MFtype[j] && (!buf.misbound[i] && !buf.misbound[j]))
		//	continue;
		/*if (buf.MFtype[i] == 1 && buf.MFtype[i] == buf.MFtype[j])
			continue;*/
		//if (buf.MFtype[i] == buf.MFtype[j] && buf.MFtype[i] == 1)
		//	continue;
		dist = (buf.mpos[i] - buf.mpos[j])*simData.psimscale;
		dsq2 = dot(dist, dist);
		dsq = sqrt(dsq2);

		if (dsq2 > r2 || dsq2 <= 0)
			continue;
		//if (buf.MFtype[i] == 1 && buf.MFtype[i] == buf.MFtype[j])
		//	continue;
		count++;
		c = simData.psmoothradius - dsq;
		//cmterm = buf.mf_restmass[j] * (buf.mpress[i] * pow(buf.mdensity[i], 2) + buf.mpress[j] * pow(buf.mdensity[j], 2));
		//force -= cmterm *c*c*dist*simData.spikykern/dsq;
		
		//force += buf.volume[j]*c*c*simData.spikykern*dist / dsq*(buf.mpress[i] + buf.mpress[j]);
		//pairwise pressure force
		if(buf.volume[j] * buf.volume[i]!=0)
			force += c*c*simData.spikykern*dist / dsq*buf.volume[j]* buf.volume[i]*(buf.mpress[j] + buf.mpress[i])/(buf.volume[i]+ buf.volume[j]);
	}
	return force;
}
//fluid pressure force
__global__ void ComputePressureForce(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;
	//if (i % 30000 == 0)
	//	printf("particle %d's type is %d,  press is %.10f\n",
	//		i, buf.MFtype[i], buf.mpress[i]);
	if (buf.misbound[i])
	{
		buf.mforce[i] = make_float3(0, 0, 0);
		buf.maccel[i] = buf.mforce[i];
		buf.pressForce[i] = make_float3(0, 0, 0);
		return;
	}
	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) return;						// particle out-of-range
	gc -= nadj;
	// Sum Pressures
	float3 pos = buf.mpos[i];
	//float dens = buf.mf_restdensity[i];
	float3 force = make_float3(0, 0, 0);
	int count = 0;
	for (int c = 0; c < simData.gridAdjCnt; c++)
	{
		force += contributePressureForce(i, pos, gc + simData.gridAdj[c], buf, count);
	}
	//if (isnan(dot(force, force)))
	//if(isnan(buf.volume[i])||isnan(buf.mpress[i]))
	//	printf("particle %d's type is %d, force is nan. press is %f, volume is %.10f,fluid percent is %f\n",
	//		i, buf.MFtype[i], buf.mpress[i], buf.volume[i], buf.fluidPercent[i]);
	if(buf.MFtype[i] == 0)
		buf.pressForce[i] = -buf.volume[i]/buf.mf_restmass[i]*force;
	else
	{
		float mass = buf.mf_restmass[i];
		for (int k = 1; k < MAX_FLUIDNUM; ++k)
			mass += buf.mf_beta[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + buf.MFtype[i] - 2] * simData.mf_mass[k];
		buf.pressForce[i] = -buf.volume[i] / mass * force;
	}
	//if (dot(buf.mforce[i], buf.mforce[i]) > 10000)
	//	printf("particle %d's type is %d, pressure force is (%f,%f,%f), pressure is %f\n",
	//		i, buf.MFtype[i], buf.mforce[i].x, buf.mforce[i].y, buf.mforce[i].z,
	//		buf.mpress[i]);
	
	//if(isnan(dot(buf.mforce[i],buf.mforce[i])))
	//if (dot(buf.mforce[i],buf.mforce[i])>10 && !buf.misbound[i])
	//	printf("particle %d's type is %d, pressure force is (%.10f,%.10f,%.10f),count is %d, press is %.10f, aii is %.10f, deltadensity is %.10f, rest mass is %.10f, volume is %.10f\n",
	//		i, buf.MFtype[i], buf.mforce[i].x, buf.mforce[i].y, buf.mforce[i].z, count, buf.mpress[i],buf.aii[i], buf.delta_density[i],buf.mf_restmass[i],buf.volume[i]);
	//if (i % 30000 == 0)
	//	printf("volume is %.10f, m/rho is %.10f\n", buf.volume[i], buf.mf_restmass[i] * buf.mdensity[i]);
}
//fluid pressure force
__global__ void ApplyPressureForce(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;

	if (buf.misbound[i])
	{
		buf.mforce[i] = make_float3(0, 0, 0);
		buf.maccel[i] = buf.mforce[i];
		return;
	}
	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) return;						// particle out-of-range
	gc -= nadj;

	/*for (uint fcount = 0; fcount<simData.mf_catnum; fcount++)
	{
		buf.mf_alphagrad[i*MAX_FLUIDNUM + fcount] = make_float3(0, 0, 0);
		buf.mf_alpha[i*MAX_FLUIDNUM + fcount] = buf.mf_alpha_next[i*MAX_FLUIDNUM + fcount];
		buf.mf_beta[i*MAX_FLUIDNUM + fcount] = buf.mf_beta_next[i*MAX_FLUIDNUM + fcount];
	}*/
	// Sum Pressures
	float3 pos = buf.mpos[i];
	//float dens = buf.mf_restdensity[i];
	float3 force = make_float3(0, 0, 0);
	int count = 0;
	for (int c = 0; c < simData.gridAdjCnt; c++)
	{
		force += contributePressureForce(i, pos, gc + simData.gridAdj[c], buf, count);
	}
	/*if(i%10000==0)
	printf("particle %d's type is %d,source is %f, aii is %.10f,press is %f, vel is (%f,%f,%f),volume is %.10f,rest volume is %.10f,press force is (%f,%f,%f),alpha is (%f,%f,%f),beta is (%f,%f,%f)\n",
		i, buf.MFtype[i], buf.source[i], buf.aii[i], buf.mpress[i],
		buf.vel_mid[i].x, buf.vel_mid[i].y, buf.vel_mid[i].z,
		buf.volume[i], buf.rest_volume[i], buf.pressForce[i].x, buf.pressForce[i].y, buf.pressForce[i].z,
		buf.mf_alpha[i*MAX_FLUIDNUM + 0], buf.mf_alpha[i*MAX_FLUIDNUM + 1], buf.mf_alpha[i*MAX_FLUIDNUM + 2],
		buf.mf_beta[i*MAX_FLUIDNUM + 0], buf.mf_beta[i*MAX_FLUIDNUM + 1], buf.mf_beta[i*MAX_FLUIDNUM + 2]);
*/
	buf.pressForce[i] = -buf.volume[i] / buf.mf_restmass[i] * force;
	if(buf.MFtype[i] == 0)
		buf.mforce[i] += -buf.volume[i] / buf.mf_restmass[i] * force;
	else
	{
		float mass = buf.mf_restmass[i];
		for (int k = 1; k < MAX_FLUIDNUM; ++k)
			mass += buf.mf_beta[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + buf.MFtype[i] - 2] * simData.mf_mass[k];
			//if (buf.mf_beta[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + buf.MFtype[i] - 2] != 0)
			//	printf("type %d's fluid beta %d is %f\n", buf.MFtype[i] - 2, k, buf.mf_beta[i*MAX_FLUIDNUM*MAX_SOLIDNUM + k*MAX_SOLIDNUM + buf.MFtype[i] - 2]);
		buf.mforce[i] += -buf.volume[i] / mass * force;
	}
	buf.fluidForce[i] += -buf.volume[i] / buf.mf_restmass[i] * force;
	//if (dot(buf.mforce[i], buf.mforce[i]) > 10000)
	//	printf("particle %d's type is %d, pressure force is (%f,%f,%f), pressure is %f\n",
	//		i, buf.MFtype[i], buf.mforce[i].x, buf.mforce[i].y, buf.mforce[i].z,
	//		buf.mpress[i]);
	buf.maccel[i] = buf.mforce[i];

	if(isnan(dot(buf.mforce[i],buf.mforce[i])))
	//if (dot(buf.mforce[i],buf.mforce[i])>10 && !buf.misbound[i])
		printf("particle %d's type is %d, pressure force is (%.10f,%.10f,%.10f),count is %d, press is %.10f, aii is %.10f, deltadensity is %.10f, rest mass is %.10f, volume is %.10f\n",
			i, buf.MFtype[i], buf.mforce[i].x, buf.mforce[i].y, buf.mforce[i].z, count, buf.mpress[i],buf.aii[i], buf.delta_density[i],buf.mf_restmass[i],buf.volume[i]);
	//if (i % 30000 == 0)
	//	printf("volume is %.10f, m/rho is %.10f\n", buf.volume[i], buf.mf_restmass[i] * buf.mdensity[i]);
}
__device__ float3 contributeViscosity(int i, int muli, float idens, float3 pos, int cell, bufList buf, float* ialpha_pre, float3* ivmk)
{
	float3 force = make_float3(0, 0, 0);
	if (buf.mgridcnt[cell] == 0)return force;

	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2;
	float3 dist, sf;
	float c, dsq2, dsq;
	int j, mulj;
	float3 vmr;
	float cmterm, vmterm;
	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];

	float xvprod, phiij, densityij, PIij, q;
	float3 fP;
	float cmterm1, vmterm1;
	for (int cndx = cfirst; cndx < clast; cndx++)
	{
		j = buf.mgrid[cndx];
		if (buf.misbound[j])
			continue;
		mulj = j * MAX_FLUIDNUM;
		dist = (buf.mpos[i] - buf.mpos[j])*simData.psimscale;
		dsq2 = dot(dist, dist);
		if (dsq2 <= 0 || dsq2 >= r2)
			continue;
		dsq = sqrt(dsq2);
		q = sqrt(dsq2 / r2);
		vmr = buf.mveleval[i] - buf.mveleval[j];
		if (q <= 0.5)
			cmterm = simData.gradCubicSplineKern * (3 * q*q - 2 * q);
		else
			cmterm = -simData.gradCubicSplineKern * pow(1 - q, 2);
		cmterm *= buf.mf_restmass[j] * buf.density_solid[j] * dot(vmr, dist) / (dsq2 + 0.01 * r2);
		vmterm = cmterm * (buf.mf_visc[i] + buf.mf_visc[j]);

		//viscosity
		c = (simData.psmoothradius - dsq);
		cmterm1 = simData.spikykern * c * c / dsq * buf.mf_restmass[j] * buf.density_solid[j];
		vmterm1 = cmterm1 * (buf.mf_visc[i] + buf.mf_visc[j]) * idens;
		//if ((buf.MFtype[i] == 4 && buf.MFtype[j] == 4))
		//	force += vmterm1 * vmr;

		if (buf.MFtype[i] == buf.MFtype[j])
		{
			if (buf.MFtype[i] != 0)
				force += vmterm1 * vmr;
			
			if (buf.MFtype[i] == 0)
			{
				float fluidsum = buf.mf_alpha_sum[i] * buf.mf_alpha_sum[j];
				if (fluidsum <= 0.01)
					fluidsum = 0;
				else
					fluidsum /= (buf.mf_alpha_sum[i] + buf.mf_alpha_sum[j]);
				float fluidsum2 = (1 - buf.mf_alpha_sum[i])*(1 - buf.mf_alpha_sum[j]);
				if (fluidsum2 <= 0.01)
					fluidsum2 = 0;
				else
					fluidsum2 /= (2 - buf.mf_alpha_sum[i] - buf.mf_alpha_sum[j]);
				//if (_example == 2)
				//	fluidsum2 = 0;
				//force += (fluidsum + fluidsum2) * vmterm * dist / dsq;
				force += (fluidsum + fluidsum2) * vmterm1 * vmr;
			}
		}
		//else
		//{
		//	float fluidsum = 0;
		//	if (buf.MFtype[i] == 0)
		//		fluidsum = buf.mf_alpha_sum[i];
		//	if (buf.MFtype[j] == 0)
		//		fluidsum = buf.mf_alpha_sum[j];
		//	force += fluidsum * vmterm1 * vmr;
		//}
		if(buf.MFtype[i] + buf.MFtype[j] == 9)
			force += vmterm1 * vmr;
	}
	return force;
}
__global__ void ComputeOtherForce(bufList buf, int pnum, float time)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;
	if (buf.misbound[i])return;
	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) return;						// particle out-of-range
	gc -= nadj;
	bool error = false;
	// Sum Pressures
	float3 pos = buf.mpos[i];
	float dens = buf.mf_restdensity[i];
	float3 force = make_float3(0, 0, 0);
	float normalize = 0;

	register uint muloffseti = i * MAX_FLUIDNUM;
	
	register float alpha[MAX_FLUIDNUM];
	register float3 ivmk[MAX_FLUIDNUM];
	
	for (uint fcount = 0; fcount < simData.mf_catnum; fcount++)
	{
		//buf.mf_alphagrad[i*MAX_FLUIDNUM + fcount] = make_float3(0, 0, 0);
		alpha[fcount] = buf.mf_alpha_next[muloffseti + fcount];
		//buf.mf_alpha_pre[i*MAX_FLUIDNUM + fcount] = buf.mf_alpha[i*MAX_FLUIDNUM + fcount];
		ivmk[fcount] = buf.mf_vel_phrel[muloffseti + fcount];
	}

	for (int c = 0; c < simData.gridAdjCnt; c++)
	{
		force += contributeViscosity(i, muloffseti, buf.mdensity[i], pos, gc + simData.gridAdj[c], buf, alpha, ivmk);

	}
	//if (dot(force, force) > 10)
	//	printf("particle %d's viscosity force is (%f,%f,%f)\n",
	//		i, force.x, force.y, force.z);
	//bound force and gravity
	//buf.mforce[i] += getBoundForce(i, buf, force, time);
	buf.mforce[i] += force;
	buf.fluidForce[i] = force;
	buf.maccel[i] = buf.mforce[i];
	/*if (buf.MFtype[i] == 0)
	{
		buf.mforce[i] *= 1-buf.absorbedPercent[i];
		buf.maccel[i] *= 1-buf.absorbedPercent[i];
	}*/
	if (isnan(dot(force,force)))
		printf("particle %d's type is %d,visco force is (%f,%f,%f),pos is (%f,%f,%f), alpha sum is %f\n",
		i, buf.MFtype[i], buf.mforce[i].x, buf.mforce[i].y, buf.mforce[i].z,
		buf.mpos[i].x, buf.mpos[i].y, buf.mpos[i].z, buf.mf_alpha_sum[i]);
}
__device__ float contributeColorValue(int i, float3 pos, int cell, bufList buf)
{
	if (buf.mgridcnt[cell] == 0)return 0;
	float sum = 0;
	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2;
	float3 dist;
	float c, dsq2, dsq;
	int j, mulj;
	float pmterm;
	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];
	float q;
	for (int cndx = cfirst; cndx < clast; cndx++)
	{
		j = buf.mgrid[cndx];
		if (buf.MFtype[j] == 2 || (_example == 2 && buf.MFtype[j] >= 2))
			continue;
		mulj = j * MAX_FLUIDNUM;
		dist = (buf.mpos[i] - buf.mpos[j])*simData.psimscale;
		dsq2 = dot(dist, dist);
		q = sqrt(dsq2 / r2);
		if (q>2)
			continue;
		if (q >= 0 && q <= 1)
			pmterm = simData.CubicSplineKern2*(1 - 1.5*q*q*(1 - q / 2));
		else
			pmterm = simData.CubicSplineKern1*pow(2 - q, 3);
		sum += pmterm * (buf.rest_colorValue[j]) * buf.mf_restmass[j] * buf.mdensity[j];

	}
	return sum;
}
__global__ void ComputeColorValue(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;
	if (buf.MFtype[i] == 2)return;
	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) return;						// particle out-of-range
	gc -= nadj;
	bool error = false;
	// Sum Pressures
	float3 pos = buf.mpos[i];
	buf.colorValue[i] = 0;
	for (int c = 0; c < simData.gridAdjCnt; c++)
	{
		buf.colorValue[i] += contributeColorValue(i, pos, gc + simData.gridAdj[c], buf);
	}
}
__device__ float3 contributeColorTensor(int i, int cell, bufList buf, float &sigma)
{
	float3 sum = make_float3(0, 0, 0);
	if (buf.mgridcnt[cell] == 0)return sum;
	
	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2;
	float3 dist;
	float c, dsq2, dsq;
	int j, mulj;
	float pmterm, cmterm;
	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];
	for (int cndx = cfirst; cndx < clast; cndx++)
	{
		j = buf.mgrid[cndx];
		if (buf.MFtype[j] == 2 || (_example == 2 && buf.MFtype[j] >= 2))
			continue;
		mulj = j * MAX_FLUIDNUM;
		dist = (buf.mpos[i] - buf.mpos[j])*simData.psimscale;
		dsq2 = dot(dist, dist);
		if (dsq2 > r2 ||dsq2 <=0)
			continue;
		dsq = sqrt(dsq2);
		c = simData.psmoothradius - dsq;
		cmterm = c*c*simData.spikykern / dsq;
		pmterm = pow(r2 - dsq2, 3)*simData.poly6kern;
		sum += cmterm * buf.colorValue[j] * buf.mf_restmass[j] * buf.mdensity[j] * dist;
		sigma += pmterm;
	}
	return sum;
}
__global__ void ComputeColorTensor(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;
	if (buf.MFtype[i]!=0)return;
	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) return;						// particle out-of-range
	gc -= nadj;
	bool error = false;
	// Sum Pressures
	float3 pos = buf.mpos[i];
	for (int k = 0; k < 9; ++k)
		buf.colorTensor[i * 9 + k] = 0;
	float3 gradCV = make_float3(0, 0, 0);
	float sigma = 0, divCV;
	for (int c = 0; c < simData.gridAdjCnt; c++)
	{
		gradCV += contributeColorTensor(i, gc + simData.gridAdj[c], buf, sigma);
	}
	divCV = dot(gradCV, gradCV);
	if ((sqrt(divCV)) < 0.000000001)
	{
		for (int k = 0; k < 9; ++k)
			buf.colorTensor[i * 9 + k] = 0;
		return;
	}
	tensorProduct(gradCV, gradCV, buf.colorTensor + i * 9);
	for (int m = 0; m < 3; ++m)
	{
		for (int n = 0; n < 3; ++n)
			if (m == n)
				buf.colorTensor[i * 9 + m * 3 + n] = divCV / 3 - buf.colorTensor[i * 9 + m * 3 + n];
			else
				buf.colorTensor[i * 9 + m * 3 + n] = - buf.colorTensor[i * 9 + m * 3 + n];
	}


	//if(abs(divCV) > 1)
	////if (i % 1000 == 0 || isnan(buf.colorValue[i]))
	//	//printf("%d's color value is %f, gradCV is (%f,%f,%f)\n", i, buf.colorValue[i], gradCV.x, gradCV.y, gradCV.z);
	//	printf("%d's color tensor is (%f,%f,%f)(%f,%f,%f)(%f,%f,%f), gradCV is (%f,%f,%f), sigma is %f\n", i,
	//		buf.colorTensor[i * 9 + 0], buf.colorTensor[i * 9 + 1], buf.colorTensor[i * 9 + 2],
	//		buf.colorTensor[i * 9 + 3], buf.colorTensor[i * 9 + 4], buf.colorTensor[i * 9 + 5],
	//		buf.colorTensor[i * 9 + 6], buf.colorTensor[i * 9 + 7], buf.colorTensor[i * 9 + 8], gradCV.x, gradCV.y, gradCV.z,
	//		sigma);
	for (int k = 0; k<9; ++k)
	{
		buf.colorTensor[i * 9 + k] *= simData.stRatio / (sqrt(divCV)*sigma*sigma);
	}
}

//__device__ float3 contributeDijPj(int i, float3 pos, int cell, bufList buf)
//{
//	float3 DijPj = make_float3(0,0,0);
//	if (buf.mgridcnt[cell] == 0)return DijPj;
//
//	register float d2 = simData.psimscale * simData.psimscale;
//	register float r2 = simData.r2;
//	float3 dist;
//	float c, dsq2, dsq;
//	int j;
//	float3 dji;
//	float cmterm;
//	float3 vmterm;
//	int cfirst = buf.mgridoff[cell];
//	int clast = cfirst + buf.mgridcnt[cell];
//	float q;
//	for (int cndx = cfirst; cndx < clast; cndx++)
//	{
//		j = buf.mgrid[cndx];
//		dist = (buf.mpos[i] - buf.mpos[j])*simData.psimscale;
//		dsq2 = dot(dist, dist);
//		dsq = sqrt(dsq2);
//		//q = dsq / simData.psmoothradius;
//		//if (q >= 2 || q <= 0)
//		//	continue;
//		//cmterm = buf.mf_restmass[j] * pow(buf.mdensity[j], 2)*buf.mpress[j];
//		//if(q>1)
//		//{
//		//	vmterm = simData.gradCubicSplineKern1*(2 - q)*(2 - q)*dist;
//		//	DijPj += cmterm*vmterm;
//		//}
//		//else
//		//{
//		//	vmterm = simData.gradCubicSplineKern2*(2.25*q*q - 3 * q)*dist;
//		//	DijPj += cmterm*vmterm;
//		//}
//		if (dsq2 > r2 || dsq2 <= 0)
//			continue;
//		c = (simData.psmoothradius - dsq);
//		cmterm = buf.mf_restmass[j] * pow(buf.mdensity[j], 2)*buf.mpress[j];
//		DijPj += c*c*dist *cmterm*simData.spikykern/dsq;
//		//DijPj += buf.mpress[j]*c*c*simData.spikykern*buf.mf_restmass[j] * pow(buf.mdensity[j], 2)*dist;
//		//DijPj += -buf.mf_restmass[j] * pow()
//	}
//	return DijPj;
//}
//__global__ void ComputeDijPj(bufList buf, int pnum)
//{
//	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
//	if (i >= pnum) return;
//
//	// Get search cell
//	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
//	uint gc = buf.mgcell[i];
//	if (gc == GRID_UNDEF) return;						// particle out-of-range
//	gc -= nadj;
//	bool error = false;
//	// Sum Pressures
//	float3 pos = buf.mpos[i];
//	float dens = buf.mf_restdensity[i];
//	buf.DijPj[i] = make_float3(0,0,0);
//	for (int c = 0; c < simData.gridAdjCnt; c++)
//	{
//		buf.DijPj[i] += contributeDijPj(i, pos, gc + simData.gridAdj[c], buf);
//	}
//	buf.DijPj[i] *= -simData.mf_dt*simData.mf_dt;
//	//if (i % 20000 == 0)
//	//	printf("particle %d's dijpj is (%f,%f,%f),press is %f\n", 
//	//		i, buf.DijPj[i].x, buf.DijPj[i].y, buf.DijPj[i].z, buf.mpress[i]);
//}

//__global__ void updatePress(bufList buf, int pnum)
//{
//	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
//	if (i >= pnum) return;
//
//	// Get search cell
//	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
//	uint gc = buf.mgcell[i];
//	if (gc == GRID_UNDEF) return;						// particle out-of-range
//	gc -= nadj;
//	bool error = false;
//	// Sum Pressures
//	float3 pos = buf.mpos[i];
//	float dens = buf.mf_restdensity[i];
//	float omega = 0.5;
//	buf.mpress_pre[i] = (1 - omega) * buf.mpress[i];
//	float sum = 0;
//	for (int c = 0; c < simData.gridAdjCnt; c++)
//	{
//		sum += contributePressureIteration(i, pos, gc + simData.gridAdj[c], buf);
//	}
//	float delta = buf.mf_restdensity[i] - buf.inter_density[i] - sum;
//	if (buf.aii[i] == 0)
//		buf.mpress_pre[i] = buf.mpress[i];
//	else
//		buf.mpress_pre[i] += omega / buf.aii[i] * (delta);
//	
//	//if (buf.mpress_pre[i] < 0)
//	//	buf.mpress_pre[i] = 0;
//	//if (i % 40000 == 0)
//	//	printf("aii is %.10f\n", buf.aii[i]);
//	//	printf("particle %d's press is %.10f,new press is %.10f, sum is %.10f, inter_density is %.10f,initial density is %f, aii is %.10f,delta is %.10f\n", 
//	//		i, buf.mpress[i], buf.mpress_pre[i], sum, buf.inter_density[i],1/buf.mdensity[i], buf.aii[i],delta);
//}
//__global__ void applyPress(bufList buf, int pnum)
//{
//	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
//	if (i >= pnum) return;
//
//	// Get search cell
//	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
//	uint gc = buf.mgcell[i];
//	if (gc == GRID_UNDEF) return;						// particle out-of-range
//	gc -= nadj;
//	if (buf.mpress_pre[i] < 0)
//		buf.mpress_pre[i] = 0;
//	buf.mpress[i] = buf.mpress_pre[i];
//	//if (i % 2000==0)
//	//	printf("particle %d's press is %f\n", i, buf.mpress[i]);
//}
__device__ float contributeCriterion(int i, int cell, bufList buf)
{
	float sum = 0;
	if (buf.mgridcnt[cell] == 0)return sum;

	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2;
	float3 dist;
	float c, dsq2, dsq;
	int j;
	float3 delta_force;
	float3 cmterm, vmterm;
	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];
	float q;
	for (int cndx = cfirst; cndx < clast; cndx++)
	{
		j = buf.mgrid[cndx];
		//if (buf.MFtype[i] != buf.MFtype[j] && (!buf.misbound[i] && !buf.misbound[j]))
		//	continue;
		dist = (buf.mpos[i] - buf.mpos[j])*simData.psimscale;
		dsq2 = dot(dist, dist);
		dsq = sqrt(dsq2);
		if (dsq2 > r2 || dsq2 <= 0)
			continue;
		//if (buf.MFtype[i] == 1 && buf.MFtype[i] == buf.MFtype[j])
		//	continue;
		c = simData.psmoothradius - dsq;
		
		//delta_force = buf.mf_restmass[j] * (buf.mforce[i] - buf.mforce[j]);
		//sum += dot(delta_force, dist)*c*c*simData.spikykern/dsq;

		//compute Ap
		//cmterm = buf.volume[j] * (buf.mforce[i] - buf.mforce[j]);
		//pairwise Ap
		if (buf.volume[i] * buf.volume[j] != 0)
			cmterm = buf.volume[i] * buf.volume[j] /(buf.volume[j]+buf.volume[i])* (buf.pressForce[i] - buf.pressForce[j]);
		else
			continue;
		sum += dot(cmterm, dist / dsq)*c*c*simData.spikykern;
	}
	return sum;
}
__global__ void ComputeCriterion(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;
	//if (buf.MFtype[i] == 3)return;
	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) return;						// particle out-of-range
	gc -= nadj;
	float sum = 0;
	float omega;
	
	omega = 0.5*buf.rest_volume[i] / pow(simData.psmoothradius / 2, 3);
	
	for (int c = 0; c < simData.gridAdjCnt; c++)
	{
		sum += contributeCriterion(i, gc + simData.gridAdj[c], buf);
	}
	sum *= pow(simData.mf_dt, 2);
	buf.delta_density[i] = buf.source[i] - sum;
	float p = buf.mpress[i];
	if (abs(buf.aii[i]) != 0)
		buf.mpress[i] = buf.mpress[i] + omega*buf.delta_density[i] / buf.aii[i];
	//float fluidsum = 0;
	//for (int k = 0; k < simData.mf_catnum; ++k)
	//	fluidsum += buf.mf_alpha[i*MAX_FLUIDNUM + k];
	//if(isnan(buf.delta_density[i]))
	//if (buf.mpress[i]!=0)
	//if(buf.mpress[i]>1000000||isnan(buf.mpress[i]))
	//if(abs(buf.delta_density[i])>1)
	//if(buf.mpos[i].y<-5)
		//printf("particle %d's type is %d, Ap is %f,source is %f, aii is %.10f,press is %f,press pre is %.10f, vel is (%f,%f,%f),volume is %.10f,rest volume is %.10f,press force is (%f,%f,%f),alpha is (%f,%f,%f),beta is (%f,%f,%f)\n",
		//	i, buf.MFtype[i], sum, buf.source[i], buf.aii[i], buf.mpress[i], p, 
		//	buf.vel_mid[i].x, buf.vel_mid[i].y,buf.vel_mid[i].z,
		//	buf.volume[i],buf.rest_volume[i], buf.pressForce[i].x, buf.pressForce[i].y, buf.pressForce[i].z,
		//	buf.mf_alpha[i*MAX_FLUIDNUM + 0], buf.mf_alpha[i*MAX_FLUIDNUM + 1], buf.mf_alpha[i*MAX_FLUIDNUM + 2],
		//	buf.mf_beta[i*MAX_FLUIDNUM + 0], buf.mf_beta[i*MAX_FLUIDNUM + 1], buf.mf_beta[i*MAX_FLUIDNUM + 2]);
	if (buf.mpress[i] < 0)
		buf.mpress[i] = 0;
	//if (buf.mpress[i] > 1000000)
	//	buf.mpress[i] = 1000000;
	if (buf.misbound[i] == 0)
	{
		if (buf.mpress[i] > 10000)
			buf.mpress[i] = 10000;
	}
	else
	{
		if (buf.mpress[i] > 1000000)
			buf.mpress[i] = 1000000;
	}

}


//************************************************************************
//pressure boundary for IISPH
__device__ float contributeBRestVolume(int i, int cell, bufList buf)
{
	float sum = 0;
	if (buf.mgridcnt[cell] == 0)return sum;

	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2;
	float3 dist;
	float c, dsq2, dsq;
	int j;
	
	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];
	for (int cndx = cfirst; cndx < clast; cndx++)
	{
		j = buf.mgrid[cndx];
		if (buf.MFtype[i]!=buf.MFtype[j])
			continue;
		dist = (buf.mpos[i] - buf.mpos[j])*simData.psimscale;
		dsq2 = dot(dist, dist);
		//dsq = sqrt(dsq2);
		if (dsq2 > r2 || dsq2 <= 0)
			continue;
		c = r2 - dsq2;
		sum += pow(c, 3)*simData.poly6kern;
	}
	return sum;
}
__global__ void ComputeBRestVolume(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;

	if (buf.MFtype[i]==0)
	{
		float sum = 0;
		for (int k = 1; k < simData.mf_catnum; ++k)
			sum += buf.mf_alpha[i*MAX_FLUIDNUM+k];
		buf.rest_volume[i] = sum*pow(simData.psmoothradius / 2, 3);
		if (isnan(sum))
			printf("error:sum is nan! fluid percent is (%f,%f,%f)\n",
				buf.mf_alpha[i*MAX_FLUIDNUM + 0],
				buf.mf_alpha[i*MAX_FLUIDNUM + 1],
				buf.mf_alpha[i*MAX_FLUIDNUM + 2]);
		return;
	}
	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) return;						// particle out-of-range
	gc -= nadj;
	float sum = 0;
	for (int c = 0; c < simData.gridAdjCnt; c++)
	{
		sum += contributeBRestVolume(i, gc + simData.gridAdj[c], buf);
	}
	sum += pow(simData.r2, 3)*simData.poly6kern;
	buf.rest_volume[i] = simData.solid_pfactor / sum;
}
__device__ float contributeVolume(int i, int cell, bufList buf)
{
	float sum = 0;
	if (buf.mgridcnt[cell] == 0)return sum;

	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2;
	float3 dist;
	float c, dsq2, dsq;
	int j;
	
	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];
	for (int cndx = cfirst; cndx < clast; cndx++)
	{
		j = buf.mgrid[cndx];
		//if (buf.MFtype[i] != buf.MFtype[j] && (!buf.misbound[i] && !buf.misbound[j]))
		//	continue;
		dist = (buf.mpos[i] - buf.mpos[j])*simData.psimscale;
		dsq2 = dot(dist, dist);
		//dsq = sqrt(dsq2);
		if (dsq2 > r2 || dsq2 <= 0)
			continue;
		c = r2 - dsq2;
		sum += buf.rest_volume[j] * pow(c, 3)*simData.poly6kern;
	}
	return sum;
}
__global__ void ComputeVolume(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;
	
	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) return;						// particle out-of-range
	gc -= nadj;
	float sum = 0;
	for (int c = 0; c < simData.gridAdjCnt; c++)
	{
		sum += contributeVolume(i, gc + simData.gridAdj[c], buf);
	}

	sum += buf.rest_volume[i] * pow(simData.r2, 3)*simData.poly6kern;

	//if (i % 30000 == 0)
	//	printf("volume sum is %.10f, 0.15*pow(simData.psmoothradius / 2, 3) is %.10f,rest_volume is %.10f\n",
	//		sum, 0.15 * pow(simData.psmoothradius / 2, 3), buf.rest_volume[i]);
	//if (buf.MFtype[i] != 0)
	if(buf.misbound[i])
		sum += 0.15*pow(simData.psmoothradius / 2, 3);
	if (sum == 0)
		buf.volume[i] = 0;
	else
		buf.volume[i] = buf.rest_volume[i] / sum;
	//if (buf.MFtype[i] == 0)
	//	buf.volume[i] *= buf.fluidPercent[i];
	//if (i % 30000 == 0)
	//if(buf.misbound[i]&&i%10000==0)

	//if (isnan(buf.volume[i])) 
	//{
	//	float fluidsum = 0;
	//	for (int k = 0; k < simData.mf_catnum; ++k)
	//		fluidsum += buf.mf_fluidPercent[i*MAX_FLUIDNUM + k];
	//	printf("particle %d's type is %d, rest_volume is %.10f, volume is %.10f, h3 is %.10f, sum is %.10f, fluidpercent is %f\n",
	//		i, buf.MFtype[i], buf.rest_volume[i], buf.volume[i], 2500000 * pow(simData.psmoothradius / 2, 3), sum, fluidsum);
	//}
}
__device__ float contributeSource(int i, int cell, bufList buf)
{
	float sum = 0;
	if (buf.mgridcnt[cell] == 0)return sum;

	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2;
	float3 dist;
	float c, dsq2, dsq;
	int j;
	
	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];
	float3 velocity,cmterm;
	float q;
	for (int cndx = cfirst; cndx < clast; cndx++)
	{
		j = buf.mgrid[cndx];
		//if (buf.MFtype[i] != buf.MFtype[j] && (!buf.misbound[i] && !buf.misbound[j]))
		//	continue;
		dist = (buf.mpos[i] - buf.mpos[j])*simData.psimscale;
		dsq2 = dot(dist, dist);
		dsq = sqrt(dsq2);
		if (dsq2 > r2 || dsq2 <= 0)
			continue;
		if (buf.MFtype[i] == 1 && buf.MFtype[i] == buf.MFtype[j])
			continue;
		//if(_example == 2 && (buf.MFtype[i] == buf.MFtype[j]) && buf.MFtype[i])
		c = simData.psmoothradius - dsq;
		//velocity = buf.vel_mid[i] - buf.vel_mid[j];
		//velocity = buf.mveleval[i] - buf.mveleval[j];
		//if(buf.MFtype[j]==0)
		//	velocity *= buf.fluidPercent[j]*buf.volume[j];
		//else
		//	velocity *= buf.volume[j];
		//pairwise divergence velocity
		if (buf.volume[i] * buf.volume[j] != 0)
			velocity = buf.volume[i] * buf.volume[j] / (buf.volume[i] + buf.volume[j]) * (buf.vel_mid[i] - buf.vel_mid[j]);
		else
			continue;
		cmterm = c*c*dist / dsq*simData.spikykern;
		sum += -dot(velocity, cmterm);
	}
	return sum;
}
__global__ void ComputeSource(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;
	
	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) return;						// particle out-of-range
	gc -= nadj;
	float sum = 0;
	for (int c = 0; c < simData.gridAdjCnt; c++)
	{
		sum += contributeSource(i, gc + simData.gridAdj[c], buf);
	}
	
	if (buf.MFtype[i] == 0) 
	{
		if(buf.volume[i] == 0)
			buf.source[i] = buf.mf_alpha_sum[i]*simData.mf_dt*sum;
		else
			buf.source[i] = (1
				- buf.rest_volume[i] / buf.volume[i]
				+ simData.mf_dt*sum)*buf.mf_alpha_sum[i];
	}
	else
		buf.source[i] = 1 - buf.rest_volume[i] / buf.volume[i] + simData.mf_dt*sum;
	//if(isnan(buf.source[i]))
	/*if (i % 30000 == 0&&buf.MFtype[i]==0)
		printf("particle %d's source is %f, fluidsum is %f,cat num is %d, rest_volume is %.10f, buf.volume is %.10f, velocity divergence is %.10f, mid vel is (%f,%f,%f)\n",
			i, buf.source[i], fluidsum, simData.mf_catnum, buf.rest_volume[i], buf.volume[i], simData.mf_dt*sum,
			buf.vel_mid[i].x, buf.vel_mid[i].y, buf.vel_mid[i].z);*/
}
__device__ float contributeAIIfluid(int i, float3 pos, int cell, bufList buf, float3&sum1, int&count)
{
	if (buf.mgridcnt[cell] == 0)return 0;
	float sum2 = 0;
	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2;
	float3 dist;
	float c, dsq2, dsq;
	int j;
	float3 dji;
	float cmterm;
	float3 vmterm;
	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];
	float q;
	for (int cndx = cfirst; cndx < clast; cndx++)
	{
		j = buf.mgrid[cndx];
		//if (buf.MFtype[i] != buf.MFtype[j] && (!buf.misbound[i] && !buf.misbound[j]))
		//	continue;
		dist = (buf.mpos[i] - buf.mpos[j])*simData.psimscale;
		dsq2 = dot(dist, dist);

		dsq = sqrt(dsq2);
		//spiky kern
		if (dsq2 > r2 || dsq2 <= 0)
			continue;
		c = (simData.psmoothradius - dsq);
		//pressure boundary
		count++;
		//if(buf.MFtype[i]==0||buf.MFtype[i]!=buf.MFtype[j])
		sum1 += buf.volume[j] * c*c*simData.spikykern*dist / dsq;
		if (!buf.misbound[j]) {
			if (buf.volume[j] == 0)
				sum2 += 0;
			else
				sum2 += buf.volume[j] * buf.volume[j] / buf.mf_restmass[j]
					* pow(c*c*simData.spikykern, 2);
		}
		//sum2 += buf.volume[j] * buf.volume[j] / (buf.mf_restmass[j]*(1-buf.absorbedPercent[i])) 
		//	* pow(c*c*simData.spikykern, 2);
	}
	return sum2;
}
__device__ float contributeAIIsolid(int i, float3 pos, int cell, bufList buf)
{
	if (buf.mgridcnt[cell] == 0)return 0;

	float sum = 0;
	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2;
	float3 dist;
	float c, dsq2, dsq;
	int j;
	float3 dji;
	float cmterm;
	float3 vmterm;
	int cfirst = buf.mgridoff[cell];
	int clast = cfirst + buf.mgridcnt[cell];
	float q;
	for (int cndx = cfirst; cndx < clast; cndx++)
	{
		j = buf.mgrid[cndx];

		dist = (buf.mpos[i] - buf.mpos[j])*simData.psimscale;
		dsq2 = dot(dist, dist);

		dsq = sqrt(dsq2);
		//spiky kern
		if (dsq2 > r2 || dsq2 <= 0)
			continue;

		c = (simData.psmoothradius - dsq);
		//iisph
		/*c = (simData.psmoothradius - dsq);
		cmterm = dot(buf.dii[i], dist)*buf.mf_restmass[j] * c*c*simData.spikykern / dsq;
		buf.aii[i] += cmterm;
		vmterm = pow(simData.mf_dt, 2)*buf.mf_restmass[i]
		* pow(buf.mdensity[i], 2) *c*c*simData.spikykern *dist /dsq;
		vmterm *= c*c*simData.spikykern/dsq*buf.mf_restmass[j];
		buf.aii[i] -= dot(vmterm, dist);*/

		//pressure boundary
		if (!buf.misbound[j]) {
			sum += buf.volume[j] * buf.volume[j] / buf.mf_restmass[j] * pow(c*c*simData.spikykern, 2);
		}
	}
	return sum;
}
__global__ void ComputeAII(bufList buf, int pnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= pnum) return;
	// Get search cell
	int nadj = (1 * simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[i];
	if (gc == GRID_UNDEF) return;						// particle out-of-range
	gc -= nadj;
	bool error = false;
	// Sum Pressures
	float3 pos = buf.mpos[i];
	float dens = buf.mf_restdensity[i];
	buf.aii[i] = 0;
	int count = 0;
	float3 sum1 = make_float3(0, 0, 0);

	for (int c = 0; c < simData.gridAdjCnt; c++)
	{
		if (!buf.misbound[i])
		//if(buf.MFtype[i]==0)
			buf.aii[i] += contributeAIIfluid(i, pos, gc + simData.gridAdj[c], buf, sum1, count);
		else
			buf.aii[i] += contributeAIIsolid(i, pos, gc + simData.gridAdj[c], buf);
	}
	float mass = buf.mf_restmass[i];

	buf.aii[i] += dot(sum1, sum1) / mass;
	//pressure boundary
	
	buf.aii[i] *= -simData.mf_dt*simData.mf_dt*buf.volume[i];
	buf.mpress[i] = 0;

}
