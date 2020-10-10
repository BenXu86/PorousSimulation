/*
  FLUIDS v.3 - SPH Fluid Simulator for CPU and GPU
  Copyright (C) 2012. Rama Hoetzlein, http://fluids3.com

  Fluids-ZLib license (* see part 1 below)
  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
	 claim that you wrote the original software. Acknowledgement of the
	 original author is required if you publish this in a paper, or use it
	 in a product. (See fluids3.com for details)
  2. Altered source versions must be plainly marked as such, and must not be
	 misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
	 
#ifndef DEF_HOST_CUDA
#define DEF_HOST_CUDA
	#include "..\\fluids\\multifluid_def.h"

	#include <vector_types.h>	
	#include <driver_types.h>			// for cudaStream_t
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

	//#define TOTAL_THREADS			1000000
	//#define BLOCK_THREADS			256
	//#define MAX_NBR					80	
	
	//#define COLOR(r,g,b)	( (uint((b)*255.0f)<<16) | (uint((g)*255.0f)<<8) | uint((r)*255.0f) )
	#define COLORA(r,g,b,a)	( (uint((a)*255.0f)<<24) | (uint((b)*255.0f)<<16) | (uint((g)*255.0f)<<8) | uint((r)*255.0f) )

	typedef unsigned int		uint;
	typedef unsigned short		ushort;
	typedef unsigned char		uchar;

	extern "C"
	{

	void cudaInit(int argc, char **argv);
	void cudaExit(int argc, char **argv);

	void FluidClearCUDA ();
	void FluidSetupRotationCUDA (float pan_r,float omega,int loadwhich, float capillaryForceRatio); //for example3 rotation
	float FluidSetupCUDA (  int num, int gsrch, int3 res, float3 size, float3 delta, float3 gmin, float3 gmax, int total, int chk);
	void FluidParamCUDA ( float ss, float sr, float pr, float mass, float rest, float3 bmin, float3 bmax, float estiff, float istiff, float pbstiff, float visc, float damp, float fmin, float fmax, float ffreq, float gslope, float gx, float gy, float gz, float al, float vl );
	void FluidParamCUDA_projectu(float v_factor, float f_factor, float s_factor,float bdamp);
	void ParamUpdateCUDA(bool hidebound, bool hidefluid,bool hidesolid,bool hiderigid, float* colorValue);
	//elastic information
	float ElasticSetupCUDA(int num,float miu,float lambda,float porosity,float* permeability,int maxNeighborNum, float* permRatio, float stRatio);
	//porous
	void PorousParamCUDA(float bulkModulus_porous, float bulkModulus_grains, float bulkModulus_solid, float	bulkModulus_fluid,float poroDeformStrength, float capillary, float relax2);
	//multi fluid
	void FluidMfParamCUDA ( float *dens, float *visc, float*mass,float diffusion, float catnum, float dt, float3 cont, float3 mb1, float3 mb2, float relax,int example);
	void CopyMfToCUDA ( float* alpha, float* alpha_pre, float* pressure_modify, float* vel_phrel, float* restmass, float* restdensity, float* visc, float* velxcor, float* alphagrad);
	void CopyMfFromCUDA ( float* alpha, float* alpha_pre, float* pressure_modify, float* vel_phrel, float* restmass, float* restdensity, float* visc, float* velxcor, float* alphagrad, int mode);

	void TestFunc();
	//emit
	void CopyEmitToCUDA ( float* pos, float* vel, float* veleval, float* force, float* pressure, float* density, uint* cluster, uint* gnext, char* clr, int startnum, int numcount ,int* mIsBound);
	void CopyEmitMfToCUDA ( float* alpha, float* alpha_pre, float* pressure_modify, float* vel_phrel, float* restmass, float* restdensity, float* visc, float* velxcor, float* alphagrad, int startnum, int numcount);
	void UpdatePNumCUDA( int newPnum);

	void CopyToCUDA ( float* pos, float* vel, float* veleval, float* force, float* pressure, float* density, uint* cluster, uint* gnext, char* clr );
	void CopyFromCUDA ( float* pos, float* vel, float* veleval, float* force, float* pressure, float* density, uint* cluster, uint* gnext, char* clr, int mode);

	void CopyBoundToCUDA(int* isbound);
	void CopyBoundFromCUDA(int* isbound);

	void CopyToCUDA_Uproject(int* mftype);
	void CopyFromCUDA_Uproject(int* mftype, float*beta);
	
	void CopyToCUDA_elastic(uint*elasticID, float*porosity,float*signDistance);
	void CopyFromCUDA_elastic();

	void prefixSumInt ( int num );
	void preallocBlockSumsInt(unsigned int num);
	void deallocBlockSumsInt();
	
	//new sort
	void InitialSortCUDA( uint* gcell, uint* ccell, int* gcnt );
	void SortGridCUDA( int* goff );
	void CountingSortFullCUDA_( uint* ggrid );
	void initSPH(float* restdensity,int* mftype);

	//Multifluid simulation
	void MfComputePressureCUDA();
	void MfComputeDriftVelCUDA();
	void MfComputeAdvanceCUDA();
	void MfComputeCorrectionCUDA();  
	void MfComputeForceCUDA ();	
	void MfAdvanceCUDA ( float time , float dt, float ss );
	void MfChangeDensityCUDA(const float scale);

	//An Implicit SPH Formulation for Incompressible Linearly Elastic Solids
	void ComputeCorrectLCUDA();
	//compute initial deformable gradient and extract the rotation
	void ComputeDeformGradCUDA();
	void ComputeElasticForceCUDA();
	//porous function
	void ComputePorousForceCUDA();
	//Project-U changes computing force
	//void ComputeForceCUDA_ProjectU(float time);
	void floatup_cuda(int mode);
	int MfGetPnum();
	void LeapFrogIntegration(float time);
	//implicit incompressible SPH
	void MfPredictAdvection(float time);
	void PressureSolve(int fluid_beginIndex,int fluid_endIndex);
	}

#endif