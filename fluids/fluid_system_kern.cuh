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

#ifndef DEF_KERN_CUDA
	#define DEF_KERN_CUDA
	#include "..\\fluids\\multifluid_def.h"

	#include <stdio.h>
	#include <math.h>
	#include <vector_types.h>

	typedef unsigned int		uint;
	typedef unsigned short int	ushort;
	// Particle & Grid Buffers
	struct bufList {
		//Particle properties
		float3*			mpos;
		float3*			maccel;
		float3*			vel_mid;		//intermediate velocity
		float3*			mveleval;		//mixture velocity u_m
		float3*			mforce;			//fluid&solid force
		float3*			poroForce;		//poro force
		float*			mpress;
		float*			mdensity;		//周围rest_mass求和的倒数
		uint*			mgcell;
		uint*			mgndx;
		uint*			mclr;			// 4 byte color
		int*			misbound;
		//End particle properties

		//multi fluid
		float*			mf_alpha;				// MAX_FLUIDNUM * 4 bytes for each particle
		float*			mf_alpha_pre;			// MAX_FLUIDNUM * 4 bytes for each particle
		float*			mf_pressure_modify;	//  4 bytes for each particle
		float3*			mf_vel_phrel;			// MAX_FLUIDNUM * 12 bytes for each particle  u_mk
		float*			mf_restmass;		//参数simData.pmass
		float*			mf_restdensity;
		float*			mf_visc;
		float3*			mf_velxcor;
		float3*			mf_alphagrad;			// MAX_FLUIDNUM * 12 bytes for each particle
		float*			mf_alphachange;
		//float*			density_fluid;
		//multi-fluid porous
		float*			permeability;
		float*			mf_beta;
		float*			mf_beta_next;
		float*			density_solid;
		float*			pressure_water;
		int*			solidCount;
		float*			totalDis;

		float3*			gradPressure;
		float3*			poroVel;
		float3*			fluidVel;

		int*            MFtype;					//0 means liquid,1 means deformation,2 means rigid，3 means fully absorbed liquid
		//End multi fluid

		//An Implicit SPH Formulation for Incompressible Linearly Elastic Solids
		float*					gradDeform;
		//float*					CorrectL;//use it to propose a correction matrix
		//float*					initialVolume;
		uint*					elasticID;
		float*					Rotation;
		//float*					stress;
		//For sorting
		char*			msortbuf;
		uint*			mgrid;	
		int*			mgridcnt;
		int*			mgridoff;
		int*			mgridactive;
		//new sort
		uint*			midsort;
		//End sorting
		
		float*			divDarcyFlux;
		float3*			SurfaceForce;
		bool*			isInside;//判断是否在固体内部
		//elastic material
		uint*	particleID;
		float*	initialVolume;
		uint*	neighborNum;
		uint*	neighborID;
		float3* neighborDistance;
		float3* kernelGrad;
		float3*	kernelRotate;
		uint*	neighborIndex;//该粒子在邻居链表中所在的索引
		float*	colorField;
		uint*	isSurface;//0 means internal particles, 1 means surface particle
		float3* normal;//固体表面的法线方向,指向外部
		float*	volumetricStrain;

		//IISPH
		float*			aii;
		float3*			pressForce;
		float*			delta_density;
		//pressure boundary for IISPH
		float*			rest_volume;
		float*			volume;
		float*			source;

	};// End particle&grid buffers

	// Temporary sort buffer offsets
	#define BUF_POS			0
	#define BUF_ACCEL		(sizeof(float3))

	#define BUF_VELEVAL		(BUF_ACCEL + sizeof(float3))
	#define BUF_FORCE		(BUF_VELEVAL + sizeof(float3))
	#define BUF_PRESS		(BUF_FORCE + sizeof(float3))
	#define BUF_DENS		(BUF_PRESS + sizeof(float))
	#define BUF_GCELL		(BUF_DENS + sizeof(float))
	#define BUF_GNDX		(BUF_GCELL + sizeof(uint))
	#define BUF_CLR			(BUF_GNDX + sizeof(uint))
	#define BUF_ISBOUND		(BUF_CLR + sizeof(uint))	

	//multi fluid sort buffer offsets
	#define BUF_ALPHA		(BUF_ISBOUND + sizeof(int))
	#define BUF_ALPHAPRE	(BUF_ALPHA + sizeof(float)*MAX_FLUIDNUM)
	#define BUF_PRESSMODI	(BUF_ALPHAPRE + sizeof(float)*MAX_FLUIDNUM)
	#define	BUF_VELPHREL	(BUF_PRESSMODI + sizeof(float))
	#define BUF_RMASS		(BUF_VELPHREL + sizeof(float3)*MAX_FLUIDNUM)
	#define BUF_RDENS		(BUF_RMASS + sizeof(float))
	#define BUF_VISC		(BUF_RDENS + sizeof(float))
	#define BUF_VELXCOR		(BUF_VISC + sizeof(float))
	#define	BUF_ALPHAGRAD	(BUF_VELXCOR + sizeof(float3))
	#define BUF_INDICATOR   (BUF_ALPHAGRAD + sizeof(float3)*MAX_FLUIDNUM)
	//implicit SPH formulation for elastic body
	//#define BUF_GRADDEFORM	(BUF_BORNID+sizeof(int))
	#define BUF_ELASTICID	(BUF_INDICATOR+sizeof(int))
	//#define BUF_ROTATION	(BUF_ELASTICID+sizeof(uint))
	#define BUF_ABSORBEDPERCENT	(BUF_ELASTICID+sizeof(int))
	#define BUF_BETANEXT	(BUF_ABSORBEDPERCENT + sizeof(float)*MAX_FLUIDNUM)

	//porous
	#define BUF_FVEL   (BUF_BETANEXT+sizeof(float)*MAX_FLUIDNUM)
	#define BUF_POROVEL		(BUF_FVEL+sizeof(float3)*MAX_FLUIDNUM)

	// Fluid Parameters (stored on both host and device)
	struct FluidParams {
		int				numThreads, numBlocks;
		int				gridThreads, gridBlocks;	

		int				szPnts, szHash, szGrid;
		int				stride, pnum;
		int				numElasticPoints;
		int				chk;
		float			pdist, pmass, prest_dens;
		float			pextstiff, pintstiff, pbstiff;
		float			pradius, psmoothradius, r2, psimscale, pvisc;
		float			pforce_min, pforce_max, pforce_freq, pground_slope;
		float			pvel_limit, paccel_limit, pdamp;
		float3			pboundmin, pboundmax, pgravity; //p is soft bound
		float3			mb1,mb2;
		float			AL, AL2, VL, VL2; //limits of acceleration and velocity
		

		float			poly6kern, spikykern, lapkern;
		float			CubicSplineKern1, CubicSplineKern2;
		float			gradCubicSplineKern1, gradCubicSplineKern2;

		float3			gridSize, gridDelta, gridMin, gridMax;
		int3			gridRes, gridScanMax;
		int				gridSrch, gridTotal, gridAdjCnt, gridActive;
		float			test1,test2,test3;
		int				gridAdj[64];
		
		//multi fluid parameters
		float			mf_dens[MAX_FLUIDNUM];
		float			mf_visc[MAX_FLUIDNUM];
		
		float			mf_diffusion;
		int				mf_catnum;
		float			mf_dt;

		uint			mf_multiFlagPNum;   //used with Buflist.mf_multiFlag, stores total number count (of flags)
		float			mf_splitVolume;
		float			mf_mergeVolume;
		uint			mf_maxPnum;
		float			cont,cont1,cont2;
		int				mf_up;
		float			relax;
		int				example;
		float			by,bxmin,bxmax,bzmin,bzmax,pan_r,omega; // for example3 rotation
		int				loadwhich;

		float			visc_factor, solid_pfactor, fluid_pfactor;
		float			bdamp;
		int				gravityfree;

		//elastic solids
		int				maxNeighborNum;
		float			miu, lambda;//parameters to compute strain&stress

		//porous
		float			rest_porosity;
		float			mf_permeability[MAX_FLUIDNUM];
		float			bulkModulus_porous;
		float			bulkModulus_grains;
		float			bulkModulus_solid;
		float			bulkModulus_fluid;
		float			CoCompressibility;

		bool			HideBound, HideFluid, HideSolid;
	};

	// Prefix Sum defines - 16 banks on G80
	#define NUM_BANKS		16
	#define LOG_NUM_BANKS	 4
	
	__global__ void insertParticles ( bufList buf, int pnum );
	__global__ void countingSortIndex ( bufList buf, int pnum );		
	__global__ void countingSortFull ( bufList buf, int pnum );		
	__global__ void computePressure ( bufList buf, int pnum );		
	__global__ void computeForce ( bufList buf, int pnum );
	__global__ void computePressureGroup ( bufList buf, int pnum );
	__global__ void advanceParticles ( float time, float dt, float ss, bufList buf, int numPnts );
	//new sort
	__global__ void InitialSort ( bufList buf, int pnum );
	__global__ void CalcFirstCnt ( bufList buf, int pnum );
	__global__ void CountingSortFull_ ( bufList buf, int pnum);
	__global__ void GetCnt ( bufList buf, int gnum );

	__global__ void initDensity(bufList buf,int pnum);
	__global__ void updateVelocity(float time, bufList buf, int pnum);

	__global__ void computeMidVel(bufList buf, int pnum);
	//multi fluid
	//calculating functions
	__global__ void mfChangeDensity (bufList buf, int pnum,const float scale);
	__global__ void mfComputeBound (bufList buf, int pnum);
	__global__ void mfFindNearest (bufList buf,int pnum);
	__global__ void mfPreComputeDensity ( bufList buf, int pnum );
	__global__ void mfComputePressure( bufList buf, int pnum );
	__global__ void mfComputeDriftVel( bufList buf, int pnum );
	__global__ void mfComputeAlphaAdvance( bufList buf, int pnum );
	__global__ void mfComputeCorrection( bufList buf, int pnum );
	__global__ void mfComputeForce( bufList buf, int pnum );
	__global__ void mfAdvanceParticles( float time, float dt, float ss, bufList buf, int numPnts );
	__global__ void mfAdvanceParticles_backup ( float time, float dt, float ss, bufList buf, int numPnts );

	//calculating functions for certain cases
	__global__ void mfComputeDriftVelVelLimit( bufList buf, int pnum );
	__global__ void mfComputeAlphaAdvanceLimitPositive( bufList buf, int pnum );
	__global__ void mfChangeDensity (bufList buf,int pnum,const float scale);

	//calculating functions for project-u
	__global__ void ComputeForce_projectu( bufList buf, int pnum );

	__global__ void AddSPHtensorForce(bufList buf,int pnum,float time);
	//end calculating functions for project-u

	//An Implicit SPH Formulation for Incompressible Linearly Elastic Solids
	__global__ void ComputeMap(bufList buf, int pnum);
	__global__ void ComputeInitialVolume(bufList buf, int pnum);
	__global__ void ComputeCorrectL(bufList buf, int pnum);
	__global__ void testFunc(bufList buf, int pnum);
	__global__ void ComputeDeformGrad(bufList buf, int pnum);
	__global__ void ComputeFinalDeformGrad(bufList buf, int pnum);
	__global__ void ComputeStrainAndStress(bufList buf, int pnum);
	__global__ void ComputeElasticForce(bufList buf, int pnum);
	__global__ void ComputeElasticColorField(bufList buf, int pnum);
	__global__ void ComputeElasticNormal(bufList buf, int pnum);
	//porous functions
	__global__ void AbsorbPercentCorrection(bufList buf, int pnum);
	__global__ void ComputeSurfaceTension(bufList buf, int pnum);
	__global__ void ComputePoroVelocity(bufList buf, int pnum);

	//new method
	__global__ void ComputeFluidAdvance(bufList buf, int pnum);
	__global__ void ComputePorePressure(bufList buf, int pnum);
	__global__ void ComputeSolidPorePressure(bufList buf, int pnum);
	__global__ void ComputeDarcyFlux(bufList buf, int pnum);
	__global__ void ComputeFluidFlux(bufList buf, int pnum);
	__global__ void ComputeFluidChange(bufList buf, int pnum);
	__global__ void ComputeFPCorrection(bufList buf, int pnum);
	__global__ void FindNearbySolid(bufList buf, int pnum);
	__global__ void ComputeSolidDarcyFlux(bufList buf, int pnum);
	__global__ void FluidPercentAdvanceByAlpha(bufList buf, int pnum);
	//implicit incompressible SPH
	__global__ void ComputePressureForce(bufList buf, int pnum);
	__global__ void ApplyPressureForce(bufList buf, int pnum);
	__global__ void ComputeCriterion(bufList buf, int pnum);

	//gravity,viscosity,etc
	__global__ void ComputeOtherForce(bufList buf, int pnum, float time);
	__global__ void ComputeAII(bufList buf, int pnum);

	//pressure boundary for iisph
	__global__ void ComputeBRestVolume(bufList buf, int pnum);
	__global__ void ComputeVolume(bufList buf, int pnum);
	__global__ void ComputeSource(bufList buf, int pnum);
	

	#define EPSILON				0.00001f
	#define GRID_UCHAR			0xFF
	#define GRID_UNDEF			4294967295

#endif
