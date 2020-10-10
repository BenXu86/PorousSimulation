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


#ifndef DEF_FLUID_SYS
	#define DEF_FLUID_SYS

	#include <iostream>
	#include <vector>
	#include <stdio.h>
	#include <stdlib.h>
	#include <math.h>

	#include "vector.h"
	#include "gl_helper.h"
	#include "xml_settings.h"
	#include "BI2Reader.h"
	#include "ObjReader.h"

	#define MAX_PARAM			60
	#define GRID_UCHAR			0xFF
	#define GRID_UNDEF			4294967295

	#define RUN_PAUSE			0
	#define RUN_SEARCH			1
	#define RUN_VALIDATE		2
	#define RUN_CPU_SLOW		3
	#define RUN_CPU_GRID		4
	#define RUN_CUDA_RADIX		5
	#define RUN_CUDA_INDEX		6
	#define RUN_CUDA_FULL		7
	#define RUN_CUDA_CLUSTER	8
	#define RUN_PLAYBACK		9
	//multi fluid
	#define RUN_MULTI_CUDA_FULL 10


	// Scalar params
	#define PMODE				0
	#define PNUM				1
	#define PEXAMPLE			2
	#define PSIMSIZE			3
	#define PSIMSCALE			4
	#define PGRID_DENSITY		5
	#define PGRIDSIZE			6
	#define PVISC				7
	#define PRESTDENSITY		8
	#define PMASS				9
	#define PRADIUS				10
	#define PDIST				11
	#define PSMOOTHRADIUS		12
	#define PINTSTIFF			13
	#define PEXTSTIFF			14
	#define PEXTDAMP			15
	#define PACCEL_LIMIT		16
	#define PVEL_LIMIT			17
	#define PSPACING			18
	#define PGROUND_SLOPE		19
	#define PFORCE_MIN			20
	#define PFORCE_MAX			21
	#define PMAX_FRAC			22
	#define PDRAWMODE			23
	#define PDRAWSIZE			24
	#define PDRAWGRID			25	
	#define PDRAWTEXT			26	
	#define PCLR_MODE			27
	#define PPOINT_GRAV_AMT		28
	#define PSTAT_OCCUPY		29
	#define PSTAT_GRIDCNT		30
	#define PSTAT_NBR			31
	#define PSTAT_NBRMAX		32
	#define PSTAT_SRCH			33
	#define PSTAT_SRCHMAX		34
	#define PSTAT_PMEM			35
	#define PSTAT_GMEM			36
	#define PTIME_INSERT		37
	#define PTIME_SORT			38
	#define PTIME_COUNT			39
	#define PTIME_PRESS			40
	#define PTIME_FORCE			41
	#define PTIME_ADVANCE		42
	#define PTIME_RECORD		43
	#define PTIME_RENDER		44
	#define PTIME_TOGPU			45
	#define PTIME_FROMGPU		46
	#define PFORCE_FREQ			47

	//multi fluid
	#define FLUID_CATNUM		48
	#define PTIMEDRIFTVEL	49
	#define PTIMEALPHA		50
	#define PTIMECORR		51
	#define PTIMESPLIT		52

	//NEW_BOUND
	#define PBVISC				53
	#define PBRESTDENSITY		54
	#define PBMASS				55
	#define PBSTIFF				56
	//POROUS
	#define PERMEABILITY		57

	// Vector params
	#define PVOLMIN				0
	#define PVOLMAX				1
	#define PBOUNDMIN			2
	#define PBOUNDMAX			3
	#define PINITMIN			4
	#define PINITMAX			5
	#define PEMIT_POS			6
	#define PEMIT_ANG			7
	#define PEMIT_DANG			8
	#define PEMIT_SPREAD		9
	#define PEMIT_RATE			10
	#define PPOINT_GRAV_POS		11	
	#define PPLANE_GRAV_DIR		12	

	// Booleans
	#define PRUN				0
	#define PDEBUG				1	
	#define PUSE_CUDA			2	
	#define	PUSE_GRID			3
	#define PWRAP_X				4
	#define PWALL_BARRIER		5
	#define PLEVY_BARRIER		6
	#define PDRAIN_BARRIER		7		
	#define PPLANE_GRAV_ON		11	
	#define PPROFILE			12
	#define PCAPTURE			13

	#define BFLUID				2
	#define HIDEBOUND			14
	#define HIDEFLUID			15
	#define HIDESOLID			16
    #define HIDERIGID			17
	//From YanXiao
	#define OUTPUT_INT			1
	#define START_OUTPUT		2
	#define SHOW_BOUND			3
	#define SAVE_STAT			4
	#define CHANGE_DEN			6

	
	struct NList {
		int num;
		int first;
	};
	struct Fluid {						// offset - TOTAL: 72 (must be multiple of 12)
		Vector3DF		pos;			// 0
		Vector3DF		vel;			// 12
		Vector3DF		veleval;		// 24
		Vector3DF		force;			// 36
		float			pressure;		// 48
		float			density;		// 52
		int				grid_cell;		// 56
		int				grid_next;		// 60
		DWORD			clr;			// 64
		//NEW_BOUND
		int				isbound;		// 68
		int				padding;		// 68

		//Multifluid Particle-----------------------TOTAL offset: 84+24*MAX_FLUIDNUM
		float			alpha[MAX_FLUIDNUM];		   //  0						| +72
		float			alpha_pre[MAX_FLUIDNUM];	   //  4 * MAX_FLUIDNUM			|
		float			pressure_modify; //				8 * MAX_FLUIDNUM			|
		Vector3DF		vel_phrel[MAX_FLUIDNUM];	   // 8 * MAX_FLUIDNUM	+ 4		|

		float			restMass;					   // 20 * MAX_FLUIDNUM	+ 4		|
		float			restDensity;				   // 20 * MAX_FLUIDNUM + 8		|
		float			visc;						   // 20 * MAX_FLUIDNUM + 12	|
		Vector3DF		velxcor;
	};

	class FluidSystem {
	public:
		FluidSystem ();
		
		// Rendering
		void Draw ( Camera3D& cam, float rad );
		void DrawGrid ();
		void DrawText ();
		void DrawParticle ( int p, int r1, int r2, Vector3DF clr2 );
		void DrawParticleInfo ()		{ DrawParticleInfo ( mSelected ); }
		void DrawParticleInfo ( int p );

		// Particle Utilities
		void AllocateParticles ( int cnt );
		int AddParticle ();
		int NumPoints ()		{ return mNumPoints;}
		int MaxPoints ()		{ return mMaxPoints;}

		// Setup
		void Setup ( bool bStart );
		void SetupRender ();
		void SetupKernels ();
		void SetupDefaultParams ();
		void SetupSpacing ();
		void SetupAddVolume ( Vector3DF min, Vector3DF max, float spacing, float offs );
#ifdef NEW_BOUND
		void SetupAddBound(BI2Reader bi2reader,int boundtype);
#endif
		int SetupAddMonster(BI2Reader bi2reader, int type, int cat);
		void SetupAddShape(BI2Reader bi2reader,int cat);
		void SetupGridAllocate ( Vector3DF min, Vector3DF max, float sim_scale, float cell_size, float border );
		void ParseXML ( std::string name, int id, bool bStart );
		void ParseXML_Bound (std::string name, int boundnum);

		void saveParticle(std::string name);
		int loadParticle(std::string name);
		
		void liftup(int mode);

		void ParseMFXML ( std::string name, int id, bool bStart );
		int SetupMfAddVolume( Vector3DF min, Vector3DF max, float spacing, Vector3DF offs, int cat);// cat: category label
		int SetupMfAddBlendVolume(Vector3DF min, Vector3DF max, float spacing, Vector3DF offs);// cat: category label
		int SetupMfAddMultiSolid(Vector3DF min, Vector3DF max, float spacing, Vector3DF offs);
		int SetupMfAddSolidSolid(Vector3DF min, Vector3DF max, float spacing, Vector3DF offs, int type);
		int SetupMfAddCylinder(Vector3DF min, Vector3DF max, float spacing, Vector3DF offs, int type);
		int SetupBoundary(Vector3DF min, Vector3DF max, float spacing, Vector3DF offs, int type);

		int SetupMfAddDeformVolume( Vector3DF min, Vector3DF max, float spacing, Vector3DF offs, int type);
		int FluidSystem::SetupMfAddSphere(Vector3DF min, Vector3DF max, float spacing, Vector3DF offs, int type);
		
		int SetupModel(PIC*model, float spacing, int type, Vector3DF displacement);
		int GenerateBunnies(PIC*bunny, float spacing, int type);

		void AddMfEmit(float spacing, int cat);
		void EmitMfParticles(int cat);
		
		void EmitUpdateCUDA(int startnum, int endnum);
		//int AddMfParticle(int cat);

		// Simulation
		void Run (int w, int h);
		void RunSimulateMultiCUDAFull();
		void OnfirstRun();

		void Exit ();
		void TransferToCUDA ();
		void TransferFromCUDA ();
		void TransferFromCUDAForLoad();
		double GetDT()		{ return m_DT; }

		void MfTestSetupExample();
		void setupSPHexample();
		int frameNum(){ return m_Frame;}

		// Debugging
		void SaveResults ();
		void CaptureVideo (int width, int height);
		void record ( int param, std::string, mint::Time& start );
		int SelectParticle ( int x, int y, int wx, int wy, Camera3D& cam );
		int GetSelected ()		{ return mSelected; }
		
		// Acceleration Grid
		int getGridCell ( int p, Vector3DI& gc );
		int getGridCell ( Vector3DF& p, Vector3DI& gc );
		int getGridTotal ()		{ return m_GridTotal; }
		int getSearchCnt ()		{ return m_GridAdjCnt; }
		Vector3DI getCell ( int gc );
		Vector3DF GetGridRes ()		{ return m_GridRes; }
		Vector3DF GetGridMin ()		{ return m_GridMin; }
		Vector3DF GetGridMax ()		{ return m_GridMax; }
		Vector3DF GetGridDelta ()	{ return m_GridDelta; }

		// Acceleration Neighbor Tables
		void AllocateNeighborTable ();
		void ClearNeighborTable ();
		int GetNeighborTableSize ()	{ return m_NeighborNum; }
		
		// GPU Support functions
		void AllocatePackBuf ();
		void PackParticles ();
		void UnpackParticles ();

		// Recording
		void StartRecord ();
		void StartPlayback ( int p );
		void Record ();
		std::string getFilename ( int n );
		int getLastRecording ();
		int getMode ()		{ return m_Param[PMODE]; }
		std::string getModeStr ();
		void getModeClr ();

		// Parameters			
		void SetParam (int p, float v )		{ m_Param[p] = v; }
		void SetParam (int p, int v )		{ m_Param[p] = (float) v; }
		float GetParam ( int p )			{ return (float) m_Param[p]; }
		float SetParam ( int p, float v, float mn, float mx )	{ m_Param[p] = v ; if ( m_Param[p] > mx ) m_Param[p] = mn; return m_Param[p];}
		float IncParam ( int p, float v, float mn, float mx )	{ 
			m_Param[p] += v; 
			if ( m_Param[p] < mn ) m_Param[p] = mn; 
			if ( m_Param[p] > mx ) m_Param[p] = mn; 
			return m_Param[p];
		}
		Vector3DF GetVec ( int p )			{ return m_Vec[p]; }
		void SetVec ( int p, Vector3DF v )	{ m_Vec[p] = v; }
		void Toggle ( int p )				{ m_Toggle[p] = !m_Toggle[p]; }		
		bool GetToggle ( int p )			{ return m_Toggle[p]; }
		std::string		getSceneName ()		{ return mSceneName; }

		//From YanXiao
		void SetYan (int p, int v)			{ m_Yan[p] = v; }
		int GetYan (int p)					{ return m_Yan[p]; }
		
		void outputFile();
		void outputepsilon(FILE* fp);

		void storeModel(char* filename);

		void LoadParticles(char* filename, Vector3DF off);
		void solveModel();
		int						recordNum;
	private:

		std::string				mSceneName;
		std::string				Bi2Dir,Bi2Dir1;
		// Time
		int							m_Frame;		
		double						m_DT;
		double						m_Time;	
		double						m_CostTime;

		// Simulation Parameters
		double						m_Param [ MAX_PARAM ];			// see defines above
		Vector3DF					m_Vec [ MAX_PARAM ];
		bool						m_Toggle [ MAX_PARAM ];
		int							m_Yan [MAX_PARAM];

		// SPH Kernel functions
		double					m_R2, m_Poly6Kern, m_LapKern, m_SpikyKern;		
		double					CubicSplineKern1, CubicSplineKern2;
		double					gradCubicSplineKern1, gradCubicSplineKern2;
		// Particle Buffers
		int						mNumPoints;
		int						NumPointsNoBound;
		int						mMaxPoints;
		int						mGoodPoints;
		Vector3DF*				mPos;
		DWORD*					mClr;
		int*					mIsBound;
		Vector3DF*				mVel;
		Vector3DF*				mVelEval;
		unsigned short*			mAge;
		float*					mPressure;
		float*					mDensity;
		Vector3DF*				mForce;
		uint*					mGridCell;
		uint*					mClusterCell;
		uint*					mGridNext;
		uint*					mNbrNdx;
		uint*					mNbrCnt;

		float*					m_alpha;  //size: mMaxPoints * MAX_FLUIDNUM
		float*					m_beta;
		float*					m_alpha_pre; //size: mMaxPoints * MAX_FLUIDNUM
		float*					m_pressure_modify; //size: mMaxPoints * MAX_FLUIDNUM
		Vector3DF*				m_vel_phrel; //size: mMaxPoints * MAX_FLUIDNUM

		float*					m_restMass;
		float*					m_restDensity;

		float*					m_visc;
		Vector3DF*				m_velxcor; //XSPH correction
		Vector3DF*				m_alphagrad; //size: mMaxPoints * MAX_FLUIDNUM
		
		int*					MF_type;  //(project-u)
		int						m_maxAllowedPoints;
		//An Implicit SPH Formulation for Incompressible Linearly Elastic Solids µœ÷
		float*					gradDeform;
		float*					CorrectL;//use it to propose a correction matrix

		// Acceleration Grid
		uint*					m_Grid;
		uint*					m_GridCnt;
		int						m_GridTotal;			// total # cells
		Vector3DI				m_GridRes;				// resolution in each axis
		Vector3DF				m_GridMin;				// volume of grid (may not match domain volume exactly)
		Vector3DF				m_GridMax;		
		Vector3DF				m_GridSize;				// physical size in each axis
		Vector3DF				m_GridDelta;
		int						m_GridSrch;
		int						m_GridAdjCnt;
		int						m_GridAdj[216];

		// Acceleration Neighbor Table
		int						m_NeighborNum;
		int						m_NeighborMax;
		int*					m_NeighborTable;
		float*					m_NeighborDist;

		char*					mPackBuf;
		int*					mPackGrid;

		int						mVBO[3];

		// Record/Playback
		int						mFileNum;
		std::string				mFileName;
		float					mFileSize;
		FILE*					mFP;
		int						mLastPoints;
		
		int						mSpherePnts;
		int						mTex[1];
		GLuint					instancingShader;

		// Selected particle
		int						mSelected;
		Image					mImg;


		// Saved results (for algorithm validation)
		uint*					mSaveNdx;
		uint*					mSaveCnt;
		uint*					mSaveNeighbors;

		// XML Settings file
		XmlSettings				xml;

		//From YanXiao
		int nOutFrame;

		//MULTIFLUID PARAMETERS
		float					m_fluidPMass[MAX_FLUIDNUM];//PMass is used by AddVolume;
		float					m_fluidDensity[MAX_FLUIDNUM];
		float					m_fluidVisc[MAX_FLUIDNUM];
		float					m_fluidDiffusion;
		//float					m_splitVolume;
		//float					m_mergeVolume;
		float					m_Permeability[MAX_FLUIDNUM*MAX_SOLIDNUM];
		float					pressureRatio[MAX_FLUIDNUM*MAX_SOLIDNUM];
		float					restColorValue[MAX_FLUIDNUM];
		float					SurfaceTensionRatio;

		double vfactor, fpfactor, spfactor;
		double bdamp;
		FILE* epsilonfile;

		//elastic information
		int						numElasticPoints;
		int						maxNeighborNum;
		uint*					elasticID;
		float					miu, lambda;//parameters to compute strain&stress
		float					porosity;
		float					bulkModulus_porous;
		float					bulkModulus_grains;
		float					bulkModulus_solid;
		float					bulkModulus_fluid;

		float*					porosity_particle;
		Vector3DF*				signDistance;//distance between solid particle to surface
		float					poroDeformStrength;
		float					capillary;
		float					capillaryForceRatio;
		float					Relax2;
		//bool*					misGhost;
	};	
#endif
