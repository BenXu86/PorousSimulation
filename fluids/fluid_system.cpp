#include <assert.h>
#include <stdio.h>
#include <conio.h>

#include "gl_helper.h"
#include <gl/glut.h>

#include "camera3d.h"
#include "common_defs.h"
#include "mtime.h"
#include "fluid_system.h"
#include "fluid_system_host.cuh"

#include <io.h>
#include <direct.h>
#include <iostream>
#include <fstream>
#include <iomanip>
//计时
#include <Mmsystem.h>
#pragma comment(lib, "Winmm.lib")
double scaleP,scaleP3,scaledis;
//double a[18],b[6];
Vector3DF volumes[10],softBoundary[2],emit[2];
Vector3DF cont,mb1,mb2;
Vector4DF massRatio,densityRatio,viscRatio;
Vector4DF permeabilityRatio;
Vector4DF pressRatio;
Vector4DF colorValue;

//float cont[3];
int loadwhich;
int example;

float panr,omega; //case 3
float relax;
float emitSpeed,emitangle,emitcircle,emitposx,emitposy,emitposz,emitfreq; //case 5

int upframe; // you know, the density will change in case 2 with upframe reached
float change_den=2;
float last_den=2;
float test=1.0;

void FluidSystem::TransferToCUDA ()
{ 
	CopyToCUDA ( (float*) mPos, (float*) mVel, (float*) mVelEval, (float*) mForce, mPressure, mDensity, mClusterCell, mGridNext, (char*) mClr); 

	CopyMfToCUDA ( m_alpha, m_alpha_pre, m_pressure_modify, (float*) m_vel_phrel, m_restMass, m_restDensity, m_visc, (float*)m_velxcor, (float*)m_alphagrad);
	CopyToCUDA_Uproject((int*) MF_type);
	CopyToCUDA_elastic(elasticID, porosity_particle, (float*)signDistance);
#ifdef NEW_BOUND
	CopyBoundToCUDA(mIsBound);
#endif

}
void FluidSystem::TransferFromCUDA ()	
{
	CopyFromCUDA ( (float*) mPos, (float*) mVel, (float*) mVelEval, (float*) mForce, mPressure, mDensity, mClusterCell, mGridNext, (char*) mClr, 1);
	CopyMfFromCUDA ( m_alpha, m_alpha_pre, m_pressure_modify, (float*) m_vel_phrel, m_restMass, m_restDensity, m_visc, (float*)m_velxcor, (float*)m_alphagrad, 1);
	CopyFromCUDA_Uproject(MF_type,m_beta);

#ifdef NEW_BOUND
	CopyBoundFromCUDA(mIsBound);
#endif
}
void FluidSystem::TransferFromCUDAForLoad ()
{
	CopyFromCUDA ( (float*) mPos, (float*) mVel, (float*) mVelEval, (float*) mForce, mPressure, mDensity, mClusterCell, mGridNext, (char*) mClr, 2);
	CopyMfFromCUDA ( m_alpha, m_alpha_pre, m_pressure_modify, (float*) m_vel_phrel, m_restMass, m_restDensity, m_visc, (float*)m_velxcor, (float*)m_alphagrad, 2);

#ifdef NEW_BOUND
	CopyBoundFromCUDA(mIsBound);
#endif
	CopyFromCUDA_Uproject(MF_type, m_beta);
}

//------------------------------ Initialization
FluidSystem::FluidSystem ()
{
	mNumPoints = 0;
	mMaxPoints = 0;
	mPackBuf = 0x0;
	mPackGrid = 0x0;
	mFP = 0x0;

	mPos = 0x0;
	mClr = 0x0;
	mIsBound = 0x0;
	mVel = 0x0;
	mVelEval = 0x0;
	mAge = 0x0;
	mPressure = 0x0;
	mDensity = 0x0;
	mForce = 0x0;
	mClusterCell = 0x0;
	mGridNext = 0x0;
	mNbrNdx = 0x0;
	mNbrCnt = 0x0;
	mSelected = -1;
	m_Grid = 0x0;
	m_GridCnt = 0x0;

	m_Frame = 0;
	
	m_NeighborTable = 0x0;
	m_NeighborDist = 0x0;

	elasticID = 0x0;

	m_Param [ PMODE ]		= RUN_MULTI_CUDA_FULL;
	m_Param [ PEXAMPLE ]	= 1;
	m_Param [ PGRID_DENSITY ] = 2.0;
	m_Param [ PNUM ]		= 8192; //65536 * 128;
	//m_Param[PNUM] = 12000;

	m_Toggle [ PDEBUG ]		=	false;
	m_Toggle [ PUSE_GRID ]	=	false;
	m_Toggle [ PPROFILE ]	=	false;
	m_Toggle [ PCAPTURE ]   =	false;
	m_Toggle [ HIDEBOUND]   =   true;
	m_Toggle [HIDEFLUID]    =   false;
	m_Toggle [HIDESOLID]    =   false;
	m_Toggle [HIDERIGID]    =   false;
	memset(m_Yan,0,sizeof(m_Yan));
	if ( !xml.Load ( "scene.xml" ) ) {
		error.PrintF ( "fluid", "ERROR: Problem loading scene.xml. Check formatting.\n" );
		error.Exit ();
	}
	//From YanXiao
	nOutFrame = 0;
}

void FluidSystem::Setup ( bool bStart )
{
	//TestPrefixSum ( 16*1024*1024 );
	m_Frame = 0;
	m_Time = 0;

	ClearNeighborTable ();
	mNumPoints = 0;
	
	SetupDefaultParams ();

	//MfTestSetupExample();
	setupSPHexample();
	//epsilonfile = fopen("OutputData\\epsilonFile.txt","w+");
	printf("max-allowed particle number is %d\n", m_maxAllowedPoints);
	printf("particle num:%d\n", NumPoints());
	printf("elastic num:%d\n", numElasticPoints);
	printf("spacing is %f, smooth radius is %f\n", m_Param[PSPACING], m_Param[PSMOOTHRADIUS]/ m_Param[PSIMSCALE]);
	SetupGridAllocate ( m_Vec[PVOLMIN], m_Vec[PVOLMAX], m_Param[PSIMSCALE], m_Param[PGRIDSIZE], 1.0 );	// Setup grid

	FluidClearCUDA ();
	Sleep ( 500 );
	FluidSetupRotationCUDA (panr, omega, loadwhich, capillaryForceRatio);

	float CudaMem = 0;

	CudaMem += ElasticSetupCUDA(numElasticPoints, miu, lambda, porosity, m_Permeability, maxNeighborNum, pressureRatio, SurfaceTensionRatio);
	PorousParamCUDA(bulkModulus_porous, bulkModulus_grains, bulkModulus_solid, bulkModulus_fluid, poroDeformStrength, capillary,Relax2);
	
	CudaMem += FluidSetupCUDA ( NumPoints(), m_GridSrch, *(int3*)& m_GridRes, *(float3*)& m_GridSize, *(float3*)& m_GridDelta, *(float3*)& m_GridMin, *(float3*)& m_GridMax, m_GridTotal, (int) m_Vec[PEMIT_RATE].x );
	std::cout << "CUDA memory cost : " << CudaMem << std::endl;
	Sleep ( 500 );

	Vector3DF grav = m_Vec[PPLANE_GRAV_DIR];
	printf("%f %f\n",m_Param[PBSTIFF],m_Param[PEXTSTIFF]);
	FluidParamCUDA ( m_Param[PSIMSCALE], m_Param[PSMOOTHRADIUS], m_Param[PRADIUS], m_Param[PMASS], m_Param[PRESTDENSITY],
		*(float3*)& m_Vec[PBOUNDMIN], *(float3*)& m_Vec[PBOUNDMAX], m_Param[PEXTSTIFF],
		m_Param[PINTSTIFF],m_Param[PBSTIFF], m_Param[PVISC],    m_Param[PEXTDAMP],   
		m_Param[PFORCE_MIN], m_Param[PFORCE_MAX],  m_Param[PFORCE_FREQ], m_Param[PGROUND_SLOPE], 
		grav.x, grav.y, grav.z, m_Param[PACCEL_LIMIT], m_Param[PVEL_LIMIT] );
	ParamUpdateCUDA(m_Toggle[HIDEBOUND], m_Toggle[HIDEFLUID], m_Toggle[HIDESOLID],m_Toggle[HIDERIGID], restColorValue);
	//FluidParamCUDA ( m_Param[PSIMSCALE], m_Param[PSMOOTHRADIUS], m_Param[PRADIUS], m_Param[PMASS], m_Param[PRESTDENSITY], *(float3*)& m_Vec[PBOUNDMIN], *(float3*)& m_Vec[PBOUNDMAX], m_Param[PBSTIFF],   m_Param[PINTSTIFF],                  m_Param[PVISC],    m_Param[PEXTDAMP],   m_Param[PFORCE_MIN], m_Param[PFORCE_MAX],  m_Param[PFORCE_FREQ], m_Param[PGROUND_SLOPE], grav.x, grav.y, grav.z, m_Param[PACCEL_LIMIT], m_Param[PVEL_LIMIT] );
	FluidParamCUDA_projectu(vfactor, fpfactor, spfactor,bdamp);
	
	FluidMfParamCUDA (m_fluidDensity,m_fluidVisc,m_fluidPMass,m_fluidDiffusion,m_Param[FLUID_CATNUM], m_DT, *(float3*)& cont,*(float3*)& mb1,*(float3*)& mb2, relax, example);
	//cout << "fluid catnum is " << m_Param[FLUID_CATNUM] << endl;
	TransferToCUDA ();		// Initial transfer
	
	//Sorting
	InitialSortCUDA( 0x0, 0x0, 0x0 );
	SortGridCUDA( 0x0 );
	CountingSortFullCUDA_( 0x0 );
	
	//Initialize:compute density of solid,store it into mf_restdensity[i]
	initSPH(m_restDensity, MF_type);
	OnfirstRun();
}
void FluidSystem::RunSimulateMultiCUDAFull()
{
	mint::Time start;
	start.SetSystemTime(ACC_NSEC);
	//printf("start time is %f\n", start.)
	//printf("RunSimulateMultiCUDAFull\n");
	LARGE_INTEGER t1, t2, tc;
	if (m_Frame == 1)
		m_CostTime = 0;
	QueryPerformanceFrequency(&tc);

	InitialSortCUDA(0x0, 0x0, 0x0);
	record(PTIME_INSERT, "Insert CUDA", start);
	start.SetSystemTime(ACC_NSEC);

	SortGridCUDA(0x0);
	CountingSortFullCUDA_(0x0);
	record(PTIME_SORT, "Full Sort CUDA", start);
	start.SetSystemTime(ACC_NSEC);

	QueryPerformanceCounter(&t1);

	MfPredictAdvection(m_Time);
	record(PTIME_SORT, "Predict Advection CUDA", start);
	start.SetSystemTime(ACC_NSEC);

	PressureSolve(0, NumPoints());

	if (numElasticPoints > 0)
		ComputeElasticForceCUDA();
	
	ComputePorousForceCUDA();

	MfComputeAlphaAdvanceCUDA();									//case 1
	record(PTIMEALPHA, "Alpha Advance CUDA", start);
	start.SetSystemTime(ACC_NSEC);

	LeapFrogIntegration(m_Time);
	record(PTIME_ADVANCE, "Advance CUDA", start);

	QueryPerformanceCounter(&t2);
	m_CostTime += (t2.QuadPart - t1.QuadPart)*1.0 / tc.QuadPart;
	if (m_Frame == 3000)
	{
		cout << "ave time :" << m_CostTime / m_Frame << endl;
	}

	TransferFromCUDA();	// return for rendering
}
void FluidSystem::OnfirstRun()
{
	printf("on first run\n");
	mint::Time start;
	start.SetSystemTime(ACC_NSEC);

	InitialSortCUDA(0x0, 0x0, 0x0);
	record(PTIME_INSERT, "Insert CUDA", start);
	start.SetSystemTime(ACC_NSEC);

	SortGridCUDA(0x0);
	CountingSortFullCUDA_(0x0);
	record(PTIME_SORT, "Full Sort CUDA", start);
	start.SetSystemTime(ACC_NSEC);

	MfComputePressureCUDA();                                          //case 3,5
	record(PTIME_PRESS, "Compute Pressure CUDA", start);
	start.SetSystemTime(ACC_NSEC);

	ComputeCorrectLCUDA();
	record(PTIME_PRESS, "Compute CorrectL CUDA", start);
	start.SetSystemTime(ACC_NSEC);

	//MfPredictAdvection(m_Time);
	//MfComputeDriftVelCUDA();
	//MfComputeDriftVelCUDA();
	//MfComputeDriftVelCUDA();                                          //case 1-diff
	//record(PTIMEDRIFTVEL, "Drift Velocity CUDA", start);
	//start.SetSystemTime(ACC_NSEC);

	//MfComputeAlphaAdvanceCUDA();									//case 1
	//record ( PTIMEALPHA, "Alpha Advance CUDA", start );
	//start.SetSystemTime ( ACC_NSEC );

	//MfComputeCorrectionCUDA();                                        //case5
	//record ( PTIMECORR, "Alpha Correction and Pressure CUDA", start );		
	//start.SetSystemTime ( ACC_NSEC );

	TransferFromCUDA();	// return for rendering
}
void FluidSystem::Record()
{
	mFileNum = getLastRecording() + 1;
	mFileName = getFilename(mFileNum);
	//if (mFP != 0x0) fclose(mFP);
	char name[100];
	strcpy(name, mFileName.c_str());
	ofstream out(name);
	//mFP = fopen(name, "wb");
	//if (mFP == 0x0) {
	//	printf("ERROR: Cannot write file %s\n", mFileName.c_str());
	//	exit(-1);
	//}
	//mLastPoints = 0;
	//mFileSize = 0;


	Vector3DF*  ppos = mPos;
	Vector3DF*  pvel = mVel;
	float*		pdens = mDensity;
	DWORD*		pclr = mClr;
	int*		bound = mIsBound;
	char*		dat = mPackBuf;
	int*		type = MF_type;
	int			channels;
	int			dsize;
	out << NumPointsNoBound << endl;
	//out << NumPointsNoBound<<" "<<softBoundary[0].x << " " << softBoundary[0].z 
	//	<< " " << softBoundary[0].y << " " << softBoundary[1].x 
	//	<< " " << softBoundary[1].z << " " << softBoundary[1].y <<endl;
	//fwrite ( &mNumPoints, sizeof(int), 1, mFP );
	//cout << "output file: " << mFP << endl;
	// How many channels to write? 

	//fwrite ( &channels, sizeof(int), 1, mFP ) ;

	// Write data
	//if ( channels == 2 ) {	
	//	dsize = sizeof(Vector3DF)+sizeof(DWORD);
	for (int n = 0; n < mNumPoints; n++) {
		if (*type == 1)
		{
			ppos++; pclr++; type++;
			continue;
		}

		/*if ((*ppos).y < 0)
		continue;*/
		//*(Vector3DF*) dat = *ppos++;		dat += sizeof(Vector3DF);
		//out << n << " " << ((Vector3DF*)dat)->x
		//*(DWORD*)	  dat = *pclr++;		dat += sizeof(DWORD);

		out << ppos->x << " " << ppos->z << " " << ppos->y << " ";
		if (_example == 1)
		{
			if (*type == 0) 
			{
				out << m_alpha[n*MAX_FLUIDNUM + 1] << " " << m_alpha[n*MAX_FLUIDNUM + 2]
					<< " " << m_alpha[n*MAX_FLUIDNUM + 3] << " " <<
					m_alpha[n*MAX_FLUIDNUM + 1] + m_alpha[n*MAX_FLUIDNUM + 2] +
					m_alpha[n*MAX_FLUIDNUM + 3] << " ";
			}
			else
			{
				float beta[MAX_FLUIDNUM];
				for (int k = 1; k < MAX_FLUIDNUM; ++k)
					beta[k] = m_beta[n*MAX_FLUIDNUM*MAX_SOLIDNUM + k * MAX_SOLIDNUM + *type - 2];
				out << 1 - (beta[2] + beta[3]) << " " << 1 - (beta[1] + beta[3]) << " "
					<< 1 - (beta[1] + beta[2]) <<" "<< 1 << " ";
				
			}
		}
		else
		{
			if (*type == 0)
			{
				out << 1 << " " << 1 << " " << 1 << " " << 1 << " ";
			}
			else
			{
				float beta[MAX_FLUIDNUM];
				for (int k = 1; k < MAX_FLUIDNUM; ++k)
					beta[k] = m_beta[n*MAX_FLUIDNUM*MAX_SOLIDNUM + k * MAX_SOLIDNUM + *type - 2];
				
				if (*type == 5)
					out << 1 / (1 + beta[2] + beta[3]) << " " << 1 / (1 + beta[1] + beta[3]) << " " << beta[3] / (1 + beta[3]) << " " << 1 << " ";
				else
					out << 0 << " " << 1 << " " << 0 << " " << 1 << " ";
			}
		}
		out << *type << endl;

		ppos++; pclr++; type++;

	}
	out.close();
	
	mFileSize += float(dsize * mNumPoints) / 1048576.0;

	mLastPoints = mNumPoints;

	//fflush ( mFP );
}
char dsttmp[100];
void FluidSystem::outputFile()
{
	FILE* fp;
	
	sprintf(dsttmp, "OutputData\\data_%04d.txt", nOutFrame);
	if (_access(dsttmp, 0) == -1)
		_mkdir(dsttmp);

	fp = fopen(dsttmp,"w");
	fprintf(fp,"%d\n",NumPoints());
	Vector3DF* ppos = mPos;
	for (int i = 0;i<NumPoints();i++,ppos++){
		//fprintf(fp,"%f %f %f\n",ppos->x,ppos->y,ppos->z);
		fprintf(fp,"%f %f %f",mPos[i].x,mPos[i].y,mPos[i].z);
		for (int j = 0;j<MAX_FLUIDNUM;j++)
			fprintf(fp," %f",*(m_alpha+i*MAX_FLUIDNUM + j));
		fprintf(fp," %f",m_restMass[i]);
		fprintf(fp," %f",m_restDensity[i]);
		fprintf(fp," %d\n",MF_type[i]);
	}
	fclose(fp);
	nOutFrame++;
}
void FluidSystem::storeModel(char* filename)
{
	FILE* fp;
	sprintf(dsttmp, filename, nOutFrame);
	fp = fopen(dsttmp, "w");
	fprintf(fp, "%d\n", numElasticPoints);
	Vector3DF* ppos = mPos;
	for (int i = 0; i<NumPoints(); i++, ppos++) {
		if(MF_type[i]==1)
			fprintf(fp, "%f %f %f\n", mPos[i].x, mPos[i].y, mPos[i].z);
	}
	fclose(fp);
}

void FluidSystem::LoadParticles(char* filename, Vector3DF off)
{
	fstream f;
	f.open(filename, ios::in);
	int num;
	f >> num;
	float x, y, z;
	int n = numElasticPoints;
	int p;
	for (int i = 0; i<num; i++) {
		f >> x >> y >> z;
		p = AddParticle();
		if (p != -1)
		{

			*(elasticID + p) = n;
			(mPos + p)->Set(x+off.x, y+off.y, z+off.z);

			*(m_alpha + p*MAX_FLUIDNUM) = 1.0f; *(m_alpha_pre + p*MAX_FLUIDNUM) = 1.0f;
			*(m_restMass + p) = m_fluidPMass[0];
			*(m_restDensity + p) = m_fluidDensity[0];

			*(m_visc + p) = m_fluidVisc[0];
			*(mIsBound + p) = false;
			*(MF_type + p) = 1; //1 means deformable

			*(porosity_particle + n) = porosity;

			n++;
		}

	}
	numElasticPoints = n;
	f.close();
}
int countNeighborNum(float radius, Vector3DF pos, Vector3DF* mpos, int num)
{
	Vector3DF d;
	float distance;
	int count = 0;
	for (int p = 0; p < num; ++p)
	{
		d = pos - mpos[p];
		distance = d.x*d.x+d.y*d.y+d.z*d.z;
		if (distance < radius)
			count++;
	}
	return count;
}
void FluidSystem::solveModel()
{
	int n = numElasticPoints;
	int count;
	float radius = m_Param[PSMOOTHRADIUS] * m_Param[PSMOOTHRADIUS]/(m_Param[PSIMSCALE]* m_Param[PSIMSCALE]);
	for(int p=0;p<n;++p)
	{
		count = countNeighborNum(radius, mPos[p], mPos, n);
		if (count < 8)
		{
			MF_type[p] = 0;
			numElasticPoints--;
		}
	}
}

extern bool    bPause;
void FluidSystem::Run (int width, int height)
{
	// Clear sim timers
	m_Param[ PTIME_INSERT ] = 0.0;
	m_Param[ PTIME_SORT ] = 0.0;
	m_Param[ PTIME_COUNT ] = 0.0;
	m_Param[ PTIME_PRESS ] = 0.0;
	m_Param[ PTIME_FORCE ] = 0.0;
	m_Param[ PTIME_ADVANCE ] = 0.0;
	ParamUpdateCUDA(m_Toggle[HIDEBOUND], m_Toggle[HIDEFLUID], m_Toggle[HIDESOLID],m_Toggle[HIDERIGID], restColorValue);

	mint::Time start;
	start.SetSystemTime(ACC_NSEC);
	RunSimulateMultiCUDAFull();
	//DWORD end = timeGetTime();
	//printf("simulate time %d\n", end - start);
	if ( GetYan(START_OUTPUT) && m_Frame % (int)(0.0025 / m_DT) == 0 && RecordNum <= 600) {
		//StartRecord();
		start.SetSystemTime ( ACC_NSEC );
		Record ();
		RecordNum++;
		record ( PTIME_RECORD, "Record", start );
	}
	//if((_example == 2 && m_Frame == 10000))
	//	saveParticle("save_stat.txt");
	if ( m_Toggle[PCAPTURE] && m_Frame %(int)(0.005/m_DT)==0){//controlled by '`'
		CaptureVideo ( width, height );
		/*if( m_Frame /(int)(0.005/m_DT)== 200){
			//bPause = true;
			liftup(1);
		}
		if( m_Frame /(int)(0.005/m_DT)== 215){
			//bPause = true;
			liftup(0);
		}
		if( m_Frame /(int)(0.005/m_DT)== 300){
			bPause = true;
		}*/
		
	}
	int k=10000;
	if(m_Frame == k)
		liftup(1);
	if(m_Frame == k+1400)
		liftup(0);
	//if(m_Frame == k+1050*2+750)
	//	liftup(2);

	if(m_Frame == 27000)
		bPause = !bPause;
	//if ( GetYan(START_OUTPUT)==1 && m_Frame %(int)(0.005/m_DT)==0 ){ //controlled by 'b'
	//	outputFile();
	//}

	m_Time += m_DT;
	m_Frame++;

	//outputepsilon(epsilonfile);
	//if(example == 2 && m_Frame == upframe)
	//	SetYan(CHANGE_DEN,1);
}


void FluidSystem::Exit ()
{
	//fclose(epsilonfile);

	free ( mPos );
	free ( mClr );
	free (mIsBound);
	free ( mVel );
	free ( mVelEval );
	free ( mAge );
	free ( mPressure );
	free ( mDensity );
	free ( mForce );
	free ( mClusterCell );
	free ( mGridCell );
	free ( mGridNext );
	free ( mNbrNdx );
	free ( mNbrCnt );

	//multi fluid
	free (m_alpha);
	free (m_alpha_pre);
	free (m_pressure_modify);
	free (m_vel_phrel);
	free (m_restMass);
	free (m_restDensity);
	free (m_visc);
	free (m_velxcor);
	free (m_alphagrad);
	free (MF_type);
	//free (MF_tensor);

	free (elasticID);
	free(signDistance);
	FluidClearCUDA();

	cudaExit (0,0);


}


// Allocate particle memory
void FluidSystem::AllocateParticles ( int cnt )
{
	int nump = 0;		// number to copy from previous data

	Vector3DF* srcPos = mPos;
	mPos = (Vector3DF*)		malloc ( EMIT_BUF_RATIO*cnt*sizeof(Vector3DF) );
	if ( srcPos != 0x0 )	{ memcpy ( mPos, srcPos, nump *sizeof(Vector3DF)); free ( srcPos ); }

	DWORD* srcClr = mClr;	
	mClr = (DWORD*)			malloc ( EMIT_BUF_RATIO*cnt*sizeof(DWORD) );
	if ( srcClr != 0x0 )	{ memcpy ( mClr, srcClr, nump *sizeof(DWORD)); free ( srcClr ); }
	
	int* srcIsBound = mIsBound;	
	mIsBound = (int*)			malloc ( cnt*sizeof(int) );
	if ( srcIsBound != 0x0 )	{ memcpy ( mIsBound, srcIsBound, nump *sizeof(int)); free ( srcIsBound ); }

	Vector3DF* srcVel = mVel;
	mVel = (Vector3DF*)		malloc ( EMIT_BUF_RATIO*cnt*sizeof(Vector3DF) );	
	if ( srcVel != 0x0 )	{ memcpy ( mVel, srcVel, nump *sizeof(Vector3DF)); free ( srcVel ); }

	Vector3DF* srcVelEval = mVelEval;
	mVelEval = (Vector3DF*)	malloc ( EMIT_BUF_RATIO*cnt*sizeof(Vector3DF) );	
	if ( srcVelEval != 0x0 ) { memcpy ( mVelEval, srcVelEval, nump *sizeof(Vector3DF)); free ( srcVelEval ); }

	unsigned short* srcAge = mAge;
	mAge = (unsigned short*) malloc ( EMIT_BUF_RATIO*cnt*sizeof(unsigned short) );
	if ( srcAge != 0x0 )	{ memcpy ( mAge, srcAge, nump *sizeof(unsigned short)); free ( srcAge ); }

	float* srcPress = mPressure;
	mPressure = (float*) malloc ( EMIT_BUF_RATIO*cnt*sizeof(float) );
	if ( srcPress != 0x0 ) { memcpy ( mPressure, srcPress, nump *sizeof(float)); free ( srcPress ); }	

	float* srcDensity = mDensity;
	mDensity = (float*) malloc ( EMIT_BUF_RATIO*cnt*sizeof(float) );
	if ( srcDensity != 0x0 ) { memcpy ( mDensity, srcDensity, nump *sizeof(float)); free ( srcDensity ); }	

	Vector3DF* srcForce = mForce;
	mForce = (Vector3DF*)	malloc ( EMIT_BUF_RATIO*cnt*sizeof(Vector3DF) );
	if ( srcForce != 0x0 )	{ memcpy ( mForce, srcForce, nump *sizeof(Vector3DF)); free ( srcForce ); }

	uint* srcCell = mClusterCell;
	mClusterCell = (uint*)	malloc ( EMIT_BUF_RATIO*cnt*sizeof(uint) );
	if ( srcCell != 0x0 )	{ memcpy ( mClusterCell, srcCell, nump *sizeof(uint)); free ( srcCell ); }

	uint* srcGCell = mGridCell;
	mGridCell = (uint*)	malloc ( EMIT_BUF_RATIO*cnt*sizeof(uint) );
	if ( srcGCell != 0x0 )	{ memcpy ( mGridCell, srcGCell, nump *sizeof(uint)); free ( srcGCell ); }

	uint* srcNext = mGridNext;
	mGridNext = (uint*)	malloc ( EMIT_BUF_RATIO*cnt*sizeof(uint) );
	if ( srcNext != 0x0 )	{ memcpy ( mGridNext, srcNext, nump *sizeof(uint)); free ( srcNext ); }
	
	uint* srcNbrNdx = mNbrNdx;
	mNbrNdx = (uint*)		malloc ( EMIT_BUF_RATIO*cnt*sizeof(uint) );
	if ( srcNbrNdx != 0x0 )	{ memcpy ( mNbrNdx, srcNbrNdx, nump *sizeof(uint)); free ( srcNbrNdx ); }
	
	uint* srcNbrCnt = mNbrCnt;
	mNbrCnt = (uint*)		malloc ( EMIT_BUF_RATIO*cnt*sizeof(uint) );
	if ( srcNbrCnt != 0x0 )	{ memcpy ( mNbrCnt, srcNbrCnt, nump *sizeof(uint)); free ( srcNbrCnt ); }

	m_Param[PSTAT_PMEM] = 68 * 2 * cnt;

	//multi fluid
	float* src_alpha = m_alpha;
	m_alpha = (float*)		malloc ( EMIT_BUF_RATIO*cnt*MAX_FLUIDNUM*sizeof(float));
	if (src_alpha != 0x0)	{ memcpy (m_alpha, src_alpha, nump * MAX_FLUIDNUM * sizeof(float)); free(src_alpha);}

	float* src_beta = m_beta;
	m_beta = (float*)malloc(EMIT_BUF_RATIO*cnt*MAX_FLUIDNUM * sizeof(float)*MAX_SOLIDNUM);
	if (src_beta != 0x0) { memcpy(m_beta, src_beta, nump * MAX_FLUIDNUM * sizeof(float)*MAX_SOLIDNUM); free(src_beta); }

	float* src_alpha_pre = m_alpha_pre;
	m_alpha_pre = (float*)	malloc ( EMIT_BUF_RATIO*cnt*MAX_FLUIDNUM*sizeof(float));
	if (src_alpha_pre != 0x0)	{ memcpy (m_alpha_pre, src_alpha_pre, nump * MAX_FLUIDNUM * sizeof(float)); free(src_alpha_pre);}

	float* src_pressure_modify = m_pressure_modify;
	m_pressure_modify = (float*)		malloc ( EMIT_BUF_RATIO*cnt*sizeof(float));
	if (src_pressure_modify != 0x0)	{ memcpy (m_pressure_modify, src_pressure_modify, nump * sizeof(float)); free(src_pressure_modify);}

	Vector3DF* src_vel_phrel = m_vel_phrel;
	m_vel_phrel = (Vector3DF*)		malloc ( EMIT_BUF_RATIO*cnt*MAX_FLUIDNUM*sizeof(Vector3DF));
	if (src_vel_phrel != 0x0)	{ memcpy (m_vel_phrel, src_vel_phrel, nump * MAX_FLUIDNUM * sizeof(Vector3DF)); free(src_vel_phrel);}

	float* src_restMass = m_restMass;
	m_restMass = (float*) malloc ( EMIT_BUF_RATIO*cnt*sizeof(float) );
	if ( src_restMass != 0x0 ) { memcpy ( m_restMass, src_restMass, nump *sizeof(float)); free ( src_restMass ); }	

	float* src_restDensity = m_restDensity;
	m_restDensity = (float*) malloc ( EMIT_BUF_RATIO*cnt*sizeof(float) );
	if ( src_restDensity != 0x0 ) { memcpy ( m_restDensity, src_restDensity, nump *sizeof(float)); free ( src_restDensity ); }	


	float* src_visc = m_visc;
	m_visc = (float*) malloc ( EMIT_BUF_RATIO*cnt*sizeof(float) );
	if ( src_visc != 0x0 ) { memcpy ( m_visc, src_visc, nump *sizeof(float)); free ( src_visc ); }	

	Vector3DF* src_velxcor = m_velxcor;
	m_velxcor = (Vector3DF*)		malloc ( EMIT_BUF_RATIO*cnt*sizeof(Vector3DF) );	
	if ( src_velxcor != 0x0 )	{ memcpy ( m_velxcor, src_velxcor, nump *sizeof(Vector3DF)); free ( src_velxcor ); }

	Vector3DF* src_alphagrad = m_alphagrad;
	m_alphagrad = (Vector3DF*)		malloc ( EMIT_BUF_RATIO*cnt*MAX_FLUIDNUM*sizeof(Vector3DF));
	if (src_alphagrad != 0x0)	{ memcpy (m_alphagrad, src_alphagrad, nump * MAX_FLUIDNUM * sizeof(Vector3DF)); free(src_alphagrad);}

	//elastic information
	uint* src_elasticID = elasticID;
	elasticID = (uint*)malloc(EMIT_BUF_RATIO*cnt * sizeof(uint));
	if (src_elasticID != 0x0) { memcpy(elasticID, src_elasticID, nump * sizeof(uint)); free(src_elasticID); }

	float* src_porosity = porosity_particle;
	porosity_particle = (float*)malloc(EMIT_BUF_RATIO*cnt * sizeof(float));
	if (src_porosity != 0x0) { memcpy(porosity_particle, src_porosity, nump * sizeof(float)); free(src_porosity); }

	Vector3DF* src_signDistance = signDistance;
	signDistance = (Vector3DF*)malloc(EMIT_BUF_RATIO* cnt * sizeof(Vector3DF));
	if (src_signDistance != 0x0) { memcpy(signDistance, src_signDistance, cnt * sizeof(Vector3DF)); free(src_signDistance); }

	//Project U
	int* src_mftype = MF_type;
	MF_type = (int*) malloc(EMIT_BUF_RATIO*cnt*sizeof(int));
	if(src_mftype != 0x0) {memcpy( MF_type, src_mftype, nump*sizeof(int)); free(src_mftype);}

	//End Project U
	m_Param[PSTAT_PMEM] = EMIT_BUF_RATIO * (92 + 36*MAX_FLUIDNUM)* 2 * cnt;

	
	mMaxPoints = cnt;
}

float unitMatrix[9] = {1,0,0,    0,1,0,     0,0,1};

int FluidSystem::AddParticle ()
{
	if ( mNumPoints >= mMaxPoints ) return -1;
	int n = mNumPoints;
	(mPos + n)->Set ( 0,0,0 );
	(mVel + n)->Set ( 0,0,0 );
	(mVelEval + n)->Set ( 0,0,0 );
	(mForce + n)->Set ( 0,0,0 );
	*(mPressure + n) = 0;
	*(mDensity + n) = 0;
	*(mGridNext + n) = -1;
	*(mClusterCell + n) = -1;
//#ifdef NEW_BOUND
	*(mIsBound+n) = 0;
//#endif

	//multi fluid
	memset(m_alpha + n*MAX_FLUIDNUM,0,MAX_FLUIDNUM*sizeof(float));
	memset(m_alpha_pre + n*MAX_FLUIDNUM,0,MAX_FLUIDNUM*sizeof(float));
	memset(m_pressure_modify + n,0,sizeof(float));
	memset(m_vel_phrel+n*MAX_FLUIDNUM,0,MAX_FLUIDNUM*sizeof(Vector3DF));
	memset(m_alphagrad+n*MAX_FLUIDNUM,0,MAX_FLUIDNUM*sizeof(Vector3DF));
	*(m_restMass + n) = 0;
	*(m_restDensity + n) = 0;
	*(m_visc + n) = 0;
	(m_velxcor + n)->Set(0,0,0);
	

	mNumPoints++;
	return n;
}

void FluidSystem::record ( int param, std::string name, mint::Time& start )
{
	mint::Time stop;
	stop.SetSystemTime ( ACC_NSEC );
	stop = stop - start;
	m_Param [ param ] = stop.GetMSec();
	if ( m_Toggle[PPROFILE] ) printf ("%s:  %s\n", name.c_str(), stop.GetReadableTime().c_str() );
}

void FluidSystem::AllocatePackBuf ()
{
	if ( mPackBuf != 0x0 ) free ( mPackBuf );	
	mPackBuf = (char*) malloc ( sizeof(Fluid) * mMaxPoints );
}

void FluidSystem::ClearNeighborTable ()
{
	if ( m_NeighborTable != 0x0 )	free (m_NeighborTable);
	if ( m_NeighborDist != 0x0)		free (m_NeighborDist );
	m_NeighborTable = 0x0;
	m_NeighborDist = 0x0;
	m_NeighborNum = 0;
	m_NeighborMax = 0;
}

// Ideal grid cell size (gs) = 2 * smoothing radius = 0.02*2 = 0.04
// Ideal domain size = k*gs/d = k*0.02*2/0.005 = k*8 = {8, 16, 24, 32, 40, 48, ..}
//    (k = number of cells, gs = cell size, d = simulation scale)
void FluidSystem::SetupGridAllocate ( Vector3DF min, Vector3DF max, float sim_scale, float cell_size, float border )
{
	float world_cellsize = cell_size / sim_scale;
	
	m_GridMin = min;
	m_GridMax = max;
	m_GridSize = m_GridMax;
	m_GridSize -= m_GridMin;
	m_GridRes.x = ceil ( m_GridSize.x / world_cellsize );		// Determine grid resolution
	m_GridRes.y = ceil ( m_GridSize.y / world_cellsize );
	m_GridRes.z = ceil ( m_GridSize.z / world_cellsize );
	m_GridSize.x = m_GridRes.x * cell_size / sim_scale;				// Adjust grid size to multiple of cell size
	m_GridSize.y = m_GridRes.y * cell_size / sim_scale;
	m_GridSize.z = m_GridRes.z * cell_size / sim_scale;
	m_GridDelta = m_GridRes;		// delta = translate from world space to cell #
	m_GridDelta /= m_GridSize;
	
	m_GridTotal = (int)(m_GridRes.x * m_GridRes.y * m_GridRes.z);

	// Allocate grid
	if ( m_Grid == 0x0 ) free (m_Grid);
	if ( m_GridCnt == 0x0 ) free (m_GridCnt);
	m_Grid = (uint*) malloc ( sizeof(uint*) * m_GridTotal );
	m_GridCnt = (uint*) malloc ( sizeof(uint*) * m_GridTotal );
	memset ( m_Grid, GRID_UCHAR, m_GridTotal*sizeof(uint) );
	memset ( m_GridCnt, GRID_UCHAR, m_GridTotal*sizeof(uint) );

	m_Param[PSTAT_GMEM] = 12 * m_GridTotal;		// Grid memory used

	// Number of cells to search:
	// n = (2r / w) +1,  where n = 1D cell search count, r = search radius, w = world cell width
	//
	m_GridSrch =  floor(2*(m_Param[PSMOOTHRADIUS]/sim_scale) / world_cellsize+0.001) + 1;
	if ( m_GridSrch < 2 ) m_GridSrch = 2;
	m_GridAdjCnt = m_GridSrch * m_GridSrch * m_GridSrch ;			// 3D search count = n^3, e.g. 2x2x2=8, 3x3x3=27, 4x4x4=64

	if ( m_GridSrch > 6 ) {
		printf ( "ERROR: Neighbor search is n > 6. \n " );
		exit(-1);
	}

	int cell = 0;
	for (int y=0; y < m_GridSrch; y++ ) 
		for (int z=0; z < m_GridSrch; z++ ) 
			for (int x=0; x < m_GridSrch; x++ ) 
				m_GridAdj[cell++] = ( y*m_GridRes.z + z )*m_GridRes.x +  x ;			// -1 compensates for ndx 0=empty
				

	printf ( "Adjacency table (CPU) \n");
	for (int n=0; n < m_GridAdjCnt; n++ ) {
		printf ( "  ADJ: %d, %d\n", n, m_GridAdj[n] );
	}

	if ( mPackGrid != 0x0 ) free ( mPackGrid );
	mPackGrid = (int*) malloc ( sizeof(int) * m_GridTotal );

	
}

int FluidSystem::getGridCell ( int p, Vector3DI& gc )
{
	return getGridCell ( *(mPos+p), gc );
}
int FluidSystem::getGridCell ( Vector3DF& pos, Vector3DI& gc )
{
	gc.x = (int)( (pos.x - m_GridMin.x) * m_GridDelta.x);			// Cell in which particle is located
	gc.y = (int)( (pos.y - m_GridMin.y) * m_GridDelta.y);
	gc.z = (int)( (pos.z - m_GridMin.z) * m_GridDelta.z);		
	return (int)( (gc.y*m_GridRes.z + gc.z)*m_GridRes.x + gc.x);		
}
Vector3DI FluidSystem::getCell ( int c )
{
	Vector3DI gc;
	int xz = m_GridRes.x*m_GridRes.z;
	gc.y = c / xz;				c -= gc.y*xz;
	gc.z = c / m_GridRes.x;		c -= gc.z*m_GridRes.x;
	gc.x = c;
	return gc;
}

void FluidSystem::SetupRender ()
{
	glEnable ( GL_TEXTURE_2D );

	glGenTextures ( 1, (GLuint*) mTex );
	glBindTexture ( GL_TEXTURE_2D, mTex[0] );
	glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);	
	glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );	
	glPixelStorei( GL_UNPACK_ALIGNMENT, 4);	
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB32F_ARB, 8, 8, 0, GL_RGB, GL_FLOAT, 0);

	glGenBuffersARB ( 3, (GLuint*) mVBO );

	// Construct a sphere in a VBO
	int udiv = 6;
	int vdiv = 6;
	float du = 180.0 / udiv;
	float dv = 360.0 / vdiv;
	float x,y,z, x1,y1,z1;

	float r = 1.0;

	Vector3DF* buf = (Vector3DF*) malloc ( sizeof(Vector3DF) * (udiv+2)*(vdiv+2)*2 );
	Vector3DF* dat = buf;
	
	mSpherePnts = 0;
	for ( float tilt=-90; tilt <= 90.0; tilt += du) {
		for ( float ang=0; ang <= 360; ang += dv) {
			x = sin ( ang*DEGtoRAD) * cos ( tilt*DEGtoRAD );
			y = cos ( ang*DEGtoRAD) * cos ( tilt*DEGtoRAD );
			z = sin ( tilt*DEGtoRAD ) ;
			x1 = sin ( ang*DEGtoRAD) * cos ( (tilt+du)*DEGtoRAD ) ;
			y1 = cos ( ang*DEGtoRAD) * cos ( (tilt+du)*DEGtoRAD ) ;
			z1 = sin ( (tilt+du)*DEGtoRAD );
		
			dat->x = x*r;
			dat->y = y*r;
			dat->z = z*r;
			dat++;
			dat->x = x1*r;
			dat->y = y1*r;
			dat->z = z1*r;
			dat++;
			mSpherePnts += 2;
		}
	}
	glBindBufferARB ( GL_ARRAY_BUFFER_ARB, mVBO[2] );
	glBufferDataARB ( GL_ARRAY_BUFFER_ARB, mSpherePnts*sizeof(Vector3DF), buf, GL_STATIC_DRAW_ARB);
	glVertexPointer ( 3, GL_FLOAT, 0, 0x0 );

	free ( buf );
		
	mImg.read ( "ball32.bmp", "ball32a.bmp" );

	// Enable Instacing shader
	//cgGLEnableProfile( vert_profile );
	//cgGLBindProgram ( cgVP );

	//cgGLEnableProfile( frag_profile );
	//cgGLBindProgram ( cgFP );
}

void FluidSystem::DrawGrid ()
{
	Vector3DF gd (1, 1, 1);
	Vector3DF gc;
	gd /= m_GridDelta;		
	
	glBegin ( GL_LINES );	
	for (int z=0; z <= m_GridRes.z; z++ ) {
		for (int y=0; y <= m_GridRes.y; y++ ) {
			gc.Set ( 1, y, z);	gc /= m_GridDelta;	gc += m_GridMin;
			glVertex3f ( m_GridMin.x, gc.y, gc.z );	glVertex3f ( m_GridMax.x, gc.y, gc.z );
		}
	}
	for (int z=0; z <= m_GridRes.z; z++ ) {
		for (int x=0; x <= m_GridRes.x; x++ ) {
			gc.Set ( x, 1, z);	gc /= m_GridDelta;	gc += m_GridMin;
			glVertex3f ( gc.x, m_GridMin.y, gc.z );	glVertex3f ( gc.x, m_GridMax.y, gc.z );
		}
	}
	for (int y=0; y <= m_GridRes.y; y++ ) {
		for (int x=0; x <= m_GridRes.x; x++ ) {
			gc.Set ( x, y, 1);	gc /= m_GridDelta;	gc += m_GridMin;
			glVertex3f ( gc.x, gc.y, m_GridMin.z );	glVertex3f ( gc.x, gc.y, m_GridMax.z );
		}
	}
	glEnd ();
}

void FluidSystem::DrawText ()
{
	char msg[100];

	
	Vector3DF* ppos = mPos;
	DWORD* pclr = mClr;
	Vector3DF clr;
	for (int n = 0; n < NumPoints(); n++) {
	
		sprintf ( msg, "%d", n );
		glColor4f ( (RED(*pclr)+1.0)*0.5, (GRN(*pclr)+1.0)*0.5, (BLUE(*pclr)+1.0)*0.5, ALPH(*pclr) );
		drawText3D ( ppos->x, ppos->y, ppos->z, msg );
		ppos++;
		pclr++;
	}
}

void FluidSystem::Draw ( Camera3D& cam, float rad )
{
	char* dat;
	Vector3DF* ppos;
	float* pdens;
	DWORD* pclr;
		

	glDisable ( GL_LIGHTING );

	switch ( (int) m_Param[PDRAWGRID] ) {
	case 0:
		break;
	case 1: 
		glColor4f ( 0.7, 0.7, 0.7, 0.05 );
		DrawGrid ();
		break;
	};

	if ( m_Param[PDRAWTEXT] == 1.0 ) {
		DrawText ();
	};

	// Draw Modes
	// DRAW_POINTS		0
	// DRAW_SPRITES		1
	
	switch ( (int) m_Param[PDRAWMODE] ) {
	case 0:
		//multi fluid bound
		if(GetYan(SHOW_BOUND) == 0)//controlled by '7', capture screen '`' BTW
		{
			for (int i = 0;i<NumPoints();i++)
				if (mIsBound[i]!=0)
					mPos[i].x=mPos[i].y=mPos[i].z=-1000;
		}

		glPointSize ( 6 );
		glEnable ( GL_POINT_SIZE );		
		glEnable( GL_BLEND ); 
		glBindBufferARB ( GL_ARRAY_BUFFER_ARB, mVBO[0] );
		glBufferDataARB ( GL_ARRAY_BUFFER_ARB, NumPoints()*sizeof(Vector3DF), mPos, GL_DYNAMIC_DRAW_ARB);		
		glVertexPointer ( 3, GL_FLOAT, 0, 0x0 );				
		glBindBufferARB ( GL_ARRAY_BUFFER_ARB, mVBO[1] );
		glBufferDataARB ( GL_ARRAY_BUFFER_ARB, NumPoints()*sizeof(uint), mClr, GL_DYNAMIC_DRAW_ARB);
		glColorPointer ( 4, GL_UNSIGNED_BYTE, 0, 0x0 ); 
		glEnableClientState ( GL_VERTEX_ARRAY );
		glEnableClientState ( GL_COLOR_ARRAY );          
		glNormal3f ( 0, 0.001, 1 );
		glColor3f ( 1, 1, 1 );
		//glLoadMatrixf ( view_mat );
		glDrawArrays ( GL_POINTS, 0, NumPoints() );
		glDisableClientState ( GL_VERTEX_ARRAY );
		glDisableClientState ( GL_COLOR_ARRAY );
		break;
	
	case 1: //actually used
		switch(GetYan(SHOW_BOUND)){ //controlled by 7
		case 1:
			for (int i = 0;i<NumPoints();i++)
				if (mIsBound[i]==1 || MF_type[i]==1)
					mPos[i].x=mPos[i].y=mPos[i].z=-1000;
			break;
		case 2:
			for (int i = 0;i<NumPoints();i++)
				if (mIsBound[i]==1 || MF_type[i]==0)
					mPos[i].x=mPos[i].y=mPos[i].z=-1000;
			break;
		}
		
		/*if(example==3)
			for (int i = 0;i<NumPoints();i++)
				if (mPos[i].y>23||mPos[i].y<0)
					mPos[i].y=mPos[i].z=mPos[i].x = -1000;*/
		/*if (GetYan(SAVE_STAT) == 1)
			saveParticle("save_stat.txt");

		int outP = 0;
		for (int i = 0;i<NumPoints();i++)
			if (mIsBound[i]==0 && (mPos[i].x<-100 ||mPos[i].y<-100||mPos[i].z<-100))
				outP++;*/

		glEnable ( GL_LIGHTING );		
		glEnable(GL_BLEND); 
		glEnable(GL_ALPHA_TEST); 
		glAlphaFunc( GL_GREATER, 0.5 ); 
		glEnable ( GL_COLOR_MATERIAL );
		glColorMaterial ( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE );
				
		// Point sprite size
	    glPointSize ( 48 );
		glEnable ( GL_POINT_SIZE );		
		glEnable(GL_POINT_SPRITE_ARB); 		
		{
			float quadratic[] =  { 0.0f, 0.3f, 0.00f };
			glEnable (  GL_POINT_DISTANCE_ATTENUATION  );
			glPointParameterfvARB(  GL_POINT_DISTANCE_ATTENUATION, quadratic );
		}
		//float maxSize = 64.0f;
		//if(example==3){
		//	glGetFloatv( GL_POINT_SIZE_MAX_ARB, &maxSize );
		//	glPointSize( maxSize );
		//	glPointParameterfARB( GL_POINT_SIZE_MAX_ARB, maxSize );
		//}
		glPointParameterfARB( GL_POINT_SIZE_MIN_ARB, 1.0f );

		// Texture and blending mode
		glEnable ( GL_TEXTURE_2D );
		glBindTexture ( GL_TEXTURE_2D, mImg.getID() );
		glTexEnvi (GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
		glTexEnvf (GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_BLEND );
		glBlendFunc ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) ;

		// Point buffers
		glBindBufferARB ( GL_ARRAY_BUFFER_ARB, mVBO[0] );
		glBufferDataARB ( GL_ARRAY_BUFFER_ARB, NumPoints()*sizeof(Vector3DF), mPos, GL_DYNAMIC_DRAW_ARB);		
		glVertexPointer ( 3, GL_FLOAT, 0, 0x0 );				
		glBindBufferARB ( GL_ARRAY_BUFFER_ARB, mVBO[1] );
		glBufferDataARB ( GL_ARRAY_BUFFER_ARB, NumPoints()*sizeof(uint), mClr, GL_DYNAMIC_DRAW_ARB);
		glColorPointer ( 4, GL_UNSIGNED_BYTE, 0, 0x0 ); 
		glEnableClientState ( GL_VERTEX_ARRAY );
		glEnableClientState ( GL_COLOR_ARRAY );
		  
		// Render - Point Sprites
		glNormal3f ( 0, 1, 0.001  );
		glColor3f ( 1, 1, 1 );
		glDrawArrays ( GL_POINTS, 0, NumPoints() );

		// Restore state
		glDisableClientState ( GL_VERTEX_ARRAY );
		glDisableClientState ( GL_COLOR_ARRAY );
		glDisable (GL_POINT_SPRITE_ARB); 
		glDisable ( GL_ALPHA_TEST );
		glDisable ( GL_TEXTURE_2D );
		glDepthMask( GL_TRUE );   

		break;
	case 2:
		if(GetYan(SHOW_BOUND) == 0)
		{
			for (int i = 0;i<NumPoints();i++)
				if (mIsBound[i]!=0)
					mPos[i].x=mPos[i].y=mPos[i].z=-1000;
		}

		// Notes:
		// # particles, time(Render), time(Total), time(Sim), Render Overhead (%)
		//  250000,  12, 110, 98,  10%   - Point sprites
		//  250000,  36, 146, 110, 24%   - Direct rendering (drawSphere)
		//  250000, 140, 252, 110, 55%   - Batch instancing
		glEnable ( GL_LIGHTING );
		ppos = mPos;
		pclr = mClr;
		pdens = mDensity;
		
		for (int n = 0; n < NumPoints(); n++) 
		if (mIsBound[n]==0)
		{
			glPushMatrix ();
			glTranslatef ( ppos->x, ppos->y, ppos->z );		
			glScalef ( rad, rad, rad );			
			glColor4f ( RED(*pclr), GRN(*pclr), BLUE(*pclr), ALPH(*pclr) );
			drawSphere ();
			glPopMatrix ();		
			ppos++;
			pclr++;
		}
		// --- HARDWARE INSTANCING
		/* cgGLEnableProfile( vert_profile );		
		// Sphere VBO
		glBindBufferARB ( GL_ARRAY_BUFFER_ARB, mVBO[2] );
		glVertexPointer ( 3, GL_FLOAT, 0, 0x0 );		
		glEnableClientState ( GL_VERTEX_ARRAY );
	
		glColor4f( 1,1,1,1 );

		CGparameter uParam = cgGetNamedParameter( cgVP, "modelViewProj" );
		glLoadMatrixf ( view_mat );
		cgGLSetStateMatrixParameter( uParam, CG_GL_MODELVIEW_PROJECTION_MATRIX, CG_GL_MATRIX_IDENTITY ); 

		uParam = cgGetNamedParameter( cgVP, "transformList" );
		int batches = NumPoints() / 768;
		int noff = 0;		
		for (int n=0; n < batches; n++ ) {
			cgGLSetParameterArray3f ( uParam, 0, 768, (float*) (mPos + noff) ); 
			glDrawArraysInstancedARB ( GL_TRIANGLE_STRIP, 0, mSpherePnts, 768 );
			noff += 768;
		}
		cgGLDisableProfile( vert_profile );
		glDisableClientState ( GL_VERTEX_ARRAY );
		glDisableClientState ( GL_COLOR_ARRAY );  */

		//--- Texture buffer technique
		/*
		uParam = cgGetNamedParameter( cgVP, "transformList");
		cgGLSetTextureParameter ( uParam, mTex[0] );
		cgGLEnableTextureParameter ( uParam );
		uParam = cgGetNamedParameter( cgVP, "primCnt");
		cgGLSetParameter1f ( uParam, NumPoints() );		
		glBindTexture ( GL_TEXTURE_2D, mTex[0] );
		glTexImage2D ( GL_TEXTURE_2D, 0, GL_RGB32F_ARB, 2048, int(NumPoints()/2048)+1, 0, GL_RGB, GL_FLOAT, mPos );
		glBindTexture ( GL_TEXTURE_2D, 0x0 );
		glFinish ();*/		
		break;
	};

	
	//-------------------------------------- DEBUGGING
	// draw neighbors of particle i
		/*int i = 320;
		int j, jndx = (mNbrList + i )->first;
		for (int nbr=0; nbr < (mNbrList+i)->num; nbr++ ) {			
			j = *(m_NeighborTable+jndx);
			ppos = (mPos + j );
			glPushMatrix ();
			glTranslatef ( ppos->x, ppos->y, ppos->z );		
			glScalef ( 0.25, 0.25, 0.25 );			
			glColor4f ( 0, 1, 0, 1);		// green
			drawSphere ();
			glPopMatrix ();		
			jndx++;
		}
		// draw particles in grid cells of i
		Vector3DF jpos;
		Grid_FindCells ( i );
		for (int cell=0; cell < 8; cell++) {
			j = m_Grid [ *(mClusterCell+i) + m_GridAdj[cell] ];			
			while ( j != -1 ) {
				if ( i==j ) { j = *(mGridNext+j); continue; }
				jpos = *(mPos + j);
				glPushMatrix ();
				glTranslatef ( jpos.x, jpos.y, jpos.z );		
				glScalef ( 0.22, 0.22, 0.22 );
				glColor4f ( 1, 1, 0, 1);		// yellow
				drawSphere ();
				glPopMatrix ();
				j = *(mGridNext+j);
			}
		}

		// draw grid cells of particle i		
		float poff = m_Param[PSMOOTHRADIUS] / m_Param[PSIMSCALE];
		int gx = (int)( (-poff + ppos->x - m_GridMin.x) * m_GridDelta.x);		// Determine grid cell
		int gy = (int)( (-poff + ppos->y - m_GridMin.y) * m_GridDelta.y);
		int gz = (int)( (-poff + ppos->z - m_GridMin.z) * m_GridDelta.z);
		Vector3DF gd (1, 1, 1);
		Vector3DF gc;
		gd /= m_GridDelta;

		*/

	// Error particles (debugging)
	/*for (int n=0; n < NumPoints(); n++) {
		if ( ALPH(*(mClr+n))==0.9 ) 
			DrawParticle ( n, 12, 14, Vector3DF(1,0,0) );
	}

	// Draw selected particle
	DrawNeighbors ( mSelected );
	DrawParticle ( mSelected, 8, 12, Vector3DF(1,1,1) );
	DrawCircle ( *(mPos+mSelected), m_Param[PSMOOTHRADIUS]/m_Param[PSIMSCALE], Vector3DF(1,1,0), cam );
	Vector3DI gc;
	int gs = getGridCell ( mSelected, gc );	// Grid cell of selected
	
	glDisable ( GL_DEPTH_TEST );	
	glColor3f ( 0.8, 0.8, 0.9 );
	gs = *(mClusterCell + mSelected);		// Cluster cell
	for (int n=0; n < m_GridAdjCnt; n++ ) {		// Cluster group
		gc = getCell ( gs + m_GridAdj[n] );	DrawCell ( gc.x, gc.y, gc.z );
	}
	glColor3f ( 1.0, 1.0, 1.0 );
	DrawCell ( gc.x, gc.y, gc.z );
	glEnable ( GL_DEPTH_TEST );*/
}

std::string FluidSystem::getFilename ( int n )
{
	char name[100];
	sprintf ( name, "OutputData%d\\particles%04d.dat", _example, n );
	return name;
}

void FluidSystem::StartRecord ()
{
	mFileNum = getLastRecording () + 1;	
	mFileName = getFilename ( mFileNum );
	if ( mFP != 0x0 ) fclose ( mFP );
	char name[100];
	strcpy ( name, mFileName.c_str() );
	mFP = fopen ( name, "wb" );		
	if ( mFP == 0x0 ) {
		printf ( "ERROR: Cannot write file %s\n", mFileName.c_str() );
		exit ( -1 );
	}
	mLastPoints = 0;
	mFileSize = 0;
}

int FluidSystem::getLastRecording ()
{
	FILE* fp;
	int num = 0;
	fp = fopen ( getFilename(num).c_str(), "rb" );	
	while ( fp != 0x0 ) {			// skip existing recordings
		fclose ( fp );
		num++;
		fp = fopen ( getFilename(num).c_str(), "rb" );	
	}
	return num-1;
}

std::string FluidSystem::getModeStr ()
{
	char buf[100];

	switch ( (int) m_Param[PMODE] ) {
	case RUN_SEARCH:		sprintf ( buf, "SEARCH ONLY (CPU)" );		break;
	case RUN_VALIDATE:		sprintf ( buf, "VALIDATE GPU to CPU");		break;
	case RUN_CPU_SLOW:		sprintf ( buf, "SIMULATE CPU Slow");		break;
	case RUN_CPU_GRID:		sprintf ( buf, "SIMULATE CPU Grid");		break;
	case RUN_CUDA_RADIX:	sprintf ( buf, "SIMULATE CUDA Radix Sort");	break;
	case RUN_CUDA_INDEX:	sprintf ( buf, "SIMULATE CUDA Index Sort" ); break;
	case RUN_CUDA_FULL:	sprintf ( buf, "SIMULATE CUDA Full Sort" );	break;
	case RUN_CUDA_CLUSTER:	sprintf ( buf, "SIMULATE CUDA Clustering" );	break;
	case RUN_PLAYBACK:		sprintf ( buf, "PLAYBACK (%s)", mFileName.c_str() ); break;
	};
	//sprintf ( buf, "RECORDING (%s, %.4f MB)", mFileName.c_str(), mFileSize ); break;
	return buf;
};

int FluidSystem::SelectParticle ( int x, int y, int wx, int wy, Camera3D& cam )
{
	Vector4DF pnt;
	Vector3DF* ppos = mPos;
	
	for (int n = 0; n < NumPoints(); n++ ) {
		pnt = cam.project ( *ppos );
		pnt.x = (pnt.x+1.0)*0.5 * wx;
		pnt.y = (pnt.y+1.0)*0.5 * wy;

		if ( x > pnt.x-8 && x < pnt.x+8 && y > pnt.y-8 && y < pnt.y+8 ) {
			mSelected = n;
			return n;
		}
		ppos++;
	}
	mSelected = -1;
	return -1;
}

void FluidSystem::DrawParticleInfo ( int p )
{
	char disp[256];

	glColor4f ( 1.0, 1.0, 1.0, 1.0 );
	sprintf ( disp, "Particle: %d", p );		drawText ( 10, 20, disp ); 

	Vector3DI gc;
	int gs = getGridCell ( p, gc );
	sprintf ( disp, "Grid Cell:    <%d, %d, %d> id: %d", gc.x, gc.y, gc.z, gs );		drawText ( 10, 40, disp ); 

	int cc = *(mClusterCell + p);
	gc = getCell ( cc );
	sprintf ( disp, "Cluster Cell: <%d, %d, %d> id: %d", gc.x, gc.y, gc.z, cc );		drawText ( 10, 50, disp ); 

	sprintf ( disp, "Neighbors:    " );
	int cnt = *(mNbrCnt + p);
	int ndx = *(mNbrNdx + p);
	for ( int n=0; n < cnt; n++ ) {
		sprintf ( disp, "%s%d, ", disp, m_NeighborTable[ ndx ] );
		ndx++;
	}
	drawText ( 10, 70, disp );

	if ( cc != -1 ) {
		sprintf ( disp, "Cluster Group: ");		drawText ( 10, 90, disp);
		int cadj;
		int stotal = 0;
		for (int n=0; n < m_GridAdjCnt; n++ ) {		// Cluster group
			cadj = cc+m_GridAdj[n];
			gc = getCell ( cadj );
			sprintf ( disp, "<%d, %d, %d> id: %d, cnt: %d ", gc.x, gc.y, gc.z, cc+m_GridAdj[n], m_GridCnt[ cadj ] );	drawText ( 20, 100+n*10, disp );
			stotal += m_GridCnt[cadj];
		}

		sprintf ( disp, "Search Overhead: %f (%d of %d), %.2f%% occupancy", float(stotal)/ cnt, cnt, stotal, float(cnt)*100.0/stotal );
		drawText ( 10, 380, disp );
	}	
}

void FluidSystem::SetupKernels ()
{
	m_Param [ PDIST ] = pow ( m_Param[PMASS] / m_Param[PRESTDENSITY], 1/3.0 );
	m_R2 = m_Param [PSMOOTHRADIUS] * m_Param[PSMOOTHRADIUS];
	m_Poly6Kern = 315.0f / (64.0f * 3.141592 * pow( m_Param[PSMOOTHRADIUS], 9) );	// Wpoly6 kernel (denominator part) - 2003 Muller, p.4
	m_SpikyKern = -45.0f / (3.141592 * pow( m_Param[PSMOOTHRADIUS], 6) );			// Laplacian of viscocity (denominator): PI h^6
	m_LapKern = 45.0f / (3.141592 * pow( m_Param[PSMOOTHRADIUS], 6) );
	CubicSplineKern1 = 1 / (4 * 3.141592*pow(m_Param[PSMOOTHRADIUS], 3));
	CubicSplineKern2 = 1 / (3.141592*pow(m_Param[PSMOOTHRADIUS], 3));
	gradCubicSplineKern1 = -3 / (4 * 3.141592*pow(m_Param[PSMOOTHRADIUS], 4));
	gradCubicSplineKern2 = 1 / (3.141592*pow(m_Param[PSMOOTHRADIUS], 4));

}

void FluidSystem::SetupDefaultParams ()
{
	//  Range = +/- 10.0 * 0.006 (r) =	   0.12			m (= 120 mm = 4.7 inch)
	//  Container Volume (Vc) =			   0.001728		m^3
	//  Rest Density (D) =				1000.0			kg / m^3
	//  Particle Mass (Pm) =			   0.00020543	kg						(mass = vol * density)
	//  Number of Particles (N) =		4000.0
	//  Water Mass (M) =				   0.821		kg (= 821 grams)
	//  Water Volume (V) =				   0.000821     m^3 (= 3.4 cups, .21 gals)
	//  Smoothing Radius (R) =             0.02			m (= 20 mm = ~3/4 inch)
	//  Particle Radius (Pr) =			   0.00366		m (= 4 mm  = ~1/8 inch)
	//  Particle Volume (Pv) =			   2.054e-7		m^3	(= .268 milliliters)
	//  Rest Distance (Pd) =			   0.0059		m
	//
	//  Given: D, Pm, N
	//    Pv = Pm / D			0.00020543 kg / 1000 kg/m^3 = 2.054e-7 m^3	
	//    Pv = 4/3*pi*Pr^3    cuberoot( 2.054e-7 m^3 * 3/(4pi) ) = 0.00366 m
	//     M = Pm * N			0.00020543 kg * 4000.0 = 0.821 kg		
	//     V =  M / D              0.821 kg / 1000 kg/m^3 = 0.000821 m^3
	//     V = Pv * N			 2.054e-7 m^3 * 4000 = 0.000821 m^3
	//    Pd = cuberoot(Pm/D)    cuberoot(0.00020543/1000) = 0.0059 m 
	//
	// Ideal grid cell size (gs) = 2 * smoothing radius = 0.02*2 = 0.04
	// Ideal domain size = k*gs/d = k*0.02*2/0.005 = k*8 = {8, 16, 24, 32, 40, 48, ..}
	//    (k = number of cells, gs = cell size, d = simulation scale)

	// "The viscosity coefficient is the dynamic viscosity, visc > 0 (units Pa.s), 
	// and to include a reasonable damping contribution, it should be chosen 
	// to be approximately a factor larger than any physical correct viscosity 
	// coefficient that can be looked up in the literature. However, care should 
	// be taken not to exaggerate the viscosity coefficient for fluid materials.
	// If the contribution of the viscosity force density is too large, the net effect 
	// of the viscosity term will introduce energy into the system, rather than 
	// draining the system from energy as intended."
	//    Actual visocity of water = 0.001 Pa.s    // viscosity of water at 20 deg C.

	m_Time = 0;							// Start at T=0
	m_DT = 0.003;	

	m_Param [ PSIMSCALE ] =		0.005;//m_Param [ PSIMSCALE ] =		0.005;			// unit size
	m_Param [ PVISC ] =			0.35;			// pascal-second (Pa.s) = 1 kg m^-1 s^-1  (see wikipedia page on viscosity)
	m_Param [ PRESTDENSITY ] =	600.0;			// kg / m^3
	m_Param [ PSPACING ]	=	0.0;			// spacing will be computed automatically from density in most examples (set to 0 for autocompute)
	m_Param [ PMASS ] =			0.00020543;		// kg
	m_Param [ PRADIUS ] =		0.02;			// m
	m_Param [ PDIST ] =			0.0059;			// m
	//m_Param [ PSMOOTHRADIUS ] =	0.005;//
	m_Param [ PSMOOTHRADIUS ] =	0.01/scaleP3;//			// m 

	m_Param [ PINTSTIFF ] =		1.5;
	m_Param [ PEXTSTIFF ] =		50000.0;
	m_Param [ PEXTDAMP ] =		100.0;
	m_Param [ PACCEL_LIMIT ] =	150.0;			// m / s^2
	m_Param [ PVEL_LIMIT ] =	3.0;			// m / s
	m_Param [ PMAX_FRAC ] = 1.0;
	m_Param [ PPOINT_GRAV_AMT ] = 0.0;

	m_Param [ PGROUND_SLOPE ] = 0.0;
	m_Param [ PFORCE_MIN ] = 0.0;
	m_Param [ PFORCE_MAX ] = 0.0;	
	m_Param [ PFORCE_FREQ ] = 8.0;	
	m_Toggle [ PWRAP_X ] = false;
	m_Toggle [ PWALL_BARRIER ] = false;
	m_Toggle [ PLEVY_BARRIER ] = false;
	m_Toggle [ PDRAIN_BARRIER ] = false;

	m_Param [ PSTAT_NBRMAX ] = 0 ;
	m_Param [ PSTAT_SRCHMAX ] = 0 ;
	
	m_Vec [ PPOINT_GRAV_POS ].Set ( 0, 50, 0 );
	m_Vec [ PPLANE_GRAV_DIR ].Set ( 0, -9.8, 0 );
	m_Vec [ PEMIT_POS ].Set ( 0, 0, 0 );
	m_Vec [ PEMIT_RATE ].Set ( 0, 0, 0 );
	m_Vec [ PEMIT_ANG ].Set ( 0, 90, 1.0 );
	m_Vec [ PEMIT_DANG ].Set ( 0, 0, 0 );

	// Default sim config
	m_Toggle [ PRUN ] = true;				// Run integrator
	m_Param [PGRIDSIZE] = m_Param[PSMOOTHRADIUS] * 2;
	m_Param [PDRAWMODE] = 1;				// Sprite drawing
	m_Param [PDRAWGRID] = 0;				// No grid 
	m_Param [PDRAWTEXT] = 0;				// No text

	// Load settings from XML (overwrite the above defaults)
	ParseXML ( "Fluid", 0, false );

	//Multifluid: component number in use
	m_Param [ FLUID_CATNUM ] = MAX_FLUIDNUM;
	for(int i=0;i<MAX_FLUIDNUM;i++)
	{
		m_fluidPMass[i]=0.00004279792;
		m_fluidDensity[i]=1000;
		m_fluidVisc[i]=0.35;
	}
	m_fluidDiffusion=0.0;

}

void FluidSystem::ParseXML_Bound (std::string name, int boundnum)
{
#ifdef NEW_BOUND
	xml.setBase ( name, boundnum );
	xml.assignValueD ( &m_Param[PBMASS],			"Mass" );
	xml.assignValueD ( &m_Param[PBSTIFF],			"BoundStiff" );
	xml.assignValueD ( &m_Param[PBVISC],			"Viscosity" );
	xml.assignValueD ( &m_Param[PBRESTDENSITY],		"RestDensity" );
#endif
}

void FluidSystem::ParseXML ( std::string name, int id, bool bStart )
{
	xml.setBase ( name, id );

	xml.assignValueD ( &m_DT, "DT" );
	xml.assignValueStr ( mSceneName, "Name" );
	if (bStart)	xml.assignValueD ( &m_Param[PNUM],			"Num" );
	xml.assignValueD ( &m_Param[PGRID_DENSITY],	"GridDensity" );
	xml.assignValueD ( &m_Param[PSIMSCALE],		"SimScale" );
	xml.assignValueD ( &m_Param[PVISC],			"Viscosity" );
	xml.assignValueD ( &m_Param[PRESTDENSITY],	"RestDensity" );
	xml.assignValueD ( &m_Param[PSPACING],		"Spacing" );
	xml.assignValueD ( &m_Param[PMASS],			"Mass" );
	xml.assignValueD ( &m_Param[PRADIUS],		"Radius" );
	xml.assignValueD ( &m_Param[PDIST],			"SearchDist" );
	xml.assignValueD ( &m_Param[PINTSTIFF],		"IntStiff" );
	xml.assignValueD ( &m_Param[PEXTSTIFF],		"BoundStiff" );
	xml.assignValueD ( &m_Param[PEXTDAMP],		"BoundDamp" );
	xml.assignValueD ( &m_Param[PACCEL_LIMIT],	"AccelLimit" );
	xml.assignValueD ( &m_Param[PVEL_LIMIT],	"VelLimit" );
	xml.assignValueD ( &m_Param[PPOINT_GRAV_AMT],	"PointGravAmt" );	
	xml.assignValueD ( &m_Param[PGROUND_SLOPE],	"GroundSlope" );
	xml.assignValueD ( &m_Param[PFORCE_MIN],	"WaveForceMin" );
	xml.assignValueD ( &m_Param[PFORCE_MAX],	"WaveForceMax" );
	xml.assignValueD ( &m_Param[PFORCE_FREQ],	"WaveForceFreq" );
	xml.assignValueD ( &m_Param[PDRAWMODE],		"DrawMode" );
	xml.assignValueD ( &m_Param[PDRAWGRID],		"DrawGrid" );
	xml.assignValueD ( &m_Param[PDRAWTEXT],		"DrawText" );
	xml.assignValueD ( &m_Param[PSMOOTHRADIUS], "SmoothRadius" );
	
	xml.assignValueV3 ( &m_Vec[PVOLMIN],		"VolMin" );
	xml.assignValueV3 ( &m_Vec[PVOLMAX],		"VolMax" );
	xml.assignValueV3 ( &m_Vec[PINITMIN],		"InitMin" );
	xml.assignValueV3 ( &m_Vec[PINITMAX],		"InitMax" );
	xml.assignValueV3 ( &m_Vec[PPOINT_GRAV_POS],	"PointGravPos" );
	xml.assignValueV3 ( &m_Vec[PPLANE_GRAV_DIR],	"PlaneGravDir" );
}

void FluidSystem::ParseMFXML ( std::string name, int id, bool bStart )
{
	xml.setBase ( name, id );

	xml.assignValueD ( &m_DT, "DT" );
	xml.assignValueStr ( mSceneName, "Name" );
	if (bStart)	xml.assignValueD ( &m_Param[PNUM],			"Num" );
	xml.assignValueD ( &m_Param[PGRID_DENSITY],	"GridDensity" );
	xml.assignValueD ( &m_Param[PSIMSCALE],		"SimScale" );
	xml.assignValueD ( &m_Param[PVISC],			"Viscosity" );
	xml.assignValueD ( &m_Param[PRESTDENSITY],	"RestDensity" );
	xml.assignValueD ( &m_Param[PSPACING],		"Spacing" );
	xml.assignValueD ( &m_Param[PMASS],			"Mass" );
	xml.assignValueD ( &m_Param[PRADIUS],		"Radius" );
	xml.assignValueD ( &m_Param[PDIST],			"SearchDist" );
	xml.assignValueD ( &m_Param[PINTSTIFF],		"IntStiff" );
	xml.assignValueD ( &m_Param[PEXTSTIFF],		"BoundStiff" );
	xml.assignValueD ( &m_Param[PEXTDAMP],		"BoundDamp" );
	xml.assignValueD ( &m_Param[PACCEL_LIMIT],	"AccelLimit" );
	xml.assignValueD ( &m_Param[PVEL_LIMIT],	"VelLimit" );
	xml.assignValueD ( &m_Param[PPOINT_GRAV_AMT],	"PointGravAmt" );	
	xml.assignValueD ( &m_Param[PGROUND_SLOPE],	"GroundSlope" );
	xml.assignValueD ( &m_Param[PFORCE_MIN],	"WaveForceMin" );
	xml.assignValueD ( &m_Param[PFORCE_MAX],	"WaveForceMax" );
	xml.assignValueD ( &m_Param[PFORCE_FREQ],	"WaveForceFreq" );
	xml.assignValueD ( &m_Param[PDRAWMODE],		"DrawMode" );
	xml.assignValueD ( &m_Param[PDRAWGRID],		"DrawGrid" );
	xml.assignValueD ( &m_Param[PDRAWTEXT],		"DrawText" );
	xml.assignValueD ( &m_Param[PSMOOTHRADIUS], "SmoothRadius" );
	
	xml.assignValueV3 ( &m_Vec[PVOLMIN],		"VolMin" );
	xml.assignValueV3 ( &m_Vec[PVOLMAX],		"VolMax" );
	xml.assignValueV3 ( &m_Vec[PINITMIN],		"InitMin" );
	xml.assignValueV3 ( &m_Vec[PINITMAX],		"InitMax" );
	xml.assignValueV3 ( &m_Vec[PPOINT_GRAV_POS],	"PointGravPos" );
	xml.assignValueV3 ( &m_Vec[PPLANE_GRAV_DIR],	"PlaneGravDir" );
	xml.assignValueV3 ( &m_Vec[PBOUNDMIN], "sb3");
	xml.assignValueV3 ( &m_Vec[PBOUNDMAX], "sb6");

	xml.assignValueV3( &volumes[0], "VolMin0");
	xml.assignValueV3( &volumes[1], "VolMax0");
	xml.assignValueV3( &volumes[2], "VolMin1");
	xml.assignValueV3( &volumes[3], "VolMax1");
	xml.assignValueV3( &volumes[4], "VolMin2");
	xml.assignValueV3( &volumes[5], "VolMax2");
	xml.assignValueV3(&volumes[6], "VolMin3");
	xml.assignValueV3(&volumes[7], "VolMax3");
	xml.assignValueV3(&volumes[8], "VolMin4");
	xml.assignValueV3(&volumes[9], "VolMax4");

	xml.assignValueV3(&softBoundary[0], "BoundMin");
	xml.assignValueV3(&softBoundary[1], "BoundMax");

	xml.assignValueD(&scaleP, "ScaleP");
	xml.assignValueD(&scaledis,"ScaleDis");
	xml.assignValueD(&m_Param[FLUID_CATNUM],"FluidCount");
	
	xml.assignValueD(&vfactor,"BoundViscFactor");
	xml.assignValueD(&fpfactor, "fluidPressureFactor");
	xml.assignValueD(&spfactor, "solidPressureFactor");

	xml.assignValueD(&bdamp,   "BoundXZdamp");

	xml.assignValueV3(&mb1,"mb3");
	xml.assignValueV3(&mb2,"mb6");
	xml.assignValueV4(&massRatio,"MassRatio");
	xml.assignValueV4(&densityRatio,"DensityRatio");
	xml.assignValueV4(&viscRatio,"ViscRatio");

	loadwhich = xml.getValueI("LoadWhich");
	upframe = xml.getValueI("Upframe");
	xml.assignValueV3(&cont, "Cont");
	xml.assignValueF(&relax, "Relax");
	xml.assignValueF(&poroDeformStrength, "poroDeformStrength");
	xml.assignValueF(&capillary, "capillary");
	xml.assignValueF(&Relax2, "Relax2");

	xml.assignValueF(&SurfaceTensionRatio, "SurfaceTension");
	xml.assignValueV4(&colorValue, "ColorValue");

	xml.assignValueV3(&emit[0],"emit3");
	xml.assignValueV3(&emit[1],"emit6");
	xml.assignValueF(&capillaryForceRatio, "capillaryForceRatio");

	panr = xml.getValueF("Panr");
	omega = xml.getValueF("Omega");
	//solid
	maxNeighborNum = xml.getValueI("maxNeighborNum");
	miu = xml.getValueF("miu");
	lambda = xml.getValueF("lambda");
	porosity = xml.getValueF("porosity");
	m_Param[PERMEABILITY] = xml.getValueF("permeability");
	//cout << "permeability:" << setprecision(15)<<m_Param[PERMEABILITY];
	for (int k = 0; k < MAX_FLUIDNUM; ++k)
	{
		xml.assignValueV4(&permeabilityRatio, string("permeabilityRatio") + to_string(k + 1));
		m_Permeability[k*MAX_SOLIDNUM + 0] = permeabilityRatio.x*m_Param[PERMEABILITY];
		m_Permeability[k*MAX_SOLIDNUM + 1] = permeabilityRatio.y*m_Param[PERMEABILITY];
		m_Permeability[k*MAX_SOLIDNUM + 2] = permeabilityRatio.z*m_Param[PERMEABILITY];
		m_Permeability[k*MAX_SOLIDNUM + 3] = permeabilityRatio.w*m_Param[PERMEABILITY];
		cout << string("permeabilityRatio") + to_string(k) << m_Permeability[k*MAX_SOLIDNUM + 0] << " "
		<< m_Permeability[k*MAX_SOLIDNUM + 1] << " " << m_Permeability[k*MAX_SOLIDNUM + 2] << " " << m_Permeability[k*MAX_SOLIDNUM + 3] << endl;
		
	}
	for (int k = 0; k < MAX_FLUIDNUM; ++k)
	{
		xml.assignValueV4(&pressRatio, string("pressureRatio") + to_string(k + 1));
		pressureRatio[k*MAX_SOLIDNUM + 0] = pressRatio.x;
		pressureRatio[k*MAX_SOLIDNUM + 1] = pressRatio.y;
		pressureRatio[k*MAX_SOLIDNUM + 2] = pressRatio.z;
		pressureRatio[k*MAX_SOLIDNUM + 3] = pressRatio.w;
		cout << string("pressureRatio") + to_string(k)<< pressureRatio[k*MAX_SOLIDNUM + 0] << " "
			<< pressureRatio[k*MAX_SOLIDNUM + 1] << " " << pressureRatio[k*MAX_SOLIDNUM + 2] << " " << pressureRatio[k*MAX_SOLIDNUM + 3] << endl;

	}
	Vector4DF v;
	xml.assignValueV4(&v, "bulkModulus");
	bulkModulus_porous = v.x;
	bulkModulus_grains = v.y;
	bulkModulus_solid = v.z;
	bulkModulus_fluid = v.w;

	emitSpeed = mb2.z;
	emitangle = emit[0].x;
	emitcircle = emit[0].y;
	emitposx = emit[0].z;
	emitposy = emit[1].x;
	emitposz = emit[1].y;
	emitfreq = emit[1].z;
	printf("Emit param: %f %f %f %f %f %f %f\n",emitSpeed,emitangle,emitcircle,emitposx,emitposy,emitposz,emitfreq);
}


void FluidSystem::SetupSpacing ()
{
	m_Param [ PSIMSIZE ] = m_Param [ PSIMSCALE ] * (m_Vec[PVOLMAX].z - m_Vec[PVOLMIN].z);	
	
	if ( m_Param[PSPACING] == 0 ) {
		// Determine spacing from density
		m_Param [PDIST] = pow ( m_Param[PMASS] / m_Param[PRESTDENSITY], 1/3.0 );	

		m_Param [PSPACING] = m_Param [ PDIST ]*scaledis / m_Param[ PSIMSCALE ];			
	} else {
		// Determine density from spacing
		m_Param [PDIST] = m_Param[PSPACING] * m_Param[PSIMSCALE] / 0.87;
		m_Param [PRESTDENSITY] = m_Param[PMASS] / pow ( m_Param[PDIST], 3.0 );
	}
	printf ( "Add Particles. Density: %f, Spacing: %f, PDist: %f\n", m_Param[PRESTDENSITY], m_Param [ PSPACING ], m_Param[ PDIST ] );

	// Particle Boundaries

	//m_Vec[PBOUNDMIN].Set(softBoundary[0].x,softBoundary[0].y,softBoundary[0].z);
	//m_Vec[PBOUNDMAX].Set(softBoundary[1].x,softBoundary[1].y,softBoundary[1].z);
}

void FluidSystem::CaptureVideo (int width, int height)
{
	Image img ( width, height, 3, 8 );			// allocates pixel memory
	
	FILE *fScreenshot;
	char fileName[64];
	
	sprintf( fileName, "ScreenOutput\\screen_%04d.bmp", m_Frame );
	
	fScreenshot = fopen( fileName, "wb");
												// record frame buffer directly to image pixels
	glReadPixels( 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, img.getPixelData() );	
  
	img.writeBMP ( fScreenshot );				// write bmp format

	fflush ( fScreenshot );						// close file
	fclose ( fScreenshot );

	//convert to BGR format    
	/*unsigned char temp;
	int i = 0;
	while (i < nSize) {
		temp = pixels[i];       //grab blue
		pixels[i] = pixels[i+2];//assign red to blue
		pixels[i+2] = temp;     //assign blue to red
		i += 3;     //skip to next blue byte
	}*/
	// TGA format
	/*unsigned char TGAheader[12]={0,0,2,0,0,0,0,0,0,0,0,0};
	unsigned char header[6] = {m_WindowWidth%256,m_WindowWidth/256,
	m_WindowHeight%256,m_WindowHeight/256,24,0};    
	fwrite(TGAheader, sizeof(unsigned char), 12, fScreenshot);
	fwrite(header, sizeof(unsigned char), 6, fScreenshot);
	fwrite(pixels, sizeof(GLubyte), nSize, fScreenshot);
	fclose(fScreenshot);*/
	
	return;
}

int FluidSystem::SetupMfAddVolume ( Vector3DF min, Vector3DF max, float spacing, Vector3DF offs, int cat )
{
	Vector3DF pos;
	int n = 0, p;
	float dx, dy, dz, x, y, z;
	int cntx, cnty, cntz;
	cntx = ceil( (max.x-min.x-offs.x) / spacing );
	cntz = ceil( (max.z-min.z-offs.z) / spacing );
	cnty = ceil((max.y - min.y - offs.y) / spacing);
	//printf("cntx is %d, cntz is %d, cnty is %d, total is %d\n", cntx, cntz, cnty, cntx*cntz*cnty);
	int cnt = cntx * cntz;
	int xp, yp, zp, c2;
	float odd;

	if(cat >= m_Param [ FLUID_CATNUM ])return 0;
	
	dx = max.x-min.x;
	dy = max.y-min.y;
	dz = max.z-min.z;
	
	c2 = cnt/2;
	for (float y = min.y+offs.y; y <= max.y; y += spacing ) 
	{	
		for (int xz=0; xz < cnt; xz++ ) {
			
			x = min.x+offs.x + (xz % int(cntx))*spacing;
			z = min.z+offs.z + (xz / int(cntx))*spacing;
			/*if ( xy < c2 ) {
				zp = xy / int(dx);
				x = min.x+offs + (xz % int(cntx/2) )*spacing*2 + (zp % 2)*spacing;
				z = min.z+offs + zp * spacing;
			} else {
				zp = (xy-c2) / int(dx);
				x = min.x+offs + ( (xz-c2) % int(cntx/2) )*spacing*2 + (zp % 2)*spacing;
				z = min.z+offs + (cntz-1-zp) * spacing;
			}*/
			p = AddParticle ();
			if ( p != -1 ) 
			{
				n++;
				(mPos+p)->Set ( x,y,z);
	//			*(mClr+p) = COLORA( (x-min.x)/dx, (y-min.y)/dy, (z-min.z)/dz, 1); 
				*(mClr+p) = COLORA( 0.25, +0.25 + (y-min.y)*.75/dy, 0.25 + (z-min.z)*.75/dz, 1);  // (x-min.x)/dx
				*(m_alpha+p*MAX_FLUIDNUM+cat) = 1.0f;*(m_alpha_pre+p*MAX_FLUIDNUM+cat) = 1.0f;
				*(m_restMass+p) = m_fluidPMass[cat];
				*(m_restDensity+p) = m_fluidDensity[cat];
				*(m_visc+p) = m_fluidVisc[cat];
				//*(m_alpha + p*MAX_FLUIDNUM + 1) = 1; //*(m_alpha_pre + p*MAX_FLUIDNUM + 1) = 0.5f;
				//*(m_alpha + p*MAX_FLUIDNUM + 1) = 0.5f; // *(m_alpha_pre + p*MAX_FLUIDNUM + 1) = 0.5f;
				//*(m_alpha + p*MAX_FLUIDNUM + 2) = 0.5f; // *(m_alpha_pre + p*MAX_FLUIDNUM + 2) = 0.5f;

				*(MF_type+p) = 0; //which means liquid (project-u)
			}
		}
	}	
	printf("%d fluid has %d particles\n",cat,n);
	return n;
}
int FluidSystem::SetupMfAddBlendVolume(Vector3DF min, Vector3DF max, float spacing, Vector3DF offs)
{
	Vector3DF pos;
	int n = 0, p;
	float dx, dy, dz, x, y, z;
	int cntx, cnty, cntz;
	cntx = ceil((max.x - min.x - offs.x) / spacing);
	cntz = ceil((max.z - min.z - offs.z) / spacing);
	cnty = ceil((max.y - min.y - offs.y) / spacing);
	//printf("cntx is %d, cntz is %d, cnty is %d, total is %d\n", cntx, cntz, cnty, cntx*cntz*cnty);
	int cnt = cntx * cntz;
	int xp, yp, zp, c2;
	float odd;

	dx = max.x - min.x;
	dy = max.y - min.y;
	dz = max.z - min.z;

	c2 = cnt / 2;
	for (float y = min.y + offs.y; y <= max.y; y += spacing)
	{
		for (int xz = 0; xz < cnt; xz++) {
			x = min.x + offs.x + (xz % int(cntx))*spacing;
			z = min.z + offs.z + (xz / int(cntx))*spacing;
			p = AddParticle();
			if (p != -1)
			{
				n++;
				(mPos + p)->Set(x, y, z);
				*(mClr + p) = COLORA(0.25, +0.25 + (y - min.y)*.75 / dy, 0.25 + (z - min.z)*.75 / dz, 1);  // (x-min.x)/dx
				*(m_alpha + p*MAX_FLUIDNUM + 1) = 0.34f; *(m_alpha_pre + p*MAX_FLUIDNUM + 1) = 0.34f;
				*(m_alpha + p*MAX_FLUIDNUM + 2) = 0.33f; *(m_alpha_pre + p*MAX_FLUIDNUM + 2) = 0.33f;
				*(m_alpha + p*MAX_FLUIDNUM + 3) = 0.33f; *(m_alpha_pre + p*MAX_FLUIDNUM + 3) = 0.33f;
				//*(m_alpha + p*MAX_FLUIDNUM + 4) = 0.25f; *(m_alpha_pre + p*MAX_FLUIDNUM + 4) = 0.25f;

				*(m_restMass + p) = 0;
				*(m_restDensity + p) = 0;
				*(m_visc + p) = 0;
				for (int k = 1; k < MAX_FLUIDNUM; ++k)
				{
					*(m_restMass + p) += 1.0 / (MAX_FLUIDNUM-1) * m_fluidPMass[k];
					*(m_restDensity + p) += 1.0 / (MAX_FLUIDNUM-1) * m_fluidDensity[k];
					*(m_visc + p) += 1.0 / (MAX_FLUIDNUM-1) * m_fluidVisc[k];
				}

				*(MF_type + p) = 0; //which means liquid (project-u)
			}
		}
	}
	//printf("%d fluid has %d particles\n", cat, n);
	return n;
}
int FluidSystem::SetupMfAddGridSolid(Vector3DF min, Vector3DF max, float spacing, Vector3DF offs, int type)
{
	Vector3DF pos;
	int n = 0, p;
	float dx, dy, dz, x, y, z;
	int cntx, cnty, cntz;
	cntx = ceil((max.x - min.x - offs.x) / spacing);
	cntz = ceil((max.z - min.z - offs.z) / spacing);
	int holeSize1 = 4, holeSize2 = 4;//10*spacing as size
	int cnt = cntx * cntz;
	int xp, yp, zp, c2;
	float odd;

	dx = max.x - min.x;
	dy = max.y - min.y;
	dz = max.z - min.z;
	float distance1;

	c2 = cnt / 2;

	float xcenter = max.x - dx / 2;
	float ycenter = max.y - dy / 2;
	float zcenter = max.z - dz / 2;
	int xindex, zindex;
	float omega = 0.0;
	float rx, ry;
	//float radius = 81 * spacing*spacing;
	float radius = pow(min(dx, dz) / 6, 2);
	float2 center = make_float2(min.x + dx / 2, min.z + dz / 2);
	center.x += (type - 4)*dx / 4; center.y += (type - 4)*dz / 4;
	for (float y = min.y + offs.y; y <= max.y; y += spacing) {
		for (int xz = 0; xz < cnt; xz++) {
			x = min.x + offs.x + (xz % int(cntx))*spacing;
			z = min.z + offs.z + (xz / int(cntx))*spacing;

			//xindex = ceil((x - min.x - offs.x) / spacing);
			//zindex = ceil((z - min.z - offs.z) / spacing);
			xindex = xz%cntx;
			zindex = xz / cntx;
			//round
			//if (pow(x - center.x, 2) + pow(z - center.y, 2) < radius)
			//	continue;

			if (!((xindex +type) % holeSize1 == 0 || (zindex +type) % holeSize2 == 0))
			{
				distance1 = min(x - min.x, z - min.z);
				distance1 = min(max.x - x, distance1);
				distance1 = min(max.z - z, distance1);
				if (distance1 >  2*spacing)
					continue;
			}
			//if (pow(x - xcenter, 2) + pow(z - zcenter, 2) < radius)
			//	continue;

			p = AddParticle();
			if (p != -1) {
				n++;
				(mPos + p)->Set(x, y, z);

				*(mClr + p) = COLORA(1, 0, 1, 1);  // (x-min.x)/dx
				//*(m_alpha + p*MAX_FLUIDNUM + type) = 1.0f; *(m_alpha_pre + p*MAX_FLUIDNUM + type) = 1.0f;
				*(m_restMass + p) = m_fluidPMass[0];
				*(m_restDensity + p) = m_fluidDensity[0];

				*(m_visc + p) = m_fluidVisc[0];
				*(MF_type + p) = type;
				*(mIsBound + p) = 1;
				rx = x - xcenter;
				ry = y - ycenter;
			}
		}
	}
	printf("%d fluid has %d particles\n", 0, n);
	return n;
}
int FluidSystem::SetupMfAddSolidSolid(Vector3DF min, Vector3DF max, float spacing, Vector3DF offs, int type)
{
	Vector3DF pos;
	int n = 0, p;
	float dx, dy, dz, x, y, z;
	int cntx, cnty, cntz;
	cntx = ceil((max.x - min.x - offs.x) / spacing);
	cntz = ceil((max.z - min.z - offs.z) / spacing);
	int holeSize = 3;//10*spacing as size
	int cnt = cntx * cntz;
	int xp, yp, zp, c2;
	float odd;

	dx = max.x - min.x;
	dy = max.y - min.y;
	dz = max.z - min.z;

	c2 = cnt / 2;

	float xcenter = max.x - dx / 2;
	float ycenter = max.y - dy / 2;
	int xindex, zindex;
	float omega = 0.0;
	float rx, ry;

	float radius2 = pow(min(dx, dz) / 2, 2);
	float2 center = make_float2(min.x + dx / 2, min.z + dz / 2);
	for (float y = min.y + offs.y; y <= max.y; y += spacing) {
		for (int xz = 0; xz < cnt; xz++) {
			x = min.x + offs.x + (xz % int(cntx))*spacing;
			z = min.z + offs.z + (xz / int(cntx))*spacing;

			//xindex = ceil((x - min.x - offs.x) / spacing);
			//zindex = ceil((z - min.z - offs.z) / spacing);
			//xindex = xz%cntx;
			//zindex = xz / cntx;
			//if (!(xindex % holeSize == 0 || zindex % holeSize == 0))
			//	continue;

			p = AddParticle();
			if (p != -1) {
				
				n++;
				(mPos + p)->Set(x, y, z);

				*(mClr + p) = COLORA(1, 0, 1, 1);  // (x-min.x)/dx
												   //*(m_alpha + p*MAX_FLUIDNUM + type) = 1.0f; *(m_alpha_pre + p*MAX_FLUIDNUM + type) = 1.0f;
				*(m_restMass + p) = m_fluidPMass[0];
				*(m_restDensity + p) = m_fluidDensity[0];

				*(m_visc + p) = m_fluidVisc[0];
				*(MF_type + p) = 3;
				*(mIsBound + p) = 1;
				//rx = x - xcenter;
				//ry = y - ycenter;
			}
		}
	}
	printf("Solid solid has %d particles\n", n);
	return n;
}

int FluidSystem::SetupBoundary(Vector3DF min, Vector3DF max, float spacing, Vector3DF offs, int cat)
{
	Vector3DF pos;
	int n = 0, p;
	float dx, dy, dz, x, y, z;
	int cntx, cnty, cntz;
	cntx = ceil((max.x - min.x - offs.x) / spacing);
	cntz = ceil((max.z - min.z - offs.z) / spacing);
	int cnt = cntx * cntz;
	int xp, yp, zp, c2;
	float odd;

	if (cat >= m_Param[FLUID_CATNUM])return n;

	dx = max.x - min.x;
	dy = max.y - min.y;
	dz = max.z - min.z;

	c2 = cnt / 2;
	float distance1,distance2,distance3;
	for (float y = min.y + offs.y; y <= max.y; y += spacing)
	{
		for (int xz = 0; xz < cnt; xz++) {

			x = min.x + offs.x + (xz % int(cntx))*spacing;
			z = min.z + offs.z + (xz / int(cntx))*spacing;
			distance1 = min(x - min.x, z - min.z);
			distance2 = min(max.x-x, y - min.y);

			distance3 = min(max.z - z, distance1);
			distance3 = min(distance3, distance2);
			//distance1 = max.y - y;
			//distance3 = min(distance3, distance1);
			if (distance3 >  1.8*spacing)
				continue;
			/*if ( xy < c2 ) {
			zp = xy / int(dx);
			x = min.x+offs + (xz % int(cntx/2) )*spacing*2 + (zp % 2)*spacing;
			z = min.z+offs + zp * spacing;
			} else {
			zp = (xy-c2) / int(dx);
			x = min.x+offs + ( (xz-c2) % int(cntx/2) )*spacing*2 + (zp % 2)*spacing;
			z = min.z+offs + (cntz-1-zp) * spacing;
			}*/
			p = AddParticle();
			if (p != -1)
			{
				//if (y > 50)
				//	cout << "y is " << y << endl;
				n++;
				(mPos + p)->Set(x, y, z);
				//*(mClr + p) = COLORA(0.25, +0.25 + (y - min.y)*.75 / dy, 0.25 + (z - min.z)*.75 / dz, 1);  // (x-min.x)/dx
				*(mIsBound + p) = 1;
				*(m_restMass + p) = m_Param[PBMASS];
				*(m_restDensity + p) = m_Param[PBRESTDENSITY];
				*(m_visc + p) = m_Param[PBVISC];
				*(MF_type + p) = 1; 
			}
		}
	}
	return n;
}
int FluidSystem::SetupMfAddCylinder(Vector3DF min, Vector3DF max, float spacing, Vector3DF offs, int type)
{
	Vector3DF pos;
	int n = 0, p;
	float dx, dy, dz, x, y, z;
	int cntx, cnty, cntz;
	cntx = ceil((max.x - min.x - offs.x) / spacing);
	cntz = ceil((max.z - min.z - offs.z) / spacing);
	int cnt = cntx * cntz;
	int xp, yp, zp, c2;
	float odd;

	dx = max.x - min.x;
	dy = max.y - min.y;
	dz = max.z - min.z;

	c2 = cnt / 2;

	float xcenter = max.x - dx / 2;
	float ycenter = max.y - dy / 2;
	float omega = 0.0;
	float rx, ry;

	float radius2 = pow(min(dx, dz) / 2,2);
	float2 center = make_float2(min.x + dx / 2, min.z + dz / 2);
	for (float y = min.y + offs.y; y <= max.y; y += spacing) {
		for (int xz = 0; xz < cnt; xz++) {
			x = min.x + offs.x + (xz % int(cntx))*spacing;
			z = min.z + offs.z + (xz / int(cntx))*spacing;
			if (pow(x - center.x, 2) + pow(z - center.y, 2) > radius2)
				continue;

			p = AddParticle();
			if (p != -1) {
				*(elasticID + p) = n;
				n++;
				(mPos + p)->Set(x, y, z);

				*(mClr + p) = COLORA(1,0,1,1);  // (x-min.x)/dx
				*(m_alpha + p*MAX_FLUIDNUM + type) = 1.0f; *(m_alpha_pre + p*MAX_FLUIDNUM + type) = 1.0f;
				*(m_restMass + p) = m_fluidPMass[0];
				*(m_restDensity + p) = m_fluidDensity[0];

				*(m_visc + p) = m_fluidVisc[0];
				*(MF_type + p) = 1; //1 means deformable

				rx = x - xcenter;
				ry = y - ycenter;
				//mVel[p].Set( ry*omega, -rx*omega, 0);
				//mVelEval[p].Set( ry*omega, -rx*omega,0);

				//mVel[p].Set( -0.4, 0, 0);
				//mVelEval[p].Set( -0.4, 0,0);

				/*
				if(mPos[p].y>15){
				(mVel + p)->Set ( 0,-0.4,0.4 );
				(mVelEval + p)->Set ( 0,-0.4,0.4 );
				}
				else{
				(mVel + p)->	Set ( 0,-0.4,0 );
				(mVelEval + p)->Set ( 0,-0.4,0 );
				}*/
			}
		}
	}
	printf("%d fluid has %d particles\n", 0, n);
	return n;
}

void FluidSystem::liftup(int mode){
	floatup_cuda(mode);
	printf("Frame: %d liftmode: %d\n", m_Frame, mode);
}
void FluidSystem::saveParticle(std::string name)
{
	TransferFromCUDAForLoad();
	FILE* fp;
	int n = 0;
	for (int i = 0;i<NumPoints();i++)
		if (mIsBound[i] == 0 && mPos[i].x>-500)
			n++;
	fp = fopen(name.c_str(),"w");
	fprintf(fp,"%d\n",n);
	Vector3DF* ppos = mPos;
	for (int i = 0;i<NumPoints();i++,ppos++)
	if (mIsBound[i] == 0 && mPos[i].x>-500)
	{
		fprintf(fp,"%f %f %f\n",ppos->x,ppos->y,ppos->z);
		//for (int j = 0;j<MAX_FLUIDNUM;j++)
		//{
		//	fprintf(fp," %f",*(m_alpha+i*MAX_FLUIDNUM + j));
		//	fprintf(fp," %f",*(m_alpha_pre+i*MAX_FLUIDNUM + j));
		//}
		//fprintf(fp," %f",m_restMass[i]);
		//fprintf(fp," %f",m_restDensity[i]);
		//fprintf(fp," %f",m_visc[i]);
		//fprintf(fp," %f %f %f",mVel[i].x, mVel[i].y, mVel[i].z);
		//fprintf(fp," %f %f %f",mVelEval[i].x, mVelEval[i].y, mVelEval[i].z);
		//fprintf(fp," %d",MF_type[i]);

	}
	fclose(fp);
	SetYan(SAVE_STAT,0);
}
int FluidSystem::loadParticle(std::string name)
{
	int n,p;
	float f1,f2,f3;
	FILE* fp;
	fp = fopen(name.c_str(),"r");
	fscanf(fp,"%d",&n);
	Vector3DF* ppos = mPos;
	for (int i = 0;i<n;i++)
	{
		p = AddParticle(); // the index of the added particle
		if (p!=-1)
		{
			fscanf(fp,"%f %f %f",&f1,&f2,&f3);
			(mPos +p)->Set(f1,f2,f3);
			
			*(m_alpha + p*MAX_FLUIDNUM + 3) = 1.0f; *(m_alpha_pre + p*MAX_FLUIDNUM + 3) = 1.0f;
			*(m_restMass + p) = m_fluidPMass[3];
			*(m_restDensity + p) = m_fluidDensity[3];
			*(m_visc + p) = m_fluidVisc[3];
			*(MF_type+p) = 0;
			//for (int j = 0;j<MAX_FLUIDNUM;j++)
			//{
			//	fscanf(fp,"%f",(m_alpha+p*MAX_FLUIDNUM + j));
			//	fscanf(fp,"%f",(m_alpha_pre+p*MAX_FLUIDNUM + j));
			//}
			//fscanf(fp,"%f",m_restMass+p);
			//fscanf(fp,"%f",m_restDensity+p);
			//fscanf(fp,"%f",m_visc+p);
			//fscanf(fp," %f %f %f",&mVel[i].x, &mVel[i].y, &mVel[i].z);
			//fscanf(fp," %f %f %f",&mVelEval[i].x, &mVelEval[i].y, &mVelEval[i].z);
			//mVel[i].x = mVel[i].y = mVel[i].z = 0;
			//mVelEval[i].x = mVelEval[i].y = mVelEval[i].z = 0;

			//fscanf(fp,"%d",&MF_type[i]);
			
	
			//*(mClr+p) = COLORA( 1,1,1,1);
		}
	}
	fclose(fp);
	//saveParticle("fluids_exa2.dat");
	return n;
}
int FluidSystem::SetupAddMonster(BI2Reader bi2reader, int type, int cat)
{
	//printf("%f %f %f %f\n", m_Param[PBMASS], m_Param[PBRESTDENSITY], m_Param[PBVISC], m_Param[PBSTIFF]);
	float x, y, z, n = 0;
	for (int i = 0; i<bi2reader.info.nbound; i++)
	{
		x = bi2reader.info.Pos[i].x / m_Param[PSIMSCALE];
		y = bi2reader.info.Pos[i].y / m_Param[PSIMSCALE];
		z = bi2reader.info.Pos[i].z / m_Param[PSIMSCALE];
		if (x < m_Vec[PVOLMIN].x || x>m_Vec[PVOLMAX].x || y<m_Vec[PVOLMIN].y || y> m_Vec[PVOLMAX].y || z<m_Vec[PVOLMIN].z || z>m_Vec[PVOLMAX].z)
			continue;

		int p = AddParticle();
		if (p != -1)
		{
			(mPos + p)->Set(bi2reader.info.Pos[i].x / m_Param[PSIMSCALE]-25, bi2reader.info.Pos[i].y / m_Param[PSIMSCALE]-25, bi2reader.info.Pos[i].z / m_Param[PSIMSCALE]+25);
			*(mClr + p) = COLORA(1, 1, 1, 0);
			*(mIsBound + p) = false;
			*(m_restMass + p) = m_Param[PBMASS];
			*(m_restDensity + p) = m_fluidDensity[cat];
			*(m_visc + p) = m_fluidVisc[cat];
			*(MF_type + p) = type;//which means rigid (project-u)
			*(m_alpha + p*MAX_FLUIDNUM + cat) = 1.0f; *(m_alpha_pre + p*MAX_FLUIDNUM + cat) = 1.0f;
			n++;
		}
	}
	return n;
}
#ifdef NEW_BOUND
void FluidSystem::SetupAddBound(BI2Reader bi2reader,int boundtype)
{
	printf("%f %f %f %f\n",m_Param[PBMASS],m_Param[PBRESTDENSITY],m_Param[PBVISC],m_Param[PBSTIFF]);
	float x,y,z;
	for (int i = 0;i<bi2reader.info.nbound;i++)
	{
		x = bi2reader.info.Pos[i].x/m_Param[PSIMSCALE];
		y = bi2reader.info.Pos[i].y/m_Param[PSIMSCALE];
		z = bi2reader.info.Pos[i].z/m_Param[PSIMSCALE];
		if( x < m_Vec[PVOLMIN].x || x>m_Vec[PVOLMAX].x || y<m_Vec[PVOLMIN].y || y> m_Vec[PVOLMAX].y || z<m_Vec[PVOLMIN].z || z>m_Vec[PVOLMAX].z)
			continue;

		int p = AddParticle();
		if (p!=-1)
		{
			(mPos+p)->Set (bi2reader.info.Pos[i].x/m_Param[PSIMSCALE],bi2reader.info.Pos[i].y/m_Param[PSIMSCALE],bi2reader.info.Pos[i].z/m_Param[PSIMSCALE]);
			*(mClr+p) = COLORA(1,1,1,0); 
			*(mIsBound+p) = boundtype;
			*(m_restMass+p) = m_Param[PBMASS];
			*(m_restDensity+p) = m_Param[PBRESTDENSITY];
			*(m_visc+p) = m_Param[PBVISC];
			*(MF_type+p) = 2;//which means rigid (project-u)
		}
	}
}
#endif

//void FluidSystem::MfTestSetupExample ()
//{
//	example= _example;
//	m_Param[PEXAMPLE] = example;
//	printf("here we have example %d\n",example);
//	ParseXML_Bound("BoundInfo",example);
//
//	//load boundary and special models
//
//	BI2Reader* bi2readers[10]; //ten pointers to build bi2reader dynamically, in use now
//	char biname[200];
//	switch(example){
//#ifdef NEW_BOUND
//	case 1:
//		sprintf(biname,".\\extra_particles\\Boundary1.bi2");
//		bi2readers[0] = new BI2Reader(biname);
//		bi2readers[0]->GetInfo(false);
//		bi2readers[0]->PrintInfo();
//		break;
//	case 12:
//		sprintf(biname,".\\extra_particles\\Boundary12.bi2");
//		bi2readers[0] = new BI2Reader(biname);
//		bi2readers[0]->GetInfo(false);
//		bi2readers[0]->PrintInfo();
//		break;
//#endif
//	case 2:
//		sprintf(biname,".\\extra_particles\\monster2.bi2");
//		bi2readers[0] = new BI2Reader(biname);
//		bi2readers[0]->GetInfo(false);
//		bi2readers[0]->PrintInfo();
//		break;
//	}
//	double particleVisc =  m_Param[PVISC];
//
//	//parse the xml and adjust some parameters according to scaleP
//	ParseMFXML ( "MultiScene", example, true );
//	
//
//	//adjust the parametres according to the scale parameter
//	scaleP3 = pow(scaleP,1.0/3.0);
//	m_Param[PMASS]/=scaleP;
//#ifdef NEW_BOUND
//	m_Param[PBMASS]/=scaleP;
//#endif
//	m_Param[PSMOOTHRADIUS]/=scaleP3;
//	m_Param[PRADIUS]/=scaleP3;
//	m_Param [PNUM]*=scaleP;
//
//	//Add the number of boundary or monster to PNUM
//	switch(example){
//#ifdef NEW_BOUND
//	case 1:
//		m_Param[PNUM] += bi2readers[0]->info.nbound; //boundary particles
//		break;
//	case 12:
//		m_Param[PNUM] += bi2readers[0]->info.nbound; //boundary particles
//		break;
//	case 2:
//		m_Param[PNUM] += bi2readers[0]->info.nbound;//monster
//		break;
//	}
//#endif
//	m_Param [PGRIDSIZE] = 2*m_Param[PSMOOTHRADIUS] / m_Param[PGRID_DENSITY];
//	
//	m_fluidPMass[0] = m_Param[PMASS]*massRatio.x;
//	m_fluidPMass[1] = m_Param[PMASS]*massRatio.y;
//	m_fluidPMass[2] = m_Param[PMASS]*massRatio.z;
//	m_fluidDensity[0] = 600.0*densityRatio.x;
//	m_fluidDensity[1] = 600.0*densityRatio.y;
//	m_fluidDensity[2] = 600.0*densityRatio.z;
//	m_fluidVisc[0] =  particleVisc*viscRatio.x;
//	m_fluidVisc[1] =  particleVisc*viscRatio.y;
//	m_fluidVisc[2] =  particleVisc*viscRatio.z;
//	if(m_Param[FLUID_CATNUM]>3){
//		m_fluidPMass[3] = m_Param[PMASS]*massRatio.w;
//		m_fluidDensity[3] = 600.0*densityRatio.w;
//		m_fluidVisc[3] =  particleVisc*viscRatio.w;
//	}
//	
//	//Allocate buffer and setup the kernels and spacing
//	AllocateParticles ( m_Param[PNUM] );
//	AllocatePackBuf ();
//	SetupKernels ();
//	SetupSpacing ();
//
//	//Add fluid particles
//	if (loadwhich == 0)
//	{
//		switch(example){
//		case 1:
//			m_Vec [ PINITMIN ].Set (volumes[0].x,volumes[0].y,volumes[0].z);
//			m_Vec [ PINITMAX ].Set (volumes[1].x,volumes[1].y,volumes[1].z);
//			SetupMfAddVolume ( m_Vec[PINITMIN], m_Vec[PINITMAX], m_Param[PSPACING], Vector3DF(0,0.1,0),0);
//
//			m_Vec [ PINITMIN ].Set (volumes[2].x,volumes[2].y,volumes[2].z);
//			m_Vec [ PINITMAX ].Set (volumes[3].x,volumes[3].y,volumes[3].z);
//			SetupMfAddVolume ( m_Vec[PINITMIN], m_Vec[PINITMAX], m_Param[PSPACING], Vector3DF(0,0.1,0),1);
//
//			m_Vec [ PINITMIN ].Set (volumes[4].x,volumes[4].y,volumes[4].z);
//			m_Vec [ PINITMAX ].Set (volumes[5].x,volumes[5].y,volumes[5].z);
//			SetupMfAddVolume ( m_Vec[PINITMIN], m_Vec[PINITMAX], m_Param[PSPACING], Vector3DF(0,0.1,0),2);
//			break;
//		case 11:
//			m_Vec [ PINITMIN ].Set (volumes[0].x,volumes[0].y,volumes[0].z);
//			m_Vec [ PINITMAX ].Set (volumes[1].x,volumes[1].y,volumes[1].z);
//			SetupMfAddVolume ( m_Vec[PINITMIN], m_Vec[PINITMAX], m_Param[PSPACING], Vector3DF(0,0.1,0),0);
//
//			m_Vec [ PINITMIN ].Set (volumes[2].x,volumes[2].y,volumes[2].z);
//			m_Vec [ PINITMAX ].Set (volumes[3].x,volumes[3].y,volumes[3].z);
//			SetupMfAddVolume ( m_Vec[PINITMIN], m_Vec[PINITMAX], m_Param[PSPACING], Vector3DF(0,0.1,0),1);
//
//			m_Vec [ PINITMIN ].Set (volumes[4].x,volumes[4].y,volumes[4].z);
//			m_Vec [ PINITMAX ].Set (volumes[5].x,volumes[5].y,volumes[5].z);
//			SetupMfAddVolume ( m_Vec[PINITMIN], m_Vec[PINITMAX], m_Param[PSPACING], Vector3DF(0,0.1,0),2);
//			break;
//		case 12:
//			m_Vec [ PINITMIN ].Set (volumes[0].x,volumes[0].y,volumes[0].z);
//			m_Vec [ PINITMAX ].Set (volumes[1].x,volumes[1].y,volumes[1].z);
//			SetupMfAddVolume ( m_Vec[PINITMIN], m_Vec[PINITMAX], m_Param[PSPACING], Vector3DF(0,0.1,0),0);
//
//			m_Vec [ PINITMIN ].Set (volumes[2].x,volumes[2].y,volumes[2].z);
//			m_Vec [ PINITMAX ].Set (volumes[3].x,volumes[3].y,volumes[3].z);
//			SetupMfAddVolume ( m_Vec[PINITMIN], m_Vec[PINITMAX], m_Param[PSPACING], Vector3DF(0,0.1,0),1);
//
//			m_Vec [ PINITMIN ].Set (volumes[4].x,volumes[4].y,volumes[4].z);
//			m_Vec [ PINITMAX ].Set (volumes[5].x,volumes[5].y,volumes[5].z);
//			SetupMfAddVolume ( m_Vec[PINITMIN], m_Vec[PINITMAX], m_Param[PSPACING], Vector3DF(0,0.1,0),2);
//			break;
//		}
//	}
//	else{ //load from file
//		switch(example){
//		case 2:
//			mMaxPoints = loadParticle(".\\save\\save_stat2.txt");
//			break;
//		case 3:
//			mMaxPoints  = loadParticle(".\\save\\save_stat3.txt");
//			break;
//		case 5:
//			mMaxPoints  = loadParticle(".\\save\\save_stat5.txt");
//			//mMaxPoints += 32767*scaleP;
//			break;
//		}
//	}
//
//	//Add particles of boundary or extra models
//
//	switch(example){
//#ifdef NEW_BOUND
//	case 1:
//		//mMaxPoints += bi2readers[0]->info.nbound;
//		//SetupAddBound(*bi2readers[0],1);
//		break;
//	case 12:
//		//mMaxPoints += bi2readers[0]->info.nbound;
//		SetupAddBound(*bi2readers[0],1);
//		break;
//#endif
//	case 2:
//		//mMaxPoints += bi2readers[0]->info.np;
//		SetupAddShape(*bi2readers[0],1);
//		break;
//	}
//	m_maxAllowedPoints = mNumPoints * EMIT_BUF_RATIO;
//
//	//set emit parametres
//	if(example==5){
//		m_maxAllowedPoints += 32767*scaleP;
//		m_Vec[PEMIT_RATE] = Vector3DF(5,256,25);
//		m_Vec[PEMIT_SPREAD] = Vector3DF(0,0,0);
//		m_Vec[PEMIT_ANG] = Vector3DF(0,135,2);
//		m_Vec[PEMIT_POS] = Vector3DF(0,60,0);
//	}
//	else{
//		//Emit Params
//		m_Vec[PEMIT_RATE] = Vector3DF(1.1,9,0);
//		m_Vec[PEMIT_SPREAD] = Vector3DF(0,0,0);
//		m_Vec[PEMIT_ANG] = Vector3DF(0,180,1);
//		m_Vec[PEMIT_POS] = Vector3DF(0,60,30);
//	}
//}

void FluidSystem::SetupAddShape(BI2Reader bi2reader,int cat)
{
	printf("%f %f %f %f\n",m_Param[PBMASS],m_Param[PBRESTDENSITY],m_Param[PBVISC],m_Param[PBSTIFF]);
	for (int i = 0;i<bi2reader.info.np;i++)
	{
		int p = AddParticle();
		if (p!=-1)
		{
		//	printf("%f %f %f\n",bi2reader.info.Pos[i].x/m_Param[PSIMSCALE],bi2reader.info.Pos[i].y/m_Param[PSIMSCALE],bi2reader.info.Pos[i].z/m_Param[PSIMSCALE]);
			(mPos+p)->Set (bi2reader.info.Pos[i].x/m_Param[PSIMSCALE],bi2reader.info.Pos[i].y/m_Param[PSIMSCALE],bi2reader.info.Pos[i].z/m_Param[PSIMSCALE]);
			*(mClr+p) = COLORA(1,1,1,1); 
			*(m_alpha+p*MAX_FLUIDNUM+cat) = 1.0f;*(m_alpha_pre+p*MAX_FLUIDNUM+cat) = 1.0f;
			*(m_restMass+p) = m_fluidPMass[cat];
			*(m_restDensity+p) = m_fluidDensity[cat];
			*(m_visc+p) = m_fluidVisc[cat];
			*(mIsBound+p) = 2;
			*(MF_type+p) = 1; //which means deformable (project-u)
		}
	}
}

void FluidSystem::EmitMfParticles (int cat)
{
	mMaxPoints = mNumPoints;
	int currentPoints = mMaxPoints;

	if ( m_Vec[PEMIT_RATE].x > 0 && (m_Frame) % (int) m_Vec[PEMIT_RATE].x == 0 ) {
		float ss = m_Param [ PDIST ] / m_Param[ PSIMSCALE ];		// simulation scale (not Schutzstaffel)
		AddMfEmit ( ss, cat );
	}

	EmitUpdateCUDA(currentPoints,mMaxPoints);
}

void FluidSystem::AddMfEmit ( float spacing, int cat )
{
	int p;
	Vector3DF dir;
	Vector3DF pos;
	float ang_rand, tilt_rand;
	//float rnd = m_Vec[PEMIT_RATE].y * 0.15;	
	int x = (int) sqrt(m_Vec[PEMIT_RATE].y);
	int offset = m_Vec[PEMIT_RATE].y/x;

	for ( int n = 0; n < m_Vec[PEMIT_RATE].y; n++ ) {
		float zOffset = -spacing * (float)(offset) * 0.5 + spacing * (n/x);
		float xOffset = -spacing * (float)(offset) * 0.5 + spacing * (n%x);
		if(zOffset*zOffset+xOffset*xOffset>m_Vec[PEMIT_RATE].z)
			continue;

		ang_rand = (float(rand()*2.0/RAND_MAX) - 1.0) * m_Vec[PEMIT_SPREAD].x;
		tilt_rand = (float(rand()*2.0/RAND_MAX) - 1.0) * m_Vec[PEMIT_SPREAD].y;
		dir.z = cos ( ( m_Vec[PEMIT_ANG].x + ang_rand) * DEGtoRAD ) * sin( ( m_Vec[PEMIT_ANG].y + tilt_rand) * DEGtoRAD ) * m_Vec[PEMIT_ANG].z;
		dir.x = sin ( ( m_Vec[PEMIT_ANG].x + ang_rand) * DEGtoRAD ) * sin( ( m_Vec[PEMIT_ANG].y + tilt_rand) * DEGtoRAD ) * m_Vec[PEMIT_ANG].z;
		dir.y = cos ( ( m_Vec[PEMIT_ANG].y + tilt_rand) * DEGtoRAD ) * m_Vec[PEMIT_ANG].z;

		//original position
		//pos = m_Vec[PEMIT_POS];
		//pos.z += spacing * (n/x);
		//pos.x += spacing * (n%x);
		pos = Vector3DF(xOffset,0,zOffset);

		////rotate \theta around a=(-sin \phi, cos \phi, 0), axis order z->x->y
		Vector3DF rotateAxis = Vector3DF(cos(m_Vec[PEMIT_ANG].x*DEGtoRAD),0.0,-sin(m_Vec[PEMIT_ANG].x*DEGtoRAD));
		Vector3DF finalpos;
		float A1[3][3],A2[3][3],M[3][3];
		A1[0][0]=rotateAxis.x*rotateAxis.x;A1[0][1]=A1[1][0]=rotateAxis.x*rotateAxis.y;A1[0][2]=A1[2][0]=rotateAxis.x*rotateAxis.z;
		A1[1][1]=rotateAxis.y*rotateAxis.y;A1[1][2]=A1[2][1]=rotateAxis.y*rotateAxis.z;A1[2][2]=rotateAxis.z*rotateAxis.z;
		A2[0][0]=A2[1][1]=A2[2][2]=0.0;A2[0][1]=rotateAxis.z;A2[1][0]=-rotateAxis.z;
		A2[0][2]=-rotateAxis.y;A2[2][0]=rotateAxis.y;A2[1][2]=rotateAxis.x;A2[2][1]=-rotateAxis.x;
		float costheta=cos(m_Vec[PEMIT_ANG].y*DEGtoRAD);
		float sintheta=sin(m_Vec[PEMIT_ANG].y*DEGtoRAD);
		M[0][0]=A1[0][0]+(1-A1[0][0])*costheta+A2[0][0]*sintheta;
		M[0][1]=A1[0][1]+(0-A1[0][1])*costheta+A2[0][1]*sintheta;
		M[0][2]=A1[0][2]+(0-A1[0][2])*costheta+A2[0][2]*sintheta;
		M[1][0]=A1[1][0]+(0-A1[1][0])*costheta+A2[1][0]*sintheta;
		M[1][1]=A1[1][1]+(1-A1[1][1])*costheta+A2[1][1]*sintheta;
		M[1][2]=A1[1][2]+(0-A1[1][2])*costheta+A2[1][2]*sintheta;
		M[2][0]=A1[2][0]+(0-A1[2][0])*costheta+A2[2][0]*sintheta;
		M[2][1]=A1[2][1]+(0-A1[2][1])*costheta+A2[2][1]*sintheta;
		M[2][2]=A1[2][2]+(1-A1[2][2])*costheta+A2[2][2]*sintheta;
		finalpos.x=pos.x*M[0][0]+pos.y*M[1][0]+pos.z*M[2][0];
		finalpos.y=pos.x*M[0][1]+pos.y*M[1][1]+pos.z*M[2][1];
		finalpos.z=pos.x*M[0][2]+pos.y*M[1][2]+pos.z*M[2][2];

		pos=finalpos+m_Vec[PEMIT_POS];
		////
		if(mMaxPoints<m_maxAllowedPoints)
		{
			mMaxPoints++;
			p = AddParticle ();
			*(mPos+p) = pos;
			*(mVel+p) = dir;
			*(mVelEval+p) = dir;
			*(mAge+p) = 0;
			*(mClr+p) = COLORA ( m_Time/10.0, m_Time/5.0, m_Time /4.0, 1 );

			*(m_alpha+p*MAX_FLUIDNUM+cat) = 1.0f;*(m_alpha_pre+p*MAX_FLUIDNUM+cat) = 1.0f;
			*(m_restMass+p) = m_fluidPMass[cat];
			*(m_restDensity+p) = m_fluidDensity[cat];
			*(m_visc+p) = m_fluidVisc[cat];
			*(mIsBound+p) = 0;
			*(MF_type+p) = 0; //which means liquid
		}
	}
}

void FluidSystem::EmitUpdateCUDA (int startnum, int endnum)
{ 
	int numpoints = endnum - startnum;
#ifdef NEW_BOUND
	CopyEmitToCUDA ( (float*) mPos, (float*) mVel, (float*) mVelEval, (float*) mForce, mPressure, mDensity, mClusterCell, mGridNext, (char*) mClr, startnum, numpoints , mIsBound); 
#else
	CopyEmitToCUDA ( (float*) mPos, (float*) mVel, (float*) mVelEval, (float*) mForce, mPressure, mDensity, mClusterCell, mGridNext, (char*) mClr, startnum, numpoints ); 
#endif

	CopyEmitMfToCUDA ( m_alpha, m_alpha_pre, m_pressure_modify, (float*) m_vel_phrel, m_restMass, m_restDensity, m_visc, (float*)m_velxcor, (float*)m_alphagrad, startnum, numpoints);

	UpdatePNumCUDA(endnum);
	cudaThreadSynchronize ();
}
int findNearestV(PIC* bunny, float x, float y, float z)
{
	float minD = 100000000;
	float distance;
	int index = -1;
	for (int i = 0; i < bunny->V.size(); ++i)
	{
		distance = pow(x - bunny->V[i].X, 2) + pow(y - bunny->V[i].Y, 2) + pow(z - bunny->V[i].Z, 2);
		if (distance <minD)
		{
			index = i;
			minD = distance;
		}
	}
	return index;
}
int FluidSystem::SetupModel(PIC* bunny, float spacing, int type, Vector3DF displacement)
{
	Vector3DF pos;
	int n = 0, p;
	float x, y, z;
	int cntx, cnty, cntz;
	int index;
	cntx = ceil((bunny->maxPos.X - bunny->minPos.X) / spacing);
	cntz = ceil((bunny->maxPos.Z - bunny->minPos.Z) / spacing);
	int cnt = cntx * cntz;
	int xp, yp, zp, c2;
	float odd;

	c2 = cnt / 2;
	for (float y = bunny->minPos.Y; y <= bunny->maxPos.Y; y += spacing)
	{
		for (int xz = 0; xz < cnt; xz++) {

			x = bunny->minPos.X + (xz % int(cntx))*spacing;
			z = bunny->minPos.Z + (xz / int(cntx))*spacing;
			index = findNearestV(bunny, x, y, z);
			/*printf("pos is (%f,%f,%f), nearest point is (%f,%f,%f), index is %d\n",
			x, y, z, bunny->V[index].X, bunny->V[index].Y, bunny->V[index].Z, index);*/
			if (bunny->VN[index].NX*(bunny->V[index].X - x) + bunny->VN[index].NY*(bunny->V[index].Y - y) +
				bunny->VN[index].NZ*(bunny->V[index].Z - z) < 0)
				continue;
			p = AddParticle();
			if (p != -1)
			{
				//dist = sqrt(pow(zcenter - z, 2) + pow(xcenter - x, 2) + pow(ycenter - y, 2));
				*(elasticID + p) = n;
				(mPos + p)->Set(x+displacement.x, y+ displacement.y, z+ displacement.z);

				*(m_alpha + p*MAX_FLUIDNUM + type) = 1.0f; *(m_alpha_pre + p*MAX_FLUIDNUM + type) = 1.0f;
				*(m_restMass + p) = m_fluidPMass[0];
				*(m_restDensity + p) = m_fluidDensity[0];

				*(m_visc + p) = m_fluidVisc[0];
				*(mIsBound + p) = false;
				*(MF_type + p) = 1; //1 means deformable

				*(porosity_particle + n) = porosity;

				n++;
			}
			
		}
	}
	return n;
}

int FluidSystem::SetupMfAddDeformVolume ( Vector3DF min, Vector3DF max, float spacing, Vector3DF offs, int type )
{
	Vector3DF pos;
	int n = 0, p;
	float dx, dy, dz, x, y, z;
	int cntx, cnty, cntz;
	cntx = ceil( (max.x-min.x-offs.x) / spacing );
	cntz = ceil( (max.z-min.z-offs.z) / spacing );
	int cnt = cntx * cntz;
	int xp, yp, zp, c2;
	float odd;

	dx = max.x-min.x;
	dy = max.y-min.y;
	dz = max.z-min.z;
	
	c2 = cnt/2;
	float radius = min(dz/2, min(dx/2, dy/2));
	radius = radius * radius;
	float xcenter = max.x - dx/2;
	float ycenter = max.y - dy/2;
	float zcenter = max.z - dz / 2;
	float omega = 0.0;
	float rx,ry;
	float d[6];
	Vector3DF dist;
	for (float y = min.y+offs.y; y <= max.y; y += spacing ) {	
		for (int xz=0; xz < cnt; xz++ ) {
			x = min.x+offs.x + (xz % int(cntx))*spacing;
			z = min.z+offs.z + (xz / int(cntx))*spacing;
			if (pow(x - xcenter, 2) + pow(y - ycenter, 2) + pow(z - zcenter, 2) > radius)
				continue;
			p = AddParticle ();
			if ( p != -1 ) {
				//dist = sqrt(pow(zcenter - z, 2) + pow(xcenter - x, 2) + pow(ycenter - y, 2));
				*(elasticID + p) = n;
				
				d[0] = x - min.x; d[1] = y - min.y; d[2] = z - min.z;
				d[3] = max.x - x; d[4] = max.y - y; d[5] = max.z - z;
				if (d[0] < d[3])dist.x = -d[0]; else dist.x = d[3];
				if (d[1] < d[4])dist.y = -d[1]; else dist.y = d[4];
				if (d[2] < d[5])dist.z = -d[2]; else dist.z = d[5];
				//假设现在只有一个固体
				(signDistance + n)->Set(dist.x,dist.y,dist.z);
				(mPos+p)->Set ( x,y,z);
	
				*(mClr+p) = COLORA( 0.25, +0.25 + (y-min.y)*.75/dy, 0.25 + (z-min.z)*.75/dz, 1);  // (x-min.x)/dx
				*(m_alpha+p*MAX_FLUIDNUM+type) = 1.0f;*(m_alpha_pre+p*MAX_FLUIDNUM+type) = 1.0f;
				*(m_restMass+p) = m_fluidPMass[0];
				*(m_restDensity+p) = m_fluidDensity[0];			

				*(m_visc+p) = m_fluidVisc[0];
				*(mIsBound + p) = false;
				//	*(MF_type + p) = 3;//3 means ghost elastic particles
				//else
				*(MF_type+p) = 2; //1 means deformable
				//omega = porosity * radius / (0.01*radius + dist);
				//if (omega > 0.95)
				//	omega = 0.95;
				*(porosity_particle + n) = porosity;
				/**(misGhost + n) = 0;
				if (x == min.x + offs.x || y == min.y + offs.y || z == min.z + offs.z)
					*(misGhost + n) = 1;
				if (x + spacing >= max.x || y + spacing >= max.y || z + spacing >= max.z)
					*(misGhost + n) = 1;*/
				rx = x - xcenter;
				ry = y - ycenter;
				n++;
			}
		}
	}	
	printf("%d elastic solid has %d particles\n",0,n);
	return n;
}
int FluidSystem::SetupMfAddSphere(Vector3DF min, Vector3DF max, float spacing, Vector3DF offs, int type)
{
	Vector3DF pos;
	int n = 0, p;
	int id = numElasticPoints;
	float dx, dy, dz, x, y, z;
	int cntx, cnty, cntz;
	cntx = ceil((max.x - min.x - offs.x) / spacing);
	cntz = ceil((max.z - min.z - offs.z) / spacing);
	int cnt = cntx * cntz;
	int xp, yp, zp, c2;
	float odd;

	dx = max.x - min.x;
	dy = max.y - min.y;
	dz = max.z - min.z;

	c2 = cnt / 2;

	float xcenter = max.x - dx / 2;
	float ycenter = max.y - dy / 2;
	float zcenter = max.z - dz / 2;
	float omega = 0.0;
	float rx, ry;
	float radius = min(dx, dz);
	radius = min(radius, dy);
	radius /= 2;
	float radius2 = pow(radius, 2);
	float3 center = make_float3(min.x + dx / 2, min.y + dy / 2, min.z + dz / 2);
	for (float y = min.y + offs.y; y <= max.y; y += spacing) {
		for (int xz = 0; xz < cnt; xz++) {
			x = min.x + offs.x + (xz % int(cntx))*spacing;
			z = min.z + offs.z + (xz / int(cntx))*spacing;
			if (pow(x - center.x, 2) + pow(z - center.z, 2) + pow(y - center.y, 2) > radius2)
				continue;

			p = AddParticle();
			if (p != -1) {
				*(elasticID + p) = id++;
				n++;
				(mPos + p)->Set(x, y, z);

				*(mClr + p) = COLORA(1, 0, 1, 1);  // (x-min.x)/dx
				*(m_alpha + p*MAX_FLUIDNUM + 0) = 1.0f; *(m_alpha_pre + p*MAX_FLUIDNUM + 0) = 1.0f;
				*(m_restMass + p) = m_fluidPMass[0];
				*(m_restDensity + p) = m_fluidDensity[0];

				*(m_visc + p) = m_fluidVisc[0];
				*(MF_type + p) = type; //1 means deformable
				*(mIsBound + p) = 0;
				rx = x - xcenter;
				ry = y - ycenter;
				//mVel[p].Set( ry*omega, -rx*omega, 0);
				//mVelEval[p].Set( ry*omega, -rx*omega,0);

				//mVel[p].Set( -0.4, 0, 0);
				//mVelEval[p].Set( -0.4, 0,0);

				/*
				if(mPos[p].y>15){
				(mVel + p)->Set ( 0,-0.4,0.4 );
				(mVelEval + p)->Set ( 0,-0.4,0.4 );
				}
				else{
				(mVel + p)->	Set ( 0,-0.4,0 );
				(mVelEval + p)->Set ( 0,-0.4,0 );
				}*/
			}
		}
	}
	printf("%d fluid has %d particles\n", 0, n);
	return n;
}
int FluidSystem::SetupMfAddMagicWand(Vector3DF min, Vector3DF max, float spacing, Vector3DF offs)
{
	Vector3DF pos;
	int n = 0, p;
	float dx, dy, dz, x, y, z;
	int cntx, cnty, cntz;
	cntx = ceil((max.x - min.x - offs.x) / spacing);
	cntz = ceil((max.z - min.z - offs.z) / spacing);
	int cnt = cntx * cntz;
	int xp, yp, zp, c2;
	float odd;

	dx = max.x - min.x;
	dy = max.y - min.y;
	dz = max.z - min.z;

	c2 = cnt / 2;

	float xcenter = min.x + dx / 4;
	float ycenter = min.y + dy / 4;
	float zcenter = min.z + dz / 4;

	float zcenter2 = max.z - dz / 2;
	float ycenter2 = max.y - dy / 2;

	
	float omega = 0.0;
	float rx, ry;
	float radius = min(dy / 2, dz / 2);

	
	//radius of cylinder
	float rc;
	//radius = min(radius, dy/2);
	//radius /= 2;
	
	rc = radius / 3;
	float radius2 = pow(radius, 2), rc2 = pow(rc, 2);
	float3 center = make_float3(min.x + radius, min.y + dy / 2, min.z + dz / 2);
	float3 center3 = make_float3(max.x - radius, max.y - dy / 2, max.z - dz / 2);
	int isSphere = true;
	for (float y = min.y + offs.y; y <= max.y; y += spacing)
	{
		for (int xz = 0; xz < cnt; xz++) {
			x = min.x + offs.x + (xz % int(cntx))*spacing;
			z = min.z + offs.z + (xz / int(cntx))*spacing;
			isSphere = true;
			if (x < min.x + 2 * radius)
			{
				if (pow(x-center.x,2) + pow(z - center.z, 2) + pow(y - center.y, 2) > radius2)
				{
					if (x > min.x + radius)
					{
						if (pow(z - zcenter2, 2) + pow(y - ycenter2, 2) > rc2)
							continue;
						else
							isSphere = false;
					}
					else
						continue;
				}
			}
			else
			{

				if (pow(z - zcenter2, 2) + pow(y - ycenter2, 2) > rc2 && pow(x-center3.x,2)+pow(y-center3.y,2)+pow(z-center3.z,2)>radius2)
					continue;
				else
					isSphere = false;
			}

			p = AddParticle();
			if (p != -1) {
				*(elasticID + p) = n;
				n++;
				(mPos + p)->Set(x, y, z);

				*(mClr + p) = COLORA(1, 0, 1, 1);  // (x-min.x)/dx
				*(m_alpha + p*MAX_FLUIDNUM + 0) = 1.0f; *(m_alpha_pre + p*MAX_FLUIDNUM + 0) = 1.0f;
				*(m_restMass + p) = m_fluidPMass[0];
				*(m_restDensity + p) = m_fluidDensity[0];

				*(m_visc + p) = m_fluidVisc[0];
				if (isSphere)
					*(MF_type + p) = 5; //1 means deformable
				else
					*(MF_type + p) = 4;
				rx = x - xcenter;
				ry = y - ycenter;


			}
		}
	}
	printf("%d fluid has %d particles\n", 0, n);
	return n;
}
void FluidSystem::
setupSPHexample()
{
	ParseXML_Bound("BoundInfo",1);
	example = _example;
	
	
	BI2Reader* bi2readers[10]; //ten pointers to build bi2reader dynamically, in use now
	char biname[200];

	//parse the xml and adjust some parameters according to scaleP
	ParseMFXML ( "MultiScene", example, true );

	double particleVisc = m_Param[PVISC];

	//adjust the parametres according to the scale parameter
	scaleP3 = pow(scaleP,1.0/3.0);
	m_Param[PNUM] = 80000;
	m_Param[PMASS]/=scaleP;
	m_Param[PBMASS]/=scaleP;
	m_Param[PSMOOTHRADIUS]/=scaleP3;
	m_Param[PRADIUS]/=scaleP3;
	printf("scale P is %f, pnum is %d\n", scaleP, m_Param[PNUM]);
	m_Param [PNUM]*=scaleP;

	m_Param [PGRIDSIZE] = 2*m_Param[PSMOOTHRADIUS] / m_Param[PGRID_DENSITY];
	
	m_fluidPMass[0] = m_Param[PMASS] * massRatio.x;
	m_fluidPMass[1] = m_Param[PMASS] * massRatio.y;
	m_fluidPMass[2] = m_Param[PMASS] * massRatio.z;
	m_fluidPMass[3] = m_Param[PMASS] * massRatio.w;
	m_fluidDensity[0] = 600.0*densityRatio.x;
	m_fluidDensity[1] = 600.0*densityRatio.y;
	m_fluidDensity[2] = 600.0*densityRatio.z;
	m_fluidDensity[3] = 600.0*densityRatio.w;
	m_fluidVisc[0] =  particleVisc*viscRatio.x;
	m_fluidVisc[1] =  particleVisc*viscRatio.y;
	m_fluidVisc[2] =  particleVisc*viscRatio.z;
	m_fluidVisc[3] =  particleVisc*viscRatio.w;

	restColorValue[0] = colorValue.x;
	restColorValue[1] = colorValue.y;
	restColorValue[2] = colorValue.z;

	AllocateParticles ( m_Param[PNUM] );
	AllocatePackBuf ();
	SetupKernels ();
	SetupSpacing ();
	numElasticPoints = 0;
	int solidNum = 0, liquidNum = 0, boundaryNum = 0;
	int pNum = 0;
	switch (_example) {
	case 1:
		m_Vec[PINITMIN].Set(volumes[0].x, volumes[0].y, volumes[0].z);
		m_Vec[PINITMAX].Set(volumes[1].x, volumes[1].y, volumes[1].z);
		solidNum += SetupMfAddGridSolid(m_Vec[PINITMIN], m_Vec[PINITMAX], m_Param[PSPACING], Vector3DF(0, 0, 0), 5);
		m_Vec[PINITMIN].Set(volumes[2].x, volumes[2].y, volumes[2].z);
		m_Vec[PINITMAX].Set(volumes[3].x, volumes[3].y, volumes[3].z);
		solidNum += SetupMfAddGridSolid(m_Vec[PINITMIN], m_Vec[PINITMAX], m_Param[PSPACING], Vector3DF(0, 0, 0), 4);
		m_Vec[PINITMIN].Set(volumes[4].x, volumes[4].y, volumes[4].z);
		m_Vec[PINITMAX].Set(volumes[5].x, volumes[5].y, volumes[5].z);
		solidNum += SetupMfAddGridSolid(m_Vec[PINITMIN], m_Vec[PINITMAX], m_Param[PSPACING], Vector3DF(0, 0, 0), 3);
		//ball
		m_Vec[PINITMIN].Set(volumes[6].x, volumes[6].y, volumes[6].z);
		m_Vec[PINITMAX].Set(volumes[7].x, volumes[7].y, volumes[7].z);
		//numElasticPoints = SetupMfAddDeformVolume(m_Vec[PINITMIN], m_Vec[PINITMAX], m_Param[PSPACING], Vector3DF(0, 0, 0), 0);
		NumPointsNoBound = numElasticPoints = 0;
		//fluid
		m_Vec[PINITMIN].Set(volumes[8].x, volumes[8].y, volumes[8].z);
		m_Vec[PINITMAX].Set(volumes[9].x, volumes[9].y, volumes[9].z);

		NumPointsNoBound += SetupMfAddBlendVolume(m_Vec[PINITMIN], m_Vec[PINITMAX], m_Param[PSPACING], Vector3DF(0, 0.1, 0));
		liquidNum = NumPointsNoBound;
		break;
	case 2:
		numElasticPoints = 0;
		m_Vec[PINITMIN].Set(volumes[0].x, volumes[0].y, volumes[0].z);
		m_Vec[PINITMAX].Set(volumes[1].x, volumes[1].y, volumes[1].z);
		numElasticPoints = SetupMfAddMagicWand(m_Vec[PINITMIN], m_Vec[PINITMAX], m_Param[PSPACING], Vector3DF(0, 0, 0));
		solidNum = numElasticPoints;
		//m_Vec[PINITMIN].Set(volumes[2].x, volumes[2].y, volumes[2].z);
		//m_Vec[PINITMAX].Set(volumes[3].x, volumes[3].y, volumes[3].z);
		//numElasticPoints += SetupMfAddMagicWand(m_Vec[PINITMIN], m_Vec[PINITMAX], m_Param[PSPACING], Vector3DF(0, 0, 0), 4);
		m_Vec[PINITMIN].Set(volumes[4].x, volumes[4].y, volumes[4].z);
		m_Vec[PINITMAX].Set(volumes[5].x, volumes[5].y, volumes[5].z);
		NumPointsNoBound = loadParticle("extra_particles\\fluids_ex2.dat");
		//NumPointsNoBound = SetupMfAddVolume(m_Vec[PINITMIN], m_Vec[PINITMAX], m_Param[PSPACING], Vector3DF(0, 0, 0), 3);
		
		liquidNum = NumPointsNoBound;
		NumPointsNoBound += numElasticPoints;
		
		break;
	case 3:
		NumPointsNoBound = numElasticPoints = 0;
		//fluid
		m_Vec[PINITMIN].Set(volumes[8].x, volumes[8].y, volumes[8].z);
		m_Vec[PINITMAX].Set(volumes[9].x, volumes[9].y, volumes[9].z);

		NumPointsNoBound += SetupMfAddBlendVolume(m_Vec[PINITMIN], m_Vec[PINITMAX], m_Param[PSPACING], Vector3DF(0, 0.1, 0));
		liquidNum = NumPointsNoBound;
		break;
	}
	pNum += NumPointsNoBound;
	cout << "without boundary, particle num is " << pNum << endl;
	
	//solveModel();
	//storeModel("bunny.txt");
	//SetupMfAddDeformVolume

	m_Vec [ PINITMIN ].Set (softBoundary[0].x, softBoundary[0].y, softBoundary[0].z);
	m_Vec [ PINITMAX ].Set (softBoundary[1].x, softBoundary[1].y, softBoundary[1].z);
	boundaryNum = SetupBoundary(m_Vec[PINITMIN], m_Vec[PINITMAX], m_Param[PSPACING], Vector3DF(0, 0, 0), 1);
	printf("liquid num is %d, solid num is %d, boundary num is %d\n", liquidNum, solidNum, boundaryNum);
	m_maxAllowedPoints = mNumPoints * EMIT_BUF_RATIO;
}
