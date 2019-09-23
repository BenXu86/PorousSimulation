/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 */
#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <scanLargeArray_kernel.cu>
#include <assert.h>

inline bool 
isPowerOfTwo(int n)
{
    return ((n&(n-1))==0) ;
}

inline int 
floorPow2(int n)
{
#ifdef WIN32
    // method 2
    return 1 << (int)logb((float)n);
#else
    // method 1
    // float nf = (float)n;
    // return 1 << (((*(int*)&nf) >> 23) - 127); 
    int exp;
    frexp((float)n, &exp);
    return 1 << (exp - 1);
#endif
}

#define BLOCK_SIZE 256

float** g_scanBlockSums;
unsigned int g_numEltsAllocated = 0;
unsigned int g_numLevelsAllocated = 0;

void preallocBlockSums(unsigned int maxNumElements)
{
    assert(g_numEltsAllocated == 0); // shouldn't be called 

    g_numEltsAllocated = maxNumElements;

    unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
    unsigned int numElts = maxNumElements;

    int level = 0;

    do
    {       
        unsigned int numBlocks = 
            max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1)
        {
            level++;
        }
        numElts = numBlocks;
    } while (numElts > 1);

    g_scanBlockSums = (float**) malloc(level * sizeof(float*));
    g_numLevelsAllocated = level;
    
    numElts = maxNumElements;
    level = 0;
    
    do
    {       
        unsigned int numBlocks = 
            max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1) 
        {
            cutilSafeCall(cudaMalloc((void**) &g_scanBlockSums[level++],  
                                      numBlocks * sizeof(float)));
        }
        numElts = numBlocks;
    } while (numElts > 1);

    cutilCheckMsg("preallocBlockSums");
}

void deallocBlockSums()
{
    for (unsigned int i = 0; i < g_numLevelsAllocated; i++)
    {
        cudaFree(g_scanBlockSums[i]);
    }

    cutilCheckMsg("deallocBlockSums");
    
    free((void**)g_scanBlockSums);

    g_scanBlockSums = 0;
    g_numEltsAllocated = 0;
    g_numLevelsAllocated = 0;
}


void prescanArrayRecursive(float *outArray, 
                           const float *inArray, 
                           int numElements, 
                           int level)
{
    unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
    unsigned int numBlocks = 
        max(1, (int)ceil((float)numElements / (2.f * blockSize)));
    unsigned int numThreads;

    if (numBlocks > 1)
        numThreads = blockSize;
    else if (isPowerOfTwo(numElements))
        numThreads = numElements / 2;
    else
        numThreads = floorPow2(numElements);

    unsigned int numEltsPerBlock = numThreads * 2;

    // if this is a non-power-of-2 array, the last block will be non-full
    // compute the smallest power of 2 able to compute its scan.
    unsigned int numEltsLastBlock = 
        numElements - (numBlocks-1) * numEltsPerBlock;
    unsigned int numThreadsLastBlock = max(1, numEltsLastBlock / 2);
    unsigned int np2LastBlock = 0;
    unsigned int sharedMemLastBlock = 0;
    
    if (numEltsLastBlock != numEltsPerBlock)
    {
        np2LastBlock = 1;

        if(!isPowerOfTwo(numEltsLastBlock))
            numThreadsLastBlock = floorPow2(numEltsLastBlock);    
        
        unsigned int extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
        sharedMemLastBlock = 
            sizeof(float) * (2 * numThreadsLastBlock + extraSpace);
    }

    // padding space is used to avoid shared memory bank conflicts
    unsigned int extraSpace = numEltsPerBlock / NUM_BANKS;
    unsigned int sharedMemSize = 
        sizeof(float) * (numEltsPerBlock + extraSpace);

#ifdef DEBUG
    if (numBlocks > 1)
    {
        assert(g_numEltsAllocated >= numElements);
    }
#endif

    // setup execution parameters
    // if NP2, we process the last block separately
    dim3  grid(max(1, numBlocks - np2LastBlock), 1, 1); 
    dim3  threads(numThreads, 1, 1);

    // make sure there are no CUDA errors before we start
    cutilCheckMsg("prescanArrayRecursive before kernels");

    // execute the scan
    if (numBlocks > 1)
    {
        prescan<true, false><<< grid, threads, sharedMemSize >>>(outArray, 
                                                                 inArray, 
                                                                 g_scanBlockSums[level],
                                                                 numThreads * 2, 0, 0);
        cutilCheckMsg("prescanWithBlockSums");
        if (np2LastBlock)
        {
            prescan<true, true><<< 1, numThreadsLastBlock, sharedMemLastBlock >>>
                (outArray, inArray, g_scanBlockSums[level], numEltsLastBlock, 
                 numBlocks - 1, numElements - numEltsLastBlock);
            cutilCheckMsg("prescanNP2WithBlockSums");
        }

        // After scanning all the sub-blocks, we are mostly done.  But now we 
        // need to take all of the last values of the sub-blocks and scan those.  
        // This will give us a new value that must be sdded to each block to 
        // get the final results.
        // recursive (CPU) call
        prescanArrayRecursive(g_scanBlockSums[level], 
                              g_scanBlockSums[level], 
                              numBlocks, 
                              level+1);

        uniformAdd<<< grid, threads >>>(outArray, 
                                        g_scanBlockSums[level], 
                                        numElements - numEltsLastBlock, 
                                        0, 0);
        cutilCheckMsg("uniformAdd");
        if (np2LastBlock)
        {
            uniformAdd<<< 1, numThreadsLastBlock >>>(outArray, 
                                                     g_scanBlockSums[level], 
                                                     numEltsLastBlock, 
                                                     numBlocks - 1, 
                                                     numElements - numEltsLastBlock);
            cutilCheckMsg("uniformAdd");
        }
    }
    else if (isPowerOfTwo(numElements))
    {
        prescan<false, false><<< grid, threads, sharedMemSize >>>(outArray, inArray,
                                                                  0, numThreads * 2, 0, 0);
        cutilCheckMsg("prescan");
    }
    else
    {
         prescan<false, true><<< grid, threads, sharedMemSize >>>(outArray, inArray, 
                                                                  0, numElements, 0, 0);
         cutilCheckMsg("prescanNP2");
    }
}

void prescanArray(float *outArray, float *inArray, int numElements)
{
    prescanArrayRecursive(outArray, inArray, numElements, 0);
}


#endif // _PRESCAN_CU_
