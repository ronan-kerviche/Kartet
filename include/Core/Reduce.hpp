/* ************************************************************************************************************* */
/*                                                                                                               */
/*     Kartet                                                                                                    */
/*     A Simple C++ Array Library for CUDA                                                                       */
/*                                                                                                               */
/*     LICENSE : The MIT License                                                                                 */
/*     Copyright (c) 2015 Ronan Kerviche                                                                         */
/*                                                                                                               */
/*     Permission is hereby granted, free of charge, to any person obtaining a copy                              */
/*     of this software and associated documentation files (the "Software"), to deal                             */
/*     in the Software without restriction, including without limitation the rights                              */
/*     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell                                 */
/*     copies of the Software, and to permit persons to whom the Software is                                     */
/*     furnished to do so, subject to the following conditions:                                                  */
/*                                                                                                               */
/*     The above copyright notice and this permission notice shall be included in                                */
/*     all copies or substantial portions of the Software.                                                       */
/*                                                                                                               */
/*     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR                                */
/*     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,                                  */
/*     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE                               */
/*     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER                                    */
/*     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,                             */
/*     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN                                 */
/*     THE SOFTWARE.                                                                                             */
/*                                                                                                               */
/* ************************************************************************************************************* */

#ifndef __KARTET_REDUCE__
#define __KARTET_REDUCE__

	#include <limits>
	#include "Core/Exceptions.hpp"
	#include "Core/TemplateSharedMemory.hpp"
	#include "Core/Array.hpp"

namespace Kartet
{
	template<template<typename,typename> class Op, typename TOut, typename TExpr>
	__global__ void reduceKernel(const Layout layout, const dim3 blockStride, const TExpr expr, const TOut defaultValue, TOut* outputBuffer, const int totalNumThreads, const int maxPow2Half)
	{
		typedef typename ExpressionEvaluation<TExpr>::ReturnType ReturnType;
		TOut *sharedData = SharedMemory<TOut>();

		const unsigned int blockId = (blockIdx.z*gridDim.y + blockIdx.y)*gridDim.x + blockIdx.x,
				   threadId = (threadIdx.z*blockDim.y + threadIdx.y)*blockDim.x + threadIdx.x;
		const index_t 	i = blockIdx.x * blockDim.x + threadIdx.x,
				j = blockIdx.y * blockDim.y + threadIdx.y,
				k = blockIdx.z * blockDim.z + threadIdx.z;

		// Initialization :
		sharedData[threadId] = defaultValue;

		if(layout.isInside(i, j, k))
		{
			// First elements :
			ReturnType v = static_cast<ReturnType>(0);
			
			// WORKING, AND FASTER (???) :
			for(int kl=0; kl<blockStride.z; kl++)
				for(int jl=0; jl<blockStride.y; jl++)
					for(int il=0; il<blockStride.x; il++)
					{
						index_t	iL = i + il * gridDim.x * blockDim.x,
							jL = j + jl * gridDim.y * blockDim.y,
							kL = k + kl * gridDim.z * blockDim.z,
							pL = layout.getIndex(iL, jL, kL);
						if(layout.isInside(iL, jL, kL))
							v = Op<ReturnType, ReturnType>::apply(v, ExpressionEvaluation<TExpr>::evaluate(expr, layout, pL, iL, jL, kL));
					}

			// THIS ONE IS SLOWER (???) :
			/*index_t   iL = i,
				  jL = j,
				  kL = k,
				  pL = 0;
			const index_t 	stepX = gridDim.x * blockDim.x, 
					stepY = gridDim.y * blockDim.y, 
					stepZ = gridDim.z * blockDim.z;
			for(int kc=0; kc<blockStride.z; kc++)
			{
				for(int jc=0; jc<blockStride.y; jc++)
				{				
					for(int ic=0; ic<blockStride.x; ic++)
					{
						pL = layout.getIndex(iL, jL, kL);
						if(layout.isInside(iL, jL, kL))
							v = Op<ReturnType, ReturnType>::apply(v, ExpressionEvaluation<TExpr>::evaluate(expr, layout, pL, iL, jL, kL));
						iL += stepX;
					}
					iL = i;
					jL += stepY;
				}
				jL = j;
				kL += stepZ;
			}*/
			sharedData[threadId] = static_cast<TOut>(v);

			// Reduce :
			for(int k=maxPow2Half; k>0; k/=2)
			{
				__syncthreads();
				if(threadId<k && (threadId+k)<totalNumThreads && threadId<totalNumThreads)
					sharedData[threadId] = Op<TOut, TOut>::apply(sharedData[threadId], sharedData[threadId + k]);
			}
		
			// Store to this block ID :
			if(threadId==0)
				outputBuffer[blockId] = sharedData[0];
		}
	}
	
	class ReduceContext
	{
		private :
			const unsigned int fillNumBlocks;
			const size_t maxMemory;			
			char	*hostPtr,
				*devicePtr;

			__host__ ReduceContext(const ReduceContext&);

		public :
			__host__ inline ReduceContext(void)
			 : 	fillNumBlocks(128),
				maxMemory(16384),
				hostPtr(NULL),
				devicePtr(NULL)		
			{
				STATIC_ASSERT(sizeof(char)==1)

				hostPtr = new char[maxMemory];
				cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&devicePtr), maxMemory);
				if(err!=cudaSuccess)
					throw static_cast<Exception>(CudaExceptionsOffset + err);
			}

			__host__ inline ~ReduceContext(void)
			{
				delete[] hostPtr;
				cudaError_t err = cudaFree(devicePtr);
				if(err!=cudaSuccess)
					throw static_cast<Exception>(CudaExceptionsOffset + err);
			}

			/*template<template<typename,typename> class Op, typename TOut, typename TExpr>
			__host__ void reduce(const TExpr expr, const Accessor<TOut>& output)
			{
				// todo.
			}*/

			template<template<typename,typename> class Op, typename TOut, typename TExpr>
			__host__ TOut reduce(const Layout& layout, const TExpr& expr, const TOut defaultValue)
			{
				TOut	*castHostPtr   = reinterpret_cast<TOut*>(hostPtr),
					*castDevicePtr = reinterpret_cast<TOut*>(devicePtr);

				const dim3	blockSize = layout.getBlockSize(),
						numBlocks = layout.getNumBlock();

				// Compute the stride for each block, in order to limit the total number of blocks to fillNumBlocks :
				dim3	reducedNumBlocks,
					blockStride;

				reducedNumBlocks.x	= std::min(numBlocks.x, fillNumBlocks);
				blockStride.x		= (numBlocks.x + reducedNumBlocks.x - 1) / reducedNumBlocks.x;
				reducedNumBlocks.y	= std::min(numBlocks.y, fillNumBlocks / reducedNumBlocks.x);
				blockStride.y		= (numBlocks.y + reducedNumBlocks.y - 1) / reducedNumBlocks.y;
				reducedNumBlocks.z	= std::min(numBlocks.z, fillNumBlocks / (reducedNumBlocks.x * reducedNumBlocks.y));
				blockStride.z		= (numBlocks.z + reducedNumBlocks.z - 1) / reducedNumBlocks.z;
				
				const int totalNumBlocks = reducedNumBlocks.x * reducedNumBlocks.y * reducedNumBlocks.z,
					  totalNumThreads = blockSize.x * blockSize.y * blockSize.z,
					  maxPow2Half = 1 << (static_cast<int>(std::floor(std::log(totalNumThreads-1)/std::log(2))));
				const size_t sharedMemorySize = totalNumThreads * sizeof(TOut);

				/*std::cout << "numBlocks        : " << numBlocks.x << ", " << numBlocks.y << ", " << numBlocks.z << std::endl;
				std::cout << "blockSize        : " << blockSize.x << ", " << blockSize.y << ", " << blockSize.z << std::endl;
				std::cout << "reducedNumBlocks : " << reducedNumBlocks.x << ", " << reducedNumBlocks.y << ", " << reducedNumBlocks.z << std::endl;
				std::cout << "blockStride      : " << blockStride.x << ", " << blockStride.y << ", " << blockStride.z << std::endl;
				std::cout << "totalNumBlocks   : " << totalNumBlocks << std::endl;
				std::cout << "totalNumThreads  : " << totalNumThreads << std::endl;
				std::cout << "maxPow2Half      : " <<  maxPow2Half << std::endl;*/

				// Do the single-pass reduction :
				reduceKernel<Op><<<reducedNumBlocks, blockSize, sharedMemorySize>>>(layout, blockStride, expr, defaultValue, castDevicePtr,  totalNumThreads, maxPow2Half);

				// Copy back to the Host side and complete :
				cudaError_t err = cudaMemcpy(reinterpret_cast<void*>(castHostPtr), reinterpret_cast<void*>(castDevicePtr), totalNumBlocks*sizeof(TOut), cudaMemcpyDeviceToHost);
				if(err!=cudaSuccess)
					throw static_cast<Exception>(CudaExceptionsOffset + err);

				for(int k=1; k<totalNumBlocks; k++)
					castHostPtr[0] = Op<TOut, TOut>::apply(castHostPtr[0], castHostPtr[k]);

				// Return :
				return castHostPtr[0];
			}

			// Specifics :
			template<typename TExpr>
			__host__ typename ExpressionEvaluation<TExpr>::ReturnType min(const Layout& layout, const TExpr& expr)
			{
				typedef typename ExpressionEvaluation<TExpr>::ReturnType ReturnType;
				return reduce<BinOp_min, ReturnType>(layout, expr, std::numeric_limits<ReturnType>::max());
			}

			template<typename T>
			__host__ T min(const Accessor<T>& accessor)
			{
				return min(accessor, accessor);
			}

			template<typename TExpr>
			__host__ typename ExpressionEvaluation<TExpr>::ReturnType max(const Layout& layout, const TExpr& expr)
			{
				typedef typename ExpressionEvaluation<TExpr>::ReturnType ReturnType;
				return reduce<BinOp_max, ReturnType>(layout, expr, std::numeric_limits<ReturnType>::max());
			}

			template<typename T>
			__host__ T max(const Accessor<T>& accessor)
			{
				return max(accessor, accessor);
			}

			template<typename TExpr>
			__host__ typename ExpressionEvaluation<TExpr>::ReturnType sum(const Layout& layout, const TExpr& expr)
			{
				typedef typename ExpressionEvaluation<TExpr>::ReturnType ReturnType;
				return reduce<BinOp_Plus, ReturnType>(layout, expr, complexCopy<ReturnType>(0));
			}

			template<typename T>
			__host__ T sum(const Accessor<T>& accessor)
			{
				return sum(accessor, accessor);
			}

			template<typename TExpr>
			__host__ typename ExpressionEvaluation<TExpr>::ReturnType prod(const Layout& layout, const TExpr& expr)
			{
				typedef typename ExpressionEvaluation<TExpr>::ReturnType ReturnType;
				return reduce<BinOp_Times, ReturnType>(layout, expr, complexCopy<ReturnType>(1));
			}

			template<typename T>
			__host__ T prod(const Accessor<T>& accessor)
			{
				return prod(accessor, accessor);
			}

			template<typename TExpr>
			__host__ bool all(const Layout& layout,const TExpr& expr)
			{
				return reduce<BinOp_And, bool>(layout, expr, true);
			}

			template<typename T>
			__host__ bool all(const Accessor<T>& accessor)
			{
				return all(accessor, accessor);
			}

			template<typename TExpr>
			__host__ bool any(const Layout& layout,const TExpr& expr)
			{
				return reduce<BinOp_Or, bool>(layout, expr, false);
			}

			template<typename T>
			__host__ bool any(const Accessor<T>& accessor)
			{
				return any(accessor, accessor);
			}
	};

} // namespace Kartet

#endif

