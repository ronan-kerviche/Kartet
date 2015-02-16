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

	#include "Core/Exceptions.hpp"
	#include "Core/TemplateSharedMemory.hpp"
	#include "Core/Array.hpp"

namespace Kartet
{
	template<template<typename,typename> class Op, typename TOut, typename TExpr>
	__global__ void reduceKernelV1(const Layout layout, const TExpr expr, TOut* outputBuffer, const int maxPow2Half)
	{
		typedef typename ExpressionEvaluation<TExpr>::ReturnType ReturnType;

		TOut *sharedData = SharedMemory<TOut>();
		const index_t 	N = layout.getNumElements(),
				blockStart = blockIdx.y*(blockDim.y*2),
				remaining = (::min(static_cast<unsigned int>(N - blockStart), blockDim.y)),
				pa = blockStart + threadIdx.y,
				pb = pa + blockDim.y;

		// Load part A, part B, apply the operator, store in shared memory :
		index_t ia = 0, 
			ja = 0, 
			ka = 0,
			ib = 0, 
			jb = 0, 
			kb = 0;
		layout.unpackIndex(pa, ia, ja, ka);
		layout.unpackIndex(pb, ib, jb, kb);
		
		if(pb<N)
		{
			const ReturnType a = ExpressionEvaluation<TExpr>::evaluate(expr, layout, pa, ia, ja, ka),
				 	 b = ExpressionEvaluation<TExpr>::evaluate(expr, layout, pb, ib, jb, kb);
			sharedData[threadIdx.y]	= static_cast<TOut>(Op<ReturnType, ReturnType>::apply(a, b));
		}
		else if(pa<N) // thread not in the remaining half
			sharedData[threadIdx.y] = ExpressionEvaluation<TExpr>::evaluate(expr, layout, pa, ia, ja, ka);

		// Reduce :
		for(int k=maxPow2Half; k>0; k/=2)
		{
			__syncthreads();
			if(threadIdx.y<k && (threadIdx.y+k)<remaining && threadIdx.y<remaining)
				sharedData[threadIdx.y] = Op<TOut, TOut>::apply(sharedData[threadIdx.y], sharedData[threadIdx.y + k]);
		}
		
		// Store to this block ID :
		if(threadIdx.y==0)
			outputBuffer[blockIdx.y] = sharedData[0];
	}
	
	class ReduceContext
	{
		private :
			const int fillNumBlocks;
			const size_t maxMemory;			
			char	*hostPtr,
				*devicePtr;

			__host__ ReduceContext(const ReduceContext&);

		public :
			__host__ inline ReduceContext(void)
			 : 	fillNumBlocks(32),
				maxMemory(16777216), // 16MB, this should be almost the maximum VRAM size divided by 1024
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
				// Build the reduction stategy :

				// Apply :
			}*/

			template<template<typename,typename> class Op, typename TOut, typename TExpr>
			__host__ TOut reduceV1(const Layout& layout, const TExpr& expr)
			{
				if(!layout.isMonolithic())
					throw InvalidOperation;

				bool firstStep = true;
				TOut	*castHostPtr   = reinterpret_cast<TOut*>(hostPtr),
					*castDevicePtr = reinterpret_cast<TOut*>(devicePtr);

				index_t numElements = layout.getNumElements();
				dim3 	blockSize,
					numBlocks;
				blockSize.x 	= 1;
				blockSize.y 	= std::min(Layout::StaticContainer<void>::numThreads, (numElements+1)/2);
				blockSize.z 	= 1;
				numBlocks.x	= 1;
				numBlocks.y	= (numElements + 2*blockSize.y - 1) / (2*blockSize.y);
				numBlocks.z	= 1;

				while(firstStep || numElements>=(Layout::StaticContainer<void>::numThreads/4))
				{
					const size_t sharedMemorySize = blockSize.y * sizeof(TOut);
					const int maxPow2Half = 1 << (static_cast<int>(std::floor(std::log(blockSize.y-1)/std::log(2))));

					if(firstStep)
					{
						reduceKernelV1<Op><<<numBlocks, blockSize, sharedMemorySize>>>(layout, expr, castDevicePtr, maxPow2Half);
						firstStep = false;
					}					
					else
						reduceKernelV1<Op><<<numBlocks, blockSize, sharedMemorySize>>>(Layout(numElements), castDevicePtr, castDevicePtr, maxPow2Half);

					// Next step :
					numElements = numBlocks.y;
					blockSize.y 	= std::max(static_cast<index_t>(1), std::min(Layout::StaticContainer<void>::numThreads, static_cast<index_t>((numElements+1)/2)));
					numBlocks.y	= (numElements + 2*blockSize.y - 1) / (2*blockSize.y);
				}

				// Finish on the CPU side :
				cudaError_t err = cudaMemcpy(reinterpret_cast<void*>(castHostPtr), reinterpret_cast<void*>(castDevicePtr), numElements*sizeof(TOut), cudaMemcpyDeviceToHost);
				if(err!=cudaSuccess)
					throw static_cast<Exception>(CudaExceptionsOffset + err);

				for(int k=1; k<numElements; k++)
					castHostPtr[0] = Op<TOut, TOut>::apply(castHostPtr[0], castHostPtr[k]);

				// Return :
				return castHostPtr[0];
			}

			template<template<typename,typename> class Op, typename TOut, typename TExpr>
			__host__ TOut reduceV2(const Layout& layout, const TExpr& expr)
			{
				bool firstStep = true;
				TOut	*castHostPtr   = reinterpret_cast<TOut*>(hostPtr),
					*castDevicePtr = reinterpret_cast<TOut*>(devicePtr);

				//const int 	maxBlocksPerFragment = std::max(std::max(Layout::StaticContainer<void>::numThreads, layout.getNumElementsPerFragment()) / Layout::StaticContainer<void>::numThreads, fillNumBlocks),
				//		numReadPerBlock = layout.getNumElementsPerFragment()/(maxBlocksPerFragment*std::min(Layout::StaticContainer<void>::numThreads, layout.getNumElementsPerFragment());
				
				/*if(leadingColumns==numRows)
				{
					if(leadingSlices==(numRows * numColumns))
					{
						// can address the full array
					}
					else
					{
						// can only address slices
					}
				}
				else
					// can only address columns*/	
				
			}

			// Specifics :
			template<typename TExpr>
			__host__ typename ExpressionEvaluation<TExpr>::ReturnType min(const Layout& layout, const TExpr& expr)
			{
				return reduceV1<BinOp_min, typename ExpressionEvaluation<TExpr>::ReturnType>(layout, expr);
			}

			template<typename T>
			__host__ T min(const Accessor<T>& accessor)
			{
				return reduceV1<BinOp_min, T>(accessor, accessor);
			}

			template<typename TExpr>
			__host__ typename ExpressionEvaluation<TExpr>::ReturnType max(const Layout& layout, const TExpr& expr)
			{
				return reduceV1<BinOp_max, typename ExpressionEvaluation<TExpr>::ReturnType>(layout, expr);
			}

			template<typename T>
			__host__ T max(const Accessor<T>& accessor)
			{
				return reduceV1<BinOp_max, T>(accessor, accessor);
			}

			template<typename TExpr>
			__host__ typename ExpressionEvaluation<TExpr>::ReturnType sum(const Layout& layout, const TExpr& expr)
			{
				return reduceV1<BinOp_Plus, typename ExpressionEvaluation<TExpr>::ReturnType>(layout, expr);
			}

			template<typename T>
			__host__ T sum(const Accessor<T>& accessor)
			{
				return reduceV1<BinOp_Plus, T>(accessor, accessor);
			}

			template<typename TExpr>
			__host__ typename ExpressionEvaluation<TExpr>::ReturnType prod(const Layout& layout, const TExpr& expr)
			{
				return reduceV1<BinOp_Times, typename ExpressionEvaluation<TExpr>::ReturnType>(layout, expr);
			}

			template<typename T>
			__host__ T prod(const Accessor<T>& accessor)
			{
				return reduceV1<BinOp_Times, T>(accessor, accessor);
			}

			template<typename TExpr>
			__host__ bool all(const Layout& layout,const TExpr& expr)
			{
				return reduceV1<BinOp_And, bool>(layout, expr);
			}

			template<typename T>
			__host__ bool all(const Accessor<T>& accessor)
			{
				return reduceV1<BinOp_And, bool>(accessor, accessor);
			}

			template<typename TExpr>
			__host__ bool any(const Layout& layout,const TExpr& expr)
			{
				return reduceV1<BinOp_Or, bool>(layout, expr);
			}

			template<typename T>
			__host__ bool any(const Accessor<T>& accessor)
			{
				return reduceV1<BinOp_Or, bool>(accessor, accessor);
			}
	};

} // namespace Kartet

#endif

