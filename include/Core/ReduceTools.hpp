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

/**
	\file    ReduceTools.hpp
	\brief   Reduction context implementations.
	\author  R. Kerviche
	\date    November 1st 2009
**/

#ifndef __KARTET_REDUCE_TOOLS__
#define __KARTET_REDUCE_TOOLS__

namespace Kartet
{
// Kernels :
#ifdef __CUDACC__
	template<template<typename,typename> class Op, typename TOut, typename TExpr>
	__global__ void reduceKernel(const Layout layout, const dim3 blockSteps, const TExpr expr, const typename ExpressionEvaluation<TExpr>::ReturnType defaultValue, TOut* outputBuffer, const unsigned int totalNumThreads, const unsigned int maxPow2Half)
	{
		typedef typename ExpressionEvaluation<TExpr>::ReturnType ReturnType;
		ReturnType *sharedData = SharedMemory<ReturnType>();

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
			ReturnType v = defaultValue;
			
			// WORKING, AND FASTER (???) :
			for(int kl=0; kl<blockSteps.z; kl++)
				for(int jl=0; jl<blockSteps.y; jl++)
					for(int il=0; il<blockSteps.x; il++)
					{
						index_t	iL = i + il * gridDim.x * blockDim.x,
							jL = j + jl * gridDim.y * blockDim.y,
							kL = k + kl * gridDim.z * blockDim.z;
						if(layout.isInside(iL, jL, kL))
							v = Op<ReturnType, ReturnType>::apply(v, ExpressionEvaluation<TExpr>::evaluate(expr, layout, iL, jL, kL));
					}

			// THIS ONE IS SLOWER (???) :
			/*index_t   iL = i,
				  jL = j,
				  kL = k,
				  pL = 0;
			const index_t 	stepX = gridDim.x * blockDim.x, 
					stepY = gridDim.y * blockDim.y, 
					stepZ = gridDim.z * blockDim.z;
			for(int kc=0; kc<blockSteps.z; kc++)
			{
				for(int jc=0; jc<blockSteps.y; jc++)
				{				
					for(int ic=0; ic<blockSteps.x; ic++)
					{
						pL = layout.getPosition(iL, jL, kL);
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
			sharedData[threadId] = v;

			// Reduce :
			for(int k=maxPow2Half; k>0; k/=2)
			{
				__syncthreads();
				if(threadId<k && (threadId+k)<totalNumThreads && threadId<totalNumThreads)
					sharedData[threadId] = Op<TOut, TOut>::apply(sharedData[threadId], sharedData[threadId + k]);
			}
		
			// Store to this block ID :
			if(threadId==0)
				outputBuffer[blockId] = complexCopy<TOut>(sharedData[0]);
		}
	}

	template<template<typename,typename> class Op, typename TOut, Location l, typename TExpr>
	__global__ void reduceToLayoutKernel_LargeReductionMode(const Layout layout, const Layout reductionBlockLayout, const dim3 blockSteps, const TExpr expr, const typename ExpressionEvaluation<TExpr>::ReturnType defaultValue, const Accessor<TOut,l> output, const unsigned int totalNumThreads, const unsigned int maxPow2Half)
	{
		typedef typename ExpressionEvaluation<TExpr>::ReturnType ReturnType;
		ReturnType *sharedData = SharedMemory<ReturnType>();

		const unsigned int threadId = (threadIdx.z*blockDim.y + threadIdx.y)*blockDim.x + threadIdx.x;
		const index_t 	i = blockIdx.x * reductionBlockLayout.numRows() + threadIdx.x,
				j = blockIdx.y * reductionBlockLayout.numColumns() + threadIdx.y,
				k = blockIdx.z * reductionBlockLayout.numSlices() + threadIdx.z;

		// Initialization :
		sharedData[threadId] = defaultValue;

		// First elements :
		ReturnType v = defaultValue;
		for(int kl=0; kl<blockSteps.z; kl++)
			for(int jl=0; jl<blockSteps.y; jl++)
				for(int il=0; il<blockSteps.x; il++)
				{
					index_t	iB = threadIdx.x + il * blockDim.x,
						jB = threadIdx.y + jl * blockDim.y,
						kB = threadIdx.z + kl * blockDim.z,
						iL = i + il * blockDim.x,
						jL = j + jl * blockDim.y,
						kL = k + kl * blockDim.z;
					if(layout.isInside(iL, jL, kL) && reductionBlockLayout.isInside(iB, jB, kB))
						v = Op<ReturnType, ReturnType>::apply(v, ExpressionEvaluation<TExpr>::evaluate(expr, layout, iL, jL, kL));
				}
		sharedData[threadId] = v;

		// Reduce :
		for(int k=maxPow2Half; k>0; k/=2)
		{
			__syncthreads();
			if(threadId<k && (threadId+k)<totalNumThreads && threadId<totalNumThreads)
				sharedData[threadId] = Op<TOut, TOut>::apply(sharedData[threadId], sharedData[threadId + k]);
		}
	
		// Store to the right block :
		if(threadId==0)
			output.data(blockIdx.x, blockIdx.y, blockIdx.z) = complexCopy<TOut>(sharedData[0]);
	}

	template<template<typename,typename> class Op, typename TOut, Location l, typename TExpr>
	__global__ void reduceToLayoutKernel_SmallReductionMode(const Layout layout, const Layout reductionBlockLayout, const dim3 blockSteps, const dim3 numSubReductionBlocks, const TExpr expr, const typename ExpressionEvaluation<TExpr>::ReturnType defaultValue, const Accessor<TOut,l> output)
	{
		typedef typename ExpressionEvaluation<TExpr>::ReturnType ReturnType;
		ReturnType *sharedData = SharedMemory<ReturnType>();

		const unsigned int subThreadId = (threadIdx.z*blockDim.y + threadIdx.y)*blockDim.x,
				   threadId = subThreadId + threadIdx.x;
		const index_t 	i = blockIdx.x * numSubReductionBlocks.x * reductionBlockLayout.numRows() + threadIdx.x,
				j = blockIdx.y * numSubReductionBlocks.y * reductionBlockLayout.numColumns() + threadIdx.y,
				k = blockIdx.z * numSubReductionBlocks.z * reductionBlockLayout.numSlices() + threadIdx.z,
				iB = threadIdx.x % reductionBlockLayout.numRows(),
				jB = threadIdx.y % reductionBlockLayout.numColumns(),
				kB = threadIdx.z % reductionBlockLayout.numSlices();
		const bool 	mainThread = (jB==0 && kB==0);

		// Initialization :
		sharedData[threadId] = defaultValue;

		// First elements :
		ReturnType v = defaultValue;
		for(unsigned int kl=0; kl<blockSteps.z; kl++)
			for(unsigned int jl=0; jl<blockSteps.y; jl++)
				for(unsigned int il=0; il<blockSteps.x; il++)
				{
					index_t	iBL = iB + il * blockDim.x,
						jBL = jB + jl * blockDim.y,
						kBL = kB + kl * blockDim.z,
						iL = i + il * blockDim.x,
						jL = j + jl * blockDim.y,
						kL = k + kl * blockDim.z;
					if(layout.isInside(iL, jL, kL) && reductionBlockLayout.isInside(iBL, jBL, kBL))
						v = Op<ReturnType, ReturnType>::apply(v, ExpressionEvaluation<TExpr>::evaluate(expr, layout, iL, jL, kL));
				}
		sharedData[threadId] = v;
		v = defaultValue; // Reset, important for what is next.
		__syncthreads();
		
		// Second pass, within the shared memory :
		if(mainThread)
		{
			for(unsigned int kl=0; kl<reductionBlockLayout.numSlices()/blockSteps.z; kl++)
				for(unsigned int jl=0; jl<reductionBlockLayout.numColumns()/blockSteps.y; jl++)
				{
					const unsigned int	jCL = threadIdx.y + jl,
								kCL = threadIdx.z + kl,
								p =  (kCL*blockDim.y + jCL)*blockDim.x + threadIdx.x;
					v = Op<ReturnType, ReturnType>::apply(v, sharedData[p]);
				}
			sharedData[threadId] = v;
		}
		v = defaultValue; // Reset, important for what is next.
		__syncthreads();

		// Finish the reduction and store to the right block :
		if(threadIdx.x<numSubReductionBlocks.x && mainThread)
		{
			// slooooooow, here the reads to the shared memory are not coalesced.
			// I need to improve this to a better method.
			for(unsigned int il=0; il<reductionBlockLayout.numRows(); il++)
				v = Op<ReturnType, ReturnType>::apply(v, sharedData[subThreadId + threadIdx.x*reductionBlockLayout.numRows() + il]);

			const index_t	iBP = blockIdx.x * numSubReductionBlocks.x + threadIdx.x,
					jBP = blockIdx.y * numSubReductionBlocks.y + threadIdx.y/reductionBlockLayout.numColumns(),
					kBP = blockIdx.z * numSubReductionBlocks.z + threadIdx.z/reductionBlockLayout.numSlices();
			
			if(output.isInside(iBP, jBP, kBP))
				output.data(iBP, jBP, kBP) = complexCopy<TOut>(v);	
		}
	}
#endif

// Implementation :
	/**
	\brief ReduceContext constructor.
	**/
	__host__ inline ReduceContext::ReduceContext(void)
	 : 	fillNumBlocks(128),
		maxMemory(16384),
		hostPtr(NULL),
		devicePtr(NULL)		
	{
		StaticAssert<sizeof(char)==1>();

		#ifdef __CUDACC__
		hostPtr = new char[maxMemory];
		cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&devicePtr), maxMemory);
		if(err!=cudaSuccess)
			throw static_cast<Exception>(CudaExceptionsOffset + err);
		#endif
	}

	__host__ inline ReduceContext::~ReduceContext(void)
	{
		#ifdef __CUDACC__
		delete[] hostPtr;
		cudaError_t err = cudaFree(devicePtr);
		if(err!=cudaSuccess)
			throw static_cast<Exception>(CudaExceptionsOffset + err);
		#endif
	}

	/**
	\brief Generic global reduction operation.
	\tparam Op Reduction operator.
	\param layout Layout of the data to be reduced.
	\param expr Expression or data to be reduced.
	\param defaultValue Default value used as initialization.
	\return The result of the reduction operation.
	**/
	template<template<typename,typename> class Op, typename TOut, typename TExpr>
	__host__ TOut ReduceContext::reduce(const Layout& layout, const TExpr& expr, const typename ExpressionEvaluation<TExpr>::ReturnType defaultValue)
	{
		typedef typename ExpressionEvaluation<TExpr>::ReturnType ReturnType;

		#ifdef __CUDACC__
		if(ExpressionEvaluation<TExpr>::location==DeviceSide || ExpressionEvaluation<TExpr>::location==AnySide)
		{
			TOut	*castHostPtr   = reinterpret_cast<TOut*>(hostPtr),
				*castDevicePtr = reinterpret_cast<TOut*>(devicePtr);

			const dim3	blockSize = layout.blockSize(),
					numBlocks = layout.numBlocks();

			// Compute the stride for each block, in order to limit the total number of blocks to fillNumBlocks :
			dim3	reducedNumBlocks,
				blockSteps;

			reducedNumBlocks.x	= std::min(numBlocks.x, fillNumBlocks);
			blockSteps.x		= (numBlocks.x + reducedNumBlocks.x - 1) / reducedNumBlocks.x;
			reducedNumBlocks.y	= std::min(numBlocks.y, fillNumBlocks / reducedNumBlocks.x);
			blockSteps.y		= (numBlocks.y + reducedNumBlocks.y - 1) / reducedNumBlocks.y;
			reducedNumBlocks.z	= std::min(numBlocks.z, fillNumBlocks / (reducedNumBlocks.x * reducedNumBlocks.y));
			blockSteps.z		= (numBlocks.z + reducedNumBlocks.z - 1) / reducedNumBlocks.z;
		
			const unsigned int 	totalNumBlocks = reducedNumBlocks.x * reducedNumBlocks.y * reducedNumBlocks.z,
						totalNumThreads = blockSize.x * blockSize.y * blockSize.z,
						maxPow2Half = 1 << (static_cast<int>(std::floor(std::log(totalNumThreads-1)/std::log(2))));
			const size_t sharedMemorySize = totalNumThreads * sizeof(ReturnType);

			/* // Testing the computation layouts :
			std::cout << "numBlocks        : " << numBlocks.x << ", " << numBlocks.y << ", " << numBlocks.z << std::endl;
			std::cout << "blockSize        : " << blockSize.x << ", " << blockSize.y << ", " << blockSize.z << std::endl;
			std::cout << "reducedNumBlocks : " << reducedNumBlocks.x << ", " << reducedNumBlocks.y << ", " << reducedNumBlocks.z << std::endl;
			std::cout << "blockSteps       : " << blockSteps.x << ", " << blockSteps.y << ", " << blockSteps.z << std::endl;
			std::cout << "totalNumBlocks   : " << totalNumBlocks << std::endl;
			std::cout << "totalNumThreads  : " << totalNumThreads << std::endl;
			std::cout << "maxPow2Half      : " <<  maxPow2Half << std::endl;
			*/

			// Do the single-pass reduction :
			reduceKernel<Op><<<reducedNumBlocks, blockSize, sharedMemorySize>>>(layout, blockSteps, expr, defaultValue, castDevicePtr,  totalNumThreads, maxPow2Half);

			cudaError_t err = cudaGetLastError();
			if(err!=cudaSuccess)
				throw static_cast<Exception>(CudaExceptionsOffset + err);

			// Copy back to the Host side and complete :
			err = cudaMemcpy(reinterpret_cast<void*>(castHostPtr), reinterpret_cast<void*>(castDevicePtr), totalNumBlocks*sizeof(TOut), cudaMemcpyDeviceToHost);
			if(err!=cudaSuccess)
				throw static_cast<Exception>(CudaExceptionsOffset + err);

			for(unsigned int k=1; k<totalNumBlocks; k++)
				castHostPtr[0] = Op<TOut, TOut>::apply(castHostPtr[0], castHostPtr[k]);

			// Return :
			return castHostPtr[0];
		#else
		if(ExpressionEvaluation<TExpr>::location==DeviceSide)
		{
			throw NotSupported;
		#endif
		}
		else
		{
			ReturnType result = defaultValue;

			for(index_t k=0, j=0, i=0, q=0; q<layout.numElements(); q++)
			{
				result = Op<ReturnType, ReturnType>::apply(result, ExpressionEvaluation<TExpr>::evaluate(expr, layout, i, j, k));
				layout.moveToNext(i, j, k);
			}

			return result;
		}
	}

	// Specific tools :
		/**
		\brief Compute the minimum of an expression.
		\param layout Layout of the expression.
		\param expr Expression.
		\return The minimum of an expression.
		**/
		template<typename TExpr>
		__host__ typename ExpressionEvaluation<TExpr>::ReturnType ReduceContext::min(const Layout& layout, const TExpr& expr)
		{
			typedef typename ExpressionEvaluation<TExpr>::ReturnType ReturnType;
			return reduce<BinOp_min, ReturnType>(layout, expr, std::numeric_limits<ReturnType>::max());
		}

		/**
		\brief Compute the minimum of an accessor.
		\param accessor Data accessor.
		\return The minimum of an accessor.
		**/
		template<typename T, Location l>
		__host__ T ReduceContext::min(const Accessor<T,l>& accessor)
		{
			return min(accessor.layout(), accessor);
		}

		/**
		\brief Compute the maximum of an expression.
		\param layout Layout of the expression.
		\param expr Expression.
		\return The maximum of an expression.
		**/
		template<typename TExpr>
		__host__ typename ExpressionEvaluation<TExpr>::ReturnType ReduceContext::max(const Layout& layout, const TExpr& expr)
		{
			typedef typename ExpressionEvaluation<TExpr>::ReturnType ReturnType;
			ReturnType defaultValue = std::numeric_limits<ReturnType>::is_integer ? std::numeric_limits<ReturnType>::min() : -std::numeric_limits<ReturnType>::max();
			return reduce<BinOp_max, ReturnType>(layout, expr, defaultValue);
		}

		/**
		\brief Compute the maximum of an accessor.
		\param accessor Data accessor.
		\return The maximum of an accessor.
		**/
		template<typename T, Location l>
		__host__ T ReduceContext::max(const Accessor<T,l>& accessor)
		{
			return max(accessor.layout(), accessor);
		}

		/**
		\brief Compute the sum of an expression.
		\param layout Layout of the expression.
		\param expr Expression.
		\return The sum of an expression.
		**/
		template<typename TExpr>
		__host__ typename ExpressionEvaluation<TExpr>::ReturnType ReduceContext::sum(const Layout& layout, const TExpr& expr)
		{
			typedef typename ExpressionEvaluation<TExpr>::ReturnType ReturnType;
			return reduce<BinOp_Plus, ReturnType>(layout, expr, complexCopy<ReturnType>(0));
		}

		/**
		\brief Compute the sum of an accessor.
		\param accessor Data accessor.
		\return The sum of an accessor.
		**/
		template<typename T, Location l>
		__host__ T ReduceContext::sum(const Accessor<T,l>& accessor)
		{
			return sum(accessor.layout(), accessor);
		}

		/**
		\brief Compute the arithmetic mean of an expression.
		\param layout Layout of the expression.
		\param expr Expression.
		\return The average of an expression.
		**/
		template<typename TExpr>
		__host__ typename ExpressionEvaluation<TExpr>::ReturnType ReduceContext::mean(const Layout& layout, const TExpr& expr)
		{
			return sum(layout, expr) / layout.numElements();
		}

		/**
		\brief Compute the arithmetic mean of an accessor.
		\param accessor Data accessor.
		\return The arithmetic mean of an accessor.
		**/
		template<typename T, Location l>
		__host__ T ReduceContext::mean(const Accessor<T,l>& accessor)
		{
			return sum(accessor.layout(), accessor) / accessor.numElements();
		}

		/**
		\brief Compute the product of an expression.
		\param layout Layout of the expression.
		\param expr Expression.
		\return The product of an expression.
		**/
		template<typename TExpr>
		__host__ typename ExpressionEvaluation<TExpr>::ReturnType ReduceContext::prod(const Layout& layout, const TExpr& expr)
		{
			typedef typename ExpressionEvaluation<TExpr>::ReturnType ReturnType;
			return reduce<BinOp_Times, ReturnType>(layout, expr, complexCopy<ReturnType>(1));
		}

		/**
		\brief Compute the product of an accessor.
		\param accessor Data accessor.
		\return The product of an accessor.
		**/
		template<typename T, Location l>
		__host__ T ReduceContext::prod(const Accessor<T,l>& accessor)
		{
			return prod(accessor.layout(), accessor);
		}

		/**
		\brief Compute logical and between all the elements of an expression.
		\param layout Layout of the expression.
		\param expr Expression.
		\return True if all elements are.
		**/
		template<typename TExpr>
		__host__ bool ReduceContext::all(const Layout& layout, const TExpr& expr)
		{
			return reduce<BinOp_And, bool>(layout, expr, true);
		}

		/**
		\brief Compute logical and between all the elements of an accessor.
		\param accessor Data accessor.
		\return True if all elements are.
		**/
		template<typename T, Location l>
		__host__ bool ReduceContext::all(const Accessor<T,l>& accessor)
		{
			return all(accessor.layout(), accessor);
		}

		/**
		\brief Compute logical or between all the elements of an expression.
		\param layout Layout of the expression.
		\param expr Expression.
		\return True if any element is.
		**/
		template<typename TExpr>
		__host__ bool ReduceContext::any(const Layout& layout, const TExpr& expr)
		{
			return reduce<BinOp_Or, bool>(layout, expr, false);
		}

		/**
		\brief Compute logical or between all the elements of an accessor.
		\param accessor Data accessor.
		\return True if any element is.
		**/
		template<typename T, Location l>
		__host__ bool ReduceContext::any(const Accessor<T,l>& accessor)
		{
			return any(accessor.layout(), accessor);
		}

	// Binning-like operations :
	/**
	\brief Generic block reduction operation.
	\tparam Op Reduction operator.
	\param layout Layout of the data to be reduced.
	\param expr Expression or data to be reduced.
	\param defaultValue Default value used as initialization.
	\param output Resulting data (written).

	The function assume all the blocks to be of identical sizes. It requires the output layout to be smaller than the input layout and that all output dimensions divide the input dimensions.
	The operator is a binary operator, usually defined via the library's macros.

	\throw Kartet::InvalidOperation If the input and output layouts are not congruent.
	**/
	template<template<typename,typename> class Op, typename TExpr, typename TOut, Location l>
	__host__ void ReduceContext::reduceBlock(const Layout& layout, const TExpr expr, const typename ExpressionEvaluation<TExpr>::ReturnType defaultValue, const Accessor<TOut,l>& output)
	{
		typedef typename ExpressionEvaluation<TExpr>::ReturnType ReturnType;
		StaticAssert<ExpressionEvaluation<TExpr>::location==l || ExpressionEvaluation<TExpr>::location==AnySide>(); // The expression must be on the same side than the output.

		if((layout.numRows() % output.numRows())!=0 || (layout.numColumns() % output.numColumns())!=0 || (layout.numSlices() % output.numSlices())!=0)	
			throw InvalidOperation;

		#ifdef __CUDACC__
		if(ExpressionEvaluation<TExpr>::location==DeviceSide || ExpressionEvaluation<TExpr>::location==AnySide)
		{
			dim3	blockSize = layout.blockSize(),
				numBlocks = layout.numBlocks(),
				blockSteps,				// The steps to cover one block.
				numSubReductionBlocks;			// The number of sub-blocks covered.
			const Layout reductionBlockLayout(layout.numRows()/output.numRows(), layout.numColumns()/output.numColumns(), layout.numSlices()/output.numSlices());

			/* The approach here is to split the code in two different methods depending on the size of the block to be reduced.
			   The chosen limit is still somewhat arbitrary and needs to be refined.
			*/
			//std::cout << "Reduction from layout " << layout << " to " << output.layout() << std::endl;
			bool largeReductionMode = (reductionBlockLayout.numElements()>=(Layout::StaticContainer<void>::numThreads/2)); // Ad-hoc coefficient decision.
			if(largeReductionMode)
			{
				//std::cout << "Large reduction mode  : " << reductionBlockLayout << std::endl;

				// Cut to the block size layout :
				blockSize.x = std::min(static_cast<index_t>(Layout::StaticContainer<void>::numThreads), reductionBlockLayout.numRows());
				blockSize.y = std::min(static_cast<index_t>(Layout::StaticContainer<void>::numThreads/blockSize.x), reductionBlockLayout.numColumns());
				blockSize.z = std::min(static_cast<index_t>(Layout::StaticContainer<void>::numThreads/(blockSize.x*blockSize.y)), reductionBlockLayout.numSlices());
			
				// The number of steps to do depends on the previous cut :
				blockSteps.x = (reductionBlockLayout.numRows() + blockSize.x - 1) / blockSize.x;
				blockSteps.y = (reductionBlockLayout.numColumns() + blockSize.y - 1) / blockSize.y;
				blockSteps.z = (reductionBlockLayout.numSlices() + blockSize.z- 1) / blockSize.z;
			
				// There will be exactly one block per output values :
				numBlocks.x = output.numRows();
				numBlocks.y = output.numColumns();
				numBlocks.z = output.numSlices();

				// Each Cuda Block will only take care of one block :
				numSubReductionBlocks.x = 1;
				numSubReductionBlocks.y = 1;
				numSubReductionBlocks.z = 1;
			}
			else
			{
				//std::cout << "Small reduction mode  : " << reductionBlockLayout << std::endl;

				// Cut the block size to fit an integer number of blocks :
				blockSize.x = std::max( static_cast<index_t>(std::max(blockSize.x - blockSize.x % reductionBlockLayout.numRows(), reductionBlockLayout.numRows() % blockSize.x)), static_cast<index_t>(1));
				blockSize.y = std::max( static_cast<index_t>(std::max(blockSize.y - blockSize.y % reductionBlockLayout.numColumns(), reductionBlockLayout.numColumns() % blockSize.y)), static_cast<index_t>(1));
				blockSize.z = std::max( static_cast<index_t>(std::max(blockSize.z - blockSize.z % reductionBlockLayout.numSlices(), reductionBlockLayout.numSlices() % blockSize.z)), static_cast<index_t>(1));
			
				// The number of steps to do also depends on the previous cut :
				blockSteps.x = (reductionBlockLayout.numRows() + blockSize.x - 1) / blockSize.x;
				blockSteps.y = (reductionBlockLayout.numColumns() + blockSize.y - 1) / blockSize.y;
				blockSteps.z = (reductionBlockLayout.numSlices() + blockSize.z - 1) / blockSize.z;
			
				// numBlocks
				numBlocks.x = (output.numRows() + (blockSize.x * blockSteps.x / reductionBlockLayout.numRows()) - 1) / (blockSize.x * blockSteps.x / reductionBlockLayout.numRows());
				numBlocks.y = (output.numColumns() + (blockSize.y * blockSteps.y / reductionBlockLayout.numColumns()) - 1) / (blockSize.y * blockSteps.y / reductionBlockLayout.numColumns());
				numBlocks.z = (output.numSlices() + (blockSize.z * blockSteps.z / reductionBlockLayout.numSlices()) - 1) / (blockSize.z * blockSteps.z / reductionBlockLayout.numSlices());
			
				// Each Cuda Block will take care of multiple blocks :
				numSubReductionBlocks.x = blockSize.x * blockSteps.x / reductionBlockLayout.numRows();
				numSubReductionBlocks.y = blockSize.y * blockSteps.y / reductionBlockLayout.numColumns();
				numSubReductionBlocks.z = blockSize.z * blockSteps.z / reductionBlockLayout.numSlices();
			}

			// Testing the computation layouts :
			/*std::cout << "numBlocks             : " << numBlocks.x << ", " << numBlocks.y << ", " << numBlocks.z << std::endl;
			std::cout << "blockSize             : " << blockSize.x << ", " << blockSize.y << ", " << blockSize.z << std::endl;
			std::cout << "blockSteps            : " << blockSteps.x << ", " << blockSteps.y << ", " << blockSteps.z << std::endl;
			std::cout << "numSubReductionBlocks : " << numSubReductionBlocks.x << ", " << numSubReductionBlocks.y << ", " << numSubReductionBlocks.z << std::endl;*/

			const unsigned int 	totalNumThreads = blockSize.x * blockSize.y * blockSize.z,
						maxPow2Half = 1 << (static_cast<int>(std::floor(std::log(totalNumThreads-1)/std::log(2))));
			const size_t sharedMemorySize = totalNumThreads * sizeof(ReturnType);

			/*std::cout << "totalNumThreads       : " << totalNumThreads << std::endl;
			std::cout << "totalNumBlocks        : " << (numBlocks.x * numBlocks.y * numBlocks.z) << std::endl;
			std::cout << "sharedMemorySize      : " << sharedMemorySize << " bytes" << std::endl;*/

			// Do the single-pass reduction :
			if(largeReductionMode)
				reduceToLayoutKernel_LargeReductionMode<Op><<<numBlocks, blockSize, sharedMemorySize>>>(layout, reductionBlockLayout, blockSteps, expr, defaultValue, output, totalNumThreads, maxPow2Half);
			else
				reduceToLayoutKernel_SmallReductionMode<Op><<<numBlocks, blockSize, sharedMemorySize>>>(layout, reductionBlockLayout, blockSteps, numSubReductionBlocks, expr, defaultValue, output);

			cudaError_t err = cudaGetLastError();
			if(err!=cudaSuccess)
				throw static_cast<Exception>(CudaExceptionsOffset + err);
		#else
		if(ExpressionEvaluation<TExpr>::location==DeviceSide)
		{
			throw NotSupported;
		#endif
		}
		else
		{
			const Layout block(layout.numRows()/output.numRows(), layout.numColumns()/output.numColumns(), layout.numSlices()/output.numSlices());

			for(index_t ko=0, jo=0, io=0, qo=0; qo<output.numElements(); qo++)
			{
				ReturnType result = defaultValue;

				for(index_t kb=0, jb=0, ib=0, qb=0; qb<block.numElements(); qb++)
				{
					index_t	i = io * block.numRows() + ib,
						j = jo * block.numColumns() + jb,
						k = ko * block.numSlices() + kb;

					result = Op<ReturnType, ReturnType>::apply(result, ExpressionEvaluation<TExpr>::evaluate(expr, layout, i, j, k));

					block.moveToNext(ib, jb, kb);
				}
				
				output.data(io, jo, ko) = complexCopy<TOut>(result);

				output.moveToNext(io, jo, ko);
			}
		}
	}

	// Specific tools :
		/**
		\brief Compute the minimum per block of an expression.
		\param layout Layout of the expression.
		\param expr Expression.
		\param output Resulting data (written).
		\throw Kartet::InvalidOperation If the input and output layouts are not congruent.
		**/
		template<typename TExpr, typename TOut, Location l>
		__host__ void ReduceContext::minBlock(const Layout& layout, const TExpr& expr, const Accessor<TOut,l>& output)
		{
			typedef typename ExpressionEvaluation<TExpr>::ReturnType ReturnType;
			reduceBlock<BinOp_min>(layout, expr, std::numeric_limits<ReturnType>::max(), output);
		}

		/**
		\brief Compute the minimum per block of an accessor.
		\param accessor Input data.
		\param output Resulting data (written).
		\throw Kartet::InvalidOperation If the input and output layouts are not congruent.
		**/
		template<typename T, typename TOut, Location l>
		__host__ void ReduceContext::minBlock(const Accessor<T,l>& accessor, const Accessor<TOut,l>& output)
		{
			minBlock(accessor.layout(), accessor, output);
		}

		/**
		\brief Compute the maximum per block of an expression.
		\param layout Layout of the expression.
		\param expr Expression.
		\param output Resulting data (written).
		\throw Kartet::InvalidOperation If the input and output layouts are not congruent.
		**/
		template<typename TExpr, typename TOut, Location l>
		__host__ void ReduceContext::maxBlock(const Layout& layout, const TExpr& expr, const Accessor<TOut,l>& output)
		{
			typedef typename ExpressionEvaluation<TExpr>::ReturnType ReturnType;
			ReturnType defaultValue = std::numeric_limits<ReturnType>::is_integer ? std::numeric_limits<ReturnType>::min() : -std::numeric_limits<ReturnType>::max();
			reduceBlock<BinOp_max>(layout, expr, defaultValue, output);
		}

		/**
		\brief Compute the maximum per block of an accessor.
		\param accessor Input data.
		\param output Resulting data (written).
		\throw Kartet::InvalidOperation If the input and output layouts are not congruent.
		**/
		template<typename T, typename TOut, Location l>
		__host__ void ReduceContext::maxBlock(const Accessor<T,l>& accessor, const Accessor<TOut,l>& output)
		{
			maxBlock(accessor.layout(), accessor, output);
		}

		/**
		\brief Compute the sum per block of an expression.
		\param layout Layout of the expression.
		\param expr Expression.
		\param output Resulting data (written).
		\throw Kartet::InvalidOperation If the input and output layouts are not congruent.
		**/
		template<typename TExpr, typename TOut, Location l>
		__host__ void ReduceContext::sumBlock(const Layout& layout, const TExpr& expr, const Accessor<TOut,l>& output)
		{
			typedef typename ExpressionEvaluation<TExpr>::ReturnType ReturnType;
			reduceBlock<BinOp_Plus>(layout, expr, complexCopy<ReturnType>(0), output);
		}

		/**
		\brief Compute the sum per block of an accessor.
		\param accessor Input data.
		\param output Resulting data (written).
		\throw Kartet::InvalidOperation If the input and output layouts are not congruent.
		**/
		template<typename T, typename TOut, Location l>
		__host__ void ReduceContext::sumBlock(const Accessor<T,l>& accessor, const Accessor<TOut,l>& output)
		{
			sumBlock(accessor.layout(), accessor, output);
		}

		/**
		\brief Compute the product per block of an expression.
		\param layout Layout of the expression.
		\param expr Expression.
		\param output Resulting data (written).
		\throw Kartet::InvalidOperation If the input and output layouts are not congruent.
		**/
		template<typename TExpr, typename TOut, Location l>
		__host__ void ReduceContext::prodBlock(const Layout& layout, const TExpr& expr, const Accessor<TOut,l>& output)
		{
			typedef typename ExpressionEvaluation<TExpr>::ReturnType ReturnType;
			reduceBlock<BinOp_Times>(layout, expr, complexCopy<ReturnType>(1), output);
		}

		/**
		\brief Compute the product per block of an accessor.
		\param accessor Input data.
		\param output Resulting data (written).
		\throw Kartet::InvalidOperation If the input and output layouts are not congruent.
		**/
		template<typename T, typename TOut, Location l>
		__host__ void ReduceContext::prodBlock(const Accessor<T,l>& accessor, const Accessor<TOut,l>& output)
		{
			prodBlock(accessor.layout(), accessor, output);
		}

		/**
		\brief Compute the logical and between all elements in all the blocks of an expression.
		\param layout Layout of the expression.
		\param expr Expression.
		\param output Resulting data (written).
		\throw Kartet::InvalidOperation If the input and output layouts are not congruent.
		**/
		template<typename TExpr, typename TOut, Location l>
		__host__ void ReduceContext::allBlock(const Layout& layout, const TExpr& expr, const Accessor<TOut,l>& output)
		{
			reduceBlock<BinOp_And>(layout, expr, true, output);
		}

		/**
		\brief Compute the logical and between all elements in all the blocks of an accessor.
		\param accessor Input data.
		\param output Resulting data (written).
		\throw Kartet::InvalidOperation If the input and output layouts are not congruent.
		**/
		template<typename T, typename TOut, Location l>
		__host__ void ReduceContext::allBlock(const Accessor<T,l>& accessor, const Accessor<TOut,l>& output)
		{
			allBlock(accessor.layout(), accessor, output);
		}

		/**
		\brief Compute the logical or between all elements in all the blocks of an expression.
		\param layout Layout of the expression.
		\param expr Expression.
		\param output Resulting data (written).
		\throw Kartet::InvalidOperation If the input and output layouts are not congruent.
		**/
		template<typename TExpr, typename TOut, Location l>
		__host__ void ReduceContext::anyBlock(const Layout& layout, const TExpr& expr, const Accessor<TOut,l>& output)
		{
			reduceBlock<BinOp_Or>(layout, expr, false, output);
		}

		/**
		\brief Compute the logical or between all elements in all the blocks of an accessor.
		\param accessor Input data.
		\param output Resulting data (written).
		\throw Kartet::InvalidOperation If the input and output layouts are not congruent.
		**/
		template<typename T, typename TOut, Location l>
		__host__ void ReduceContext::anyBlock(const Accessor<T,l>& accessor, const Accessor<TOut,l>& output)
		{
			anyBlock(accessor.layout(), accessor, output);
		}
} // namespace Kartet

#endif

