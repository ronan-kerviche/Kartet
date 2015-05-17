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

#ifndef __KARTET_EXCEPTIONS__
#define __KARTET_EXCEPTIONS__

// Includes :
	#ifdef __CUDACC__
		#include <cublas_v2.h>
		#include <curand.h>
		#include <cufft.h>

		#ifndef __CUDA_API_VERSION__
			#define __CUDA_API_VERSION__ 5000
		#endif
	#endif
	#include "Core/LibTools.hpp"

namespace Kartet
{
	enum Exception
	{	
		// Cuda Specifics :
		// cudaSuccess, see NoExceptions
		#define DEFINE_CUDA_EXCEPTION( x ) C##x = CudaExceptionsOffset + c##x	
		CudaExceptionsOffset	= 1024,
		#ifdef __CUDACC__
		DEFINE_CUDA_EXCEPTION( udaErrorMissingConfiguration ), 
		DEFINE_CUDA_EXCEPTION( udaErrorMemoryAllocation ), 
		DEFINE_CUDA_EXCEPTION( udaErrorInitializationError ), 
		DEFINE_CUDA_EXCEPTION( udaErrorLaunchFailure ),	
		DEFINE_CUDA_EXCEPTION( udaErrorPriorLaunchFailure ), 
		DEFINE_CUDA_EXCEPTION( udaErrorLaunchTimeout ), 
		DEFINE_CUDA_EXCEPTION( udaErrorLaunchOutOfResources ), 
		DEFINE_CUDA_EXCEPTION( udaErrorInvalidDeviceFunction ), 
		DEFINE_CUDA_EXCEPTION( udaErrorInvalidConfiguration ), 
		DEFINE_CUDA_EXCEPTION( udaErrorInvalidDevice ), 
		DEFINE_CUDA_EXCEPTION( udaErrorInvalidValue ), 
		DEFINE_CUDA_EXCEPTION( udaErrorInvalidPitchValue ), 
		DEFINE_CUDA_EXCEPTION( udaErrorInvalidSymbol ), 
		DEFINE_CUDA_EXCEPTION( udaErrorMapBufferObjectFailed ), 
		DEFINE_CUDA_EXCEPTION( udaErrorUnmapBufferObjectFailed ), 
		DEFINE_CUDA_EXCEPTION( udaErrorInvalidHostPointer ), 
		DEFINE_CUDA_EXCEPTION( udaErrorInvalidDevicePointer ), 
		DEFINE_CUDA_EXCEPTION( udaErrorInvalidTexture ), 
		DEFINE_CUDA_EXCEPTION( udaErrorInvalidTextureBinding ), 
		DEFINE_CUDA_EXCEPTION( udaErrorInvalidChannelDescriptor ), 
		DEFINE_CUDA_EXCEPTION( udaErrorInvalidMemcpyDirection ), 
		DEFINE_CUDA_EXCEPTION( udaErrorAddressOfConstant ), 
		DEFINE_CUDA_EXCEPTION( udaErrorTextureFetchFailed ), 
		DEFINE_CUDA_EXCEPTION( udaErrorTextureNotBound ), 
		DEFINE_CUDA_EXCEPTION( udaErrorSynchronizationError ), 
		DEFINE_CUDA_EXCEPTION( udaErrorInvalidFilterSetting ), 
		DEFINE_CUDA_EXCEPTION( udaErrorInvalidNormSetting ), 
		DEFINE_CUDA_EXCEPTION( udaErrorMixedDeviceExecution ), 
		DEFINE_CUDA_EXCEPTION( udaErrorCudartUnloading ), 
		DEFINE_CUDA_EXCEPTION( udaErrorUnknown ), 
		DEFINE_CUDA_EXCEPTION( udaErrorNotYetImplemented ), 
		DEFINE_CUDA_EXCEPTION( udaErrorMemoryValueTooLarge ), 
		DEFINE_CUDA_EXCEPTION( udaErrorInvalidResourceHandle ), 
		DEFINE_CUDA_EXCEPTION( udaErrorNotReady ), 
		DEFINE_CUDA_EXCEPTION( udaErrorInsufficientDriver ), 
		DEFINE_CUDA_EXCEPTION( udaErrorSetOnActiveProcess ), 
		DEFINE_CUDA_EXCEPTION( udaErrorInvalidSurface ), 
		DEFINE_CUDA_EXCEPTION( udaErrorNoDevice ), 
		DEFINE_CUDA_EXCEPTION( udaErrorECCUncorrectable ), 
		DEFINE_CUDA_EXCEPTION( udaErrorSharedObjectSymbolNotFound ), 
		DEFINE_CUDA_EXCEPTION( udaErrorSharedObjectInitFailed ), 
		DEFINE_CUDA_EXCEPTION( udaErrorUnsupportedLimit ), 
		DEFINE_CUDA_EXCEPTION( udaErrorDuplicateVariableName ), 
		DEFINE_CUDA_EXCEPTION( udaErrorDuplicateTextureName ), 
		DEFINE_CUDA_EXCEPTION( udaErrorDuplicateSurfaceName ), 
		DEFINE_CUDA_EXCEPTION( udaErrorDevicesUnavailable ), 
		DEFINE_CUDA_EXCEPTION( udaErrorInvalidKernelImage ), 
		DEFINE_CUDA_EXCEPTION( udaErrorNoKernelImageForDevice ), 
		DEFINE_CUDA_EXCEPTION( udaErrorIncompatibleDriverContext ), 
		DEFINE_CUDA_EXCEPTION( udaErrorStartupFailure ), 
		DEFINE_CUDA_EXCEPTION( udaErrorApiFailureBase ), 
		DEFINE_CUDA_EXCEPTION( udaErrorPeerAccessAlreadyEnabled ), 
		DEFINE_CUDA_EXCEPTION( udaErrorPeerAccessNotEnabled ), 
		DEFINE_CUDA_EXCEPTION( udaErrorDeviceAlreadyInUse ), 
		DEFINE_CUDA_EXCEPTION( udaErrorProfilerDisabled ), 
		DEFINE_CUDA_EXCEPTION( udaErrorProfilerNotInitialized ), 
		DEFINE_CUDA_EXCEPTION( udaErrorProfilerAlreadyStarted ), 
		DEFINE_CUDA_EXCEPTION( udaErrorProfilerAlreadyStopped ), 
		DEFINE_CUDA_EXCEPTION( udaErrorAssert ), 
		DEFINE_CUDA_EXCEPTION( udaErrorTooManyPeers ), 
		DEFINE_CUDA_EXCEPTION( udaErrorHostMemoryAlreadyRegistered ), 
		DEFINE_CUDA_EXCEPTION( udaErrorHostMemoryNotRegistered ), 
		DEFINE_CUDA_EXCEPTION( udaErrorOperatingSystem ), 
		DEFINE_CUDA_EXCEPTION( udaErrorPeerAccessUnsupported ), 
		DEFINE_CUDA_EXCEPTION( udaErrorLaunchMaxDepthExceeded ), 
		DEFINE_CUDA_EXCEPTION( udaErrorLaunchFileScopedTex ), 
		DEFINE_CUDA_EXCEPTION( udaErrorLaunchFileScopedSurf ), 
		DEFINE_CUDA_EXCEPTION( udaErrorSyncDepthExceeded ), 
		DEFINE_CUDA_EXCEPTION( udaErrorLaunchPendingCountExceeded ), 
		DEFINE_CUDA_EXCEPTION( udaErrorNotPermitted ), 
		DEFINE_CUDA_EXCEPTION( udaErrorNotSupported ),
		#if __CUDA_API_VERSION__ > 5000
		DEFINE_CUDA_EXCEPTION( udaErrorHardwareStackError ),
		DEFINE_CUDA_EXCEPTION( udaErrorIllegalInstruction ),
		DEFINE_CUDA_EXCEPTION( udaErrorMisalignedAddress ),
		DEFINE_CUDA_EXCEPTION( udaErrorInvalidAddressSpace ),
		DEFINE_CUDA_EXCEPTION( udaErrorInvalidPc ),
		DEFINE_CUDA_EXCEPTION( udaErrorIllegalAddress ),
		DEFINE_CUDA_EXCEPTION( udaErrorStartupFailure ),
		DEFINE_CUDA_EXCEPTION( udaErrorApiFailureBase ),
		#endif
		#endif
		// CuBLAS Specifics :
		CuBLASExceptionOffset		= 2048,
		#ifdef __CUDACC__
		CuBLASNotInitialized		= CuBLASExceptionOffset + CUBLAS_STATUS_NOT_INITIALIZED,
		CuBLASAllocFailed		= CuBLASExceptionOffset + CUBLAS_STATUS_ALLOC_FAILED,
		CuBLASInvalidValue		= CuBLASExceptionOffset + CUBLAS_STATUS_INVALID_VALUE,
		CuBLASArchMismatch		= CuBLASExceptionOffset + CUBLAS_STATUS_ARCH_MISMATCH,
		CuBLASMappingError		= CuBLASExceptionOffset + CUBLAS_STATUS_MAPPING_ERROR,
		CuBLASExecutionFailed		= CuBLASExceptionOffset + CUBLAS_STATUS_EXECUTION_FAILED,
		CuBLASInternalError		= CuBLASExceptionOffset + CUBLAS_STATUS_INTERNAL_ERROR,
		#endif
		// CuRand Specifics :
		CuRandExceptionOffset		= 3072,
		#ifdef __CUDACC__
		CuRandVersionMismatch		= CuRandExceptionOffset + CURAND_STATUS_VERSION_MISMATCH,
		CuRandNotInitialized		= CuRandExceptionOffset + CURAND_STATUS_NOT_INITIALIZED,
		CuRandAllocationFailed		= CuRandExceptionOffset + CURAND_STATUS_ALLOCATION_FAILED,
		CuRandTypeError			= CuRandExceptionOffset + CURAND_STATUS_TYPE_ERROR,
		CuRandOutOfRange		= CuRandExceptionOffset + CURAND_STATUS_OUT_OF_RANGE,
		CuRandLengthNotMultiple		= CuRandExceptionOffset + CURAND_STATUS_LENGTH_NOT_MULTIPLE,
		CuRandDoublePrecisionRequired	= CuRandExceptionOffset + CURAND_STATUS_DOUBLE_PRECISION_REQUIRED,
		CuRandLaunchFailure		= CuRandExceptionOffset + CURAND_STATUS_LAUNCH_FAILURE,
		CuRandPreexistingFailure	= CuRandExceptionOffset + CURAND_STATUS_PREEXISTING_FAILURE,
		CuRandInitializationFailed	= CuRandExceptionOffset + CURAND_STATUS_INITIALIZATION_FAILED,
		CuRandArchMismatch		= CuRandExceptionOffset + CURAND_STATUS_ARCH_MISMATCH,
		CuRandInternalError		= CuRandExceptionOffset + CURAND_STATUS_INTERNAL_ERROR,
		#endif
		// CuFFT Specifics :
		CuFFTExceptionOffset		= 4096,
		#ifdef __CUDACC__
		CuFFTInvalidPlan		= CuFFTExceptionOffset + CUFFT_INVALID_PLAN,
		CuFFTAllocFailed		= CuFFTExceptionOffset + CUFFT_ALLOC_FAILED,
		CuFFTInvalidType		= CuFFTExceptionOffset + CUFFT_INVALID_TYPE,
		CuFFTInvalidValue		= CuFFTExceptionOffset + CUFFT_INVALID_VALUE,
		CuFFTInternalError		= CuFFTExceptionOffset + CUFFT_INTERNAL_ERROR,
		CuFFTExecFailed			= CuFFTExceptionOffset + CUFFT_EXEC_FAILED,
		CuFFTSetupFailed		= CuFFTExceptionOffset + CUFFT_SETUP_FAILED,
		CuFFTInvalidSize		= CuFFTExceptionOffset + CUFFT_INVALID_SIZE,
		CuFFTUnalignedData		= CuFFTExceptionOffset + CUFFT_UNALIGNED_DATA,
		#endif
		#undef DEFINE_CUDA_EXCEPTION
		// Kartet Specifics :
		InvalidNegativeSize,
		InvalidNegativeStep,
		OutOfRange,
		OutOfMemory,
		InvalidOperation,
		IncompatibleLayout,
		InvalidLayoutChange,
		InvalidFileStream,
		InvalidFileHeader,
		UnknownTypeIndex,
		InvalidBLASContext,
		InvalidCuRandContext,
		InvalidCuFFTContext,
		InvalidContext,
		InvalidLocation,
		NullPointer,
		NotSupported,
		// Others :
		NoException = 0,
	};
} // namespace Kartet

#include <iostream>

	inline std::ostream& operator<<(std::ostream& os, const Kartet::Exception& e)
	{
		switch(e)
		{
			#define EXCEPTION_MESSAGE(a) case Kartet::a : os << #a; break;	
			// Cuda :
			#ifdef __CUDACC__
			EXCEPTION_MESSAGE( CudaErrorMissingConfiguration ) 
			EXCEPTION_MESSAGE( CudaErrorMemoryAllocation ) 
			EXCEPTION_MESSAGE( CudaErrorInitializationError ) 
			EXCEPTION_MESSAGE( CudaErrorLaunchFailure )
			EXCEPTION_MESSAGE( CudaErrorPriorLaunchFailure ) 
			EXCEPTION_MESSAGE( CudaErrorLaunchTimeout ) 
			EXCEPTION_MESSAGE( CudaErrorLaunchOutOfResources ) 
			EXCEPTION_MESSAGE( CudaErrorInvalidDeviceFunction ) 
			EXCEPTION_MESSAGE( CudaErrorInvalidConfiguration ) 
			EXCEPTION_MESSAGE( CudaErrorInvalidDevice ) 
			EXCEPTION_MESSAGE( CudaErrorInvalidValue ) 
			EXCEPTION_MESSAGE( CudaErrorInvalidPitchValue ) 
			EXCEPTION_MESSAGE( CudaErrorInvalidSymbol ) 
			EXCEPTION_MESSAGE( CudaErrorMapBufferObjectFailed ) 
			EXCEPTION_MESSAGE( CudaErrorUnmapBufferObjectFailed ) 
			EXCEPTION_MESSAGE( CudaErrorInvalidHostPointer ) 
			EXCEPTION_MESSAGE( CudaErrorInvalidDevicePointer ) 
			EXCEPTION_MESSAGE( CudaErrorInvalidTexture ) 
			EXCEPTION_MESSAGE( CudaErrorInvalidTextureBinding ) 
			EXCEPTION_MESSAGE( CudaErrorInvalidChannelDescriptor ) 
			EXCEPTION_MESSAGE( CudaErrorInvalidMemcpyDirection ) 
			EXCEPTION_MESSAGE( CudaErrorAddressOfConstant ) 
			EXCEPTION_MESSAGE( CudaErrorTextureFetchFailed ) 
			EXCEPTION_MESSAGE( CudaErrorTextureNotBound ) 
			EXCEPTION_MESSAGE( CudaErrorSynchronizationError ) 
			EXCEPTION_MESSAGE( CudaErrorInvalidFilterSetting ) 
			EXCEPTION_MESSAGE( CudaErrorInvalidNormSetting ) 
			EXCEPTION_MESSAGE( CudaErrorMixedDeviceExecution ) 
			EXCEPTION_MESSAGE( CudaErrorCudartUnloading ) 
			EXCEPTION_MESSAGE( CudaErrorUnknown ) 
			EXCEPTION_MESSAGE( CudaErrorNotYetImplemented ) 
			EXCEPTION_MESSAGE( CudaErrorMemoryValueTooLarge ) 
			EXCEPTION_MESSAGE( CudaErrorInvalidResourceHandle ) 
			EXCEPTION_MESSAGE( CudaErrorNotReady ) 
			EXCEPTION_MESSAGE( CudaErrorInsufficientDriver ) 
			EXCEPTION_MESSAGE( CudaErrorSetOnActiveProcess ) 
			EXCEPTION_MESSAGE( CudaErrorInvalidSurface ) 
			EXCEPTION_MESSAGE( CudaErrorNoDevice ) 
			EXCEPTION_MESSAGE( CudaErrorECCUncorrectable ) 
			EXCEPTION_MESSAGE( CudaErrorSharedObjectSymbolNotFound ) 
			EXCEPTION_MESSAGE( CudaErrorSharedObjectInitFailed ) 
			EXCEPTION_MESSAGE( CudaErrorUnsupportedLimit ) 
			EXCEPTION_MESSAGE( CudaErrorDuplicateVariableName ) 
			EXCEPTION_MESSAGE( CudaErrorDuplicateTextureName ) 
			EXCEPTION_MESSAGE( CudaErrorDuplicateSurfaceName ) 
			EXCEPTION_MESSAGE( CudaErrorDevicesUnavailable ) 
			EXCEPTION_MESSAGE( CudaErrorInvalidKernelImage ) 
			EXCEPTION_MESSAGE( CudaErrorNoKernelImageForDevice ) 
			EXCEPTION_MESSAGE( CudaErrorIncompatibleDriverContext ) 
			EXCEPTION_MESSAGE( CudaErrorStartupFailure ) 
			EXCEPTION_MESSAGE( CudaErrorApiFailureBase ) 
			EXCEPTION_MESSAGE( CudaErrorPeerAccessAlreadyEnabled ) 
			EXCEPTION_MESSAGE( CudaErrorPeerAccessNotEnabled ) 
			EXCEPTION_MESSAGE( CudaErrorDeviceAlreadyInUse ) 
			EXCEPTION_MESSAGE( CudaErrorProfilerDisabled ) 
			EXCEPTION_MESSAGE( CudaErrorProfilerNotInitialized ) 
			EXCEPTION_MESSAGE( CudaErrorProfilerAlreadyStarted ) 
			EXCEPTION_MESSAGE( CudaErrorProfilerAlreadyStopped ) 
			EXCEPTION_MESSAGE( CudaErrorAssert ) 
			EXCEPTION_MESSAGE( CudaErrorTooManyPeers ) 
			EXCEPTION_MESSAGE( CudaErrorHostMemoryAlreadyRegistered ) 
			EXCEPTION_MESSAGE( CudaErrorHostMemoryNotRegistered ) 
			EXCEPTION_MESSAGE( CudaErrorOperatingSystem ) 
			EXCEPTION_MESSAGE( CudaErrorPeerAccessUnsupported ) 
			EXCEPTION_MESSAGE( CudaErrorLaunchMaxDepthExceeded ) 
			EXCEPTION_MESSAGE( CudaErrorLaunchFileScopedTex ) 
			EXCEPTION_MESSAGE( CudaErrorLaunchFileScopedSurf ) 
			EXCEPTION_MESSAGE( CudaErrorSyncDepthExceeded ) 
			EXCEPTION_MESSAGE( CudaErrorLaunchPendingCountExceeded ) 
			EXCEPTION_MESSAGE( CudaErrorNotPermitted ) 
			EXCEPTION_MESSAGE( CudaErrorNotSupported )
			#if __CUDA_API_VERSION__ > 5000
			EXCEPTION_MESSAGE( CudaErrorHardwareStackError )
			EXCEPTION_MESSAGE( CudaErrorIllegalInstruction )
			EXCEPTION_MESSAGE( CudaErrorMisalignedAddress )
			EXCEPTION_MESSAGE( CudaErrorInvalidAddressSpace )
			EXCEPTION_MESSAGE( CudaErrorInvalidPc )
			EXCEPTION_MESSAGE( CudaErrorIllegalAddress )
			EXCEPTION_MESSAGE( CudaErrorStartupFailure )
			EXCEPTION_MESSAGE( CudaErrorApiFailureBase )
			#endif
			// CuBLAS :
			EXCEPTION_MESSAGE( CuBLASNotInitialized )	
			EXCEPTION_MESSAGE( CuBLASAllocFailed )
			EXCEPTION_MESSAGE( CuBLASInvalidValue )
			EXCEPTION_MESSAGE( CuBLASArchMismatch )
			EXCEPTION_MESSAGE( CuBLASMappingError )
			EXCEPTION_MESSAGE( CuBLASExecutionFailed )
			EXCEPTION_MESSAGE( CuBLASInternalError )
			// CuRAND :
			EXCEPTION_MESSAGE( CuRandVersionMismatch )
			EXCEPTION_MESSAGE( CuRandNotInitialized )
			EXCEPTION_MESSAGE( CuRandAllocationFailed )
			EXCEPTION_MESSAGE( CuRandTypeError )
			EXCEPTION_MESSAGE( CuRandOutOfRange )
			EXCEPTION_MESSAGE( CuRandLengthNotMultiple )
			EXCEPTION_MESSAGE( CuRandDoublePrecisionRequired )
			EXCEPTION_MESSAGE( CuRandLaunchFailure )
			EXCEPTION_MESSAGE( CuRandPreexistingFailure )
			EXCEPTION_MESSAGE( CuRandInitializationFailed )
			EXCEPTION_MESSAGE( CuRandArchMismatch )
			EXCEPTION_MESSAGE( CuRandInternalError )
			// CuFFT :
			EXCEPTION_MESSAGE( CuFFTInvalidPlan )
			EXCEPTION_MESSAGE( CuFFTAllocFailed )
			EXCEPTION_MESSAGE( CuFFTInvalidType )
			EXCEPTION_MESSAGE( CuFFTInvalidValue )
			EXCEPTION_MESSAGE( CuFFTInternalError )
			EXCEPTION_MESSAGE( CuFFTExecFailed )
			EXCEPTION_MESSAGE( CuFFTSetupFailed )
			EXCEPTION_MESSAGE( CuFFTInvalidSize )
			EXCEPTION_MESSAGE( CuFFTUnalignedData )
			#endif
			// Kartet : 
			EXCEPTION_MESSAGE( InvalidNegativeSize )
			EXCEPTION_MESSAGE( InvalidNegativeStep )
			EXCEPTION_MESSAGE( OutOfRange )
			EXCEPTION_MESSAGE( OutOfMemory )
			EXCEPTION_MESSAGE( InvalidOperation )
			EXCEPTION_MESSAGE( IncompatibleLayout )
			EXCEPTION_MESSAGE( InvalidLayoutChange )
			EXCEPTION_MESSAGE( InvalidFileStream )
			EXCEPTION_MESSAGE( InvalidFileHeader )
			EXCEPTION_MESSAGE( UnknownTypeIndex )
			EXCEPTION_MESSAGE( InvalidBLASContext )
			EXCEPTION_MESSAGE( InvalidCuRandContext )
			EXCEPTION_MESSAGE( InvalidContext )
			EXCEPTION_MESSAGE( InvalidLocation )
			EXCEPTION_MESSAGE( NullPointer )
			EXCEPTION_MESSAGE( NotSupported )
			EXCEPTION_MESSAGE( NoException )
			#undef EXCEPTION_MESSAGE
			case Kartet::CudaExceptionsOffset :
			case Kartet::CuBLASExceptionOffset :
			case Kartet::CuRandExceptionOffset :
			case Kartet::CuFFTExceptionOffset :
				os << "NoException(ExceptionOffset)";
				break;
			default :
				os << "UnknownException";
		}

		return os;
	}

#endif

