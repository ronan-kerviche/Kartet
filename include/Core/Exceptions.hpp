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

namespace Kartet
{
	enum Exception
	{	
		// Cuda Specifics :
		// cudaSuccess, see NoExceptions
		#define DEFINE_CUDA_EXCEPTION( x ) C##x = c##x	 	
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
		//DEFINE_CUDA_EXCEPTION( udaErrorHardwareStackError ), 
		//DEFINE_CUDA_EXCEPTION( udaErrorIllegalInstruction ), 
		//DEFINE_CUDA_EXCEPTION( udaErrorMisalignedAddress ), 
		//DEFINE_CUDA_EXCEPTION( udaErrorInvalidPc ), 
		//DEFINE_CUDA_EXCEPTION( udaErrorIllegalAddress ), 
		//DEFINE_CUDA_EXCEPTION( udaErrorInvalidPtx ), 
		//DEFINE_CUDA_EXCEPTION( udaErrorInvalidGraphicsContext ), 
		#undef DEFINE_CUDA_EXCEPTION
		// Kartet Specifics :
		InvalidNegativeSize,
		InvalidNegativeStep,
		OutOfRange,
		OutOfMemory,
		InvalidOperation,
		InvalidLayoutChange,
		NullPointer,
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
			EXCEPTION_MESSAGE( InvalidNegativeSize )
			EXCEPTION_MESSAGE( InvalidNegativeStep )
			EXCEPTION_MESSAGE( OutOfRange )
			EXCEPTION_MESSAGE( OutOfMemory )
			EXCEPTION_MESSAGE( InvalidOperation )
			EXCEPTION_MESSAGE( InvalidLayoutChange )
			EXCEPTION_MESSAGE( NullPointer )	
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
			EXCEPTION_MESSAGE( NoException )
			#undef EXCEPTION_MESSAGE
			default :
				os << "UnknownException";
		}

		return os;
	}

#endif

