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

#ifndef __KARTET_FFT__
#define __KARTET_FFT__

	#include <cufft.h>
	#include "Core/Array.hpp"

namespace Kartet
{
	template<typename TIn, typename TOut>
	class FFTContext
	{
		private :
			cufftHandle handle;
			cufftType fftType;

			__host__ FFTContext(const FFTContext&);
		public :
			const Layout inputLayout, outputLayout;

			__host__ FFTContext(const Layout& inputL, const Layout& outputL);
			__host__ ~FFTContext(void);

			__host__ void fft(const Accessor<TIn>& input, const Accessor<TOut>& output, bool forward=true);
			__host__ void ifft(const Accessor<TIn>& input, const Accessor<TOut>& output);

		// static :
			__host__ static bool isValid(const Layout& input, const Layout& output);
	};

// Impl.
	template<typename TIn, typename TOut>
	__host__ FFTContext<TIn, TOut>::FFTContext(const Layout& inputL, const Layout& outputL)
	 :	inputLayout(inputL),
		outputLayout(outputL),
		fftType(static_cast<cufftType>(0))
	{
		cufftResult err = CUFFT_SUCCESS;

		// Check the feasibility :
		if(!isValid(inputLayout, outputLayout))
			throw IncompatibleLayout;

		// Find the type / Choose the computation mode :
		if(SameTypes<TIn,float>::test && SameTypes<TOut,cuFloatComplex>::test)
			fftType = CUFFT_R2C;
		else if(SameTypes<TIn,cuFloatComplex>::test && SameTypes<TOut,float>::test)
			fftType = CUFFT_C2R;
		else if(SameTypes<TIn,cuFloatComplex>::test && SameTypes<TOut,cuFloatComplex>::test)
			fftType = CUFFT_C2C;
		else if(SameTypes<TIn,double>::test && SameTypes<TOut,cuDoubleComplex>::test)
			fftType = CUFFT_D2Z;
		else if(SameTypes<TIn,cuDoubleComplex>::test && SameTypes<TOut,double>::test)
			fftType = CUFFT_Z2D;
		else if(SameTypes<TIn,cuDoubleComplex>::test && SameTypes<TOut,cuDoubleComplex>::test)
			fftType = CUFFT_Z2Z;
		else
			throw InvalidOperation;

		// Create the handle :
		if(inputLayout.getNumSlices()==1)
		{
			err = cufftPlan2d(&handle, inputLayout.getNumColumns(), inputLayout.getNumRows(), fftType);
			if(err!=CUFFT_SUCCESS)
				throw static_cast<Exception>(CuFFTExceptionOffset + err);

			err = cufftSetCompatibilityMode(handle, CUFFT_COMPATIBILITY_NATIVE);
			if(err!=CUFFT_SUCCESS)
				throw static_cast<Exception>(CuFFTExceptionOffset + err);
		}
		else if(inputLayout.getNumColumns()!=1 && inputLayout.getNumRows()!=1)
		{
			int nI[2] = {inputLayout.getNumColumns(), inputLayout.getNumRows()},
			    nO[2] = {outputLayout.getNumColumns(), outputLayout.getNumRows()};
			int idist = inputLayout.getLeadingSlices(),
			    odist = outputLayout.getLeadingSlices();

			err = cufftPlanMany(&handle, 2, nI, nI, 1, idist, nO, 1, odist, fftType, inputLayout.getNumSlices());
			if(err!=CUFFT_SUCCESS)
				throw static_cast<Exception>(CuFFTExceptionOffset + err);

			err = cufftSetCompatibilityMode(handle, CUFFT_COMPATIBILITY_NATIVE);
			if(err!=CUFFT_SUCCESS)
				throw static_cast<Exception>(CuFFTExceptionOffset + err);
		}
		else
			throw NotSupported;
	}

	template<typename TIn, typename TOut>
	__host__ FFTContext<TIn, TOut>::~FFTContext(void)
	{
		// Destroy the handle :
		cufftDestroy(handle);
	}

	template<typename TIn, typename TOut>
	__host__ void FFTContext<TIn, TOut>::fft(const Accessor<TIn>& input, const Accessor<TOut>& output, bool forward)
	{	
		if(!input.sameLayoutAs(inputLayout) || !output.sameLayoutAs(outputLayout))	
			throw IncompatibleLayout;
		int direction = CUFFT_FORWARD;
		if(!forward && !(fftType==CUFFT_C2C || fftType==CUFFT_Z2Z))
			throw InvalidOperation;
		if(!forward)
			direction = CUFFT_INVERSE;

		cufftResult err = CUFFT_SUCCESS;
		switch(fftType)
		{
			case CUFFT_R2C : 
				err = cufftExecR2C(handle, reinterpret_cast<float*>(input.getPtr()), reinterpret_cast<cuFloatComplex*>(output.getPtr()));
				break;
			case CUFFT_C2R : 
				err = cufftExecC2R(handle, reinterpret_cast<cuFloatComplex*>(input.getPtr()), reinterpret_cast<float*>(output.getPtr()));
				break;
			case CUFFT_C2C : 
				err = cufftExecC2C(handle, reinterpret_cast<cuFloatComplex*>(input.getPtr()), reinterpret_cast<cuFloatComplex*>(output.getPtr()), direction);
				break;
			case CUFFT_D2Z : 
				err = cufftExecD2Z(handle, reinterpret_cast<double*>(input.getPtr()), reinterpret_cast<cuDoubleComplex*>(output.getPtr()));
				break;
			case CUFFT_Z2D : 	
				err = cufftExecZ2D(handle, reinterpret_cast<cuDoubleComplex*>(input.getPtr()), reinterpret_cast<double*>(output.getPtr()));
				break;
			case CUFFT_Z2Z : 
				err = cufftExecZ2Z(handle, reinterpret_cast<cuDoubleComplex*>(input.getPtr()), reinterpret_cast<cuDoubleComplex*>(output.getPtr()), direction);
				break;
			default :
				throw InvalidOperation;
		}

		if(err!=CUFFT_SUCCESS)
			throw static_cast<Exception>(CuFFTExceptionOffset + err);
	}

	template<typename TIn, typename TOut>
	__host__ void FFTContext<TIn, TOut>::ifft(const Accessor<TIn>& input, const Accessor<TOut>& output)
	{
		fft(input, output, false);
	}

	template<typename TIn, typename TOut>
	__host__ bool FFTContext<TIn, TOut>::isValid(const Layout& input, const Layout& output)
	{
		
		if( 	// Check for the dimensions :
			(input.getNumSlices()!=output.getNumSlices()) ||
			(input.getNumColumns()==1 || input.getNumRows()==1) ||
			(output.getNumColumns()==1 || output.getNumRows()==1) ||
			(!input.isSliceMonolithic() || !output.isSliceMonolithic()) ||
			// Check for the types :
			(!TypeInfo<TIn>::isComplex && !TypeInfo<TOut>::isComplex) ||
			// Check for the sizes : 
			(TypeInfo<TIn>::isComplex && TypeInfo<TOut>::isComplex && !(input.getNumColumns()==output.getNumColumns() && input.getNumRows()==output.getNumRows())) ||
			(TypeInfo<TIn>::isComplex && !TypeInfo<TOut>::isComplex && !(input.getNumColumns()==output.getNumColumns() && input.getNumRows()/2+1==output.getNumRows())) ||
			(!TypeInfo<TIn>::isComplex && TypeInfo<TOut>::isComplex && !(input.getNumColumns()==output.getNumColumns() && input.getNumRows()==output.getNumRows()/2+1)) )
			return false;
		else
			return true;
	}

} // namespace Kartet

#endif

