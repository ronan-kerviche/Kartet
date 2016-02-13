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
	\file    FFT.hpp
	\brief   FFT Context definition.
	\author  R. Kerviche
	\date    November 1st 2009
**/

#ifndef __KARTET_FFT__
#define __KARTET_FFT__

	#ifdef __CUDACC__
		#include <cufft.h>
	#endif

	#ifdef KARTET_USE_FFTW
		#ifdef __cplusplus
		extern "C"
		{
		#endif
			#include <fftw3.h>
		#ifdef __cplusplus
		}
		#endif
	#endif
	
	#include "Core/LibTools.hpp"
	#include "Core/Array.hpp"

namespace Kartet
{
	/**
	\related Kartet::FFTContext
	\brief Transform types.
	**/
	enum FFTOperation
	{
		/// Real to complex.
		R2C,
		/// Complex to real.
		C2R,
		/// Complex to complex.
		C2C,
		/// Real to complex (double precision).
		D2Z,
		/// Complex to real (double precision).
		Z2D,
		/// Complex to complex (double precision).
		Z2Z
	};

	/**
	\related Kartet::FFTContext
	\brief Transform modes.
	**/
	enum FFTArrayMode
	{
		/// Compute over the full volume.
		Volume,
		/// Split the transform into planes. The outter most non-singular dimension is treated as a concatenation of arrays for the transform.
		Planes
	};

	/**
	\brief FFT Context.

	When compiled for device code (with NVCC), the library will use CuFFT. In all the binary, the library can perform FFT transforms on the host side with any of the following define : 
	\code	
	-D KARTET_USE_FFTW
	\endcode

	Example :
	\code
	Kartet::Array< Kartet::Complex<float> > A(8, 8, 4), B(8, 8, 4);
	Kartet::FFTContext<float> fft(A, B, Kartet::Planes);
	A = Kartet::IndexI() + Kartet::IndexJ() + Kartet::IndexK();
	std::cout << fft.fft(A, B) << std::endl;
	std::cout << fft.ifft(B, A) << std::endl;
	\endcode
	**/
	template<typename T, Location l=KARTET_DEFAULT_LOCATION>
	class FFTContext
	{
		private : 
			#ifdef __CUDACC__
				cufftHandle cufftHandle;
			#endif

			#ifdef KARTET_USE_FFTW
				fftwf_plan 	fftwHandleFloat,
						ifftwHandleFloat;
				fftw_plan 	fftwHandleDouble,
						ifftwHandleDouble;
			#endif

			__host__ FFTContext(const FFTContext&);
			__host__ static FFTOperation getOperation(const Layout& inputL, const Layout& outputL);
			#ifdef __CUDACC__
			__host__ static cufftType getCuFFTType(const FFTOperation& op);
			#endif
			__host__ void initialize(void);

		public :
			const FFTOperation 	operation;
			const FFTArrayMode 	arrayMode;
			const Layout 		inputLayout, 
						outputLayout;

			__host__ FFTContext(const Layout& inputL, const Layout& outputL, const FFTArrayMode& _arrayMode=Volume);
			__host__ FFTContext(const Layout& inputOutputL, const FFTArrayMode& _arrayMode=Volume);
			__host__ ~FFTContext(void);

			__host__ const Accessor<Complex<T>,l>& fft(const Accessor<Complex<T>,l>& input, const Accessor<Complex<T>,l>& output, const bool forward=true);
			__host__ const Accessor<Complex<T>,l>& ifft(const Accessor<Complex<T>,l>& input, const Accessor<Complex<T>,l>& output);
			__host__ const Accessor<T,l>& fft(const Accessor<Complex<T>,l>& input, const Accessor<T,l>& output);
			__host__ const Accessor<Complex<T>,l>& fft(const Accessor<T,l>& input, const Accessor<Complex<T>,l>& output);
			
	};

} // namespace Kartet

	#include "FFTTools.hpp"

#endif

