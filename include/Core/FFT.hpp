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
	class FFTContext
	{
		public :
			enum Operation
			{
				R2C,
				C2R,
				C2C,
				D2Z,
				Z2D,
				Z2Z
			};

			enum PlaneFlag
			{
				Single,
				Many
			};

		private :
			#ifdef __CUDACC__
				cufftHandle handle;
			#endif

			#ifdef KARTET_USE_FFTW
				fftwf_plan fftwHandleFloat;
				fftw_plan fftwHandleDouble;				
			#endif
			
			__host__ inline FFTContext(const FFTContext&);
			__host__ inline void setup(void);
		public :
			const Operation operation;
			const PlaneFlag planeFlag;
			const Layout inputLayout, outputLayout;

			__host__ inline FFTContext(const Operation& _operation, const Layout& inputL, const Layout& outputL, const PlaneFlag& _planeFlag=Single);
			template<typename TIn, typename TOut, Location l>
			__host__ FFTContext(const Accessor<TIn,l>& input, const Accessor<TOut,l>& output, const PlaneFlag& _planeFlag=Single);
			__host__ inline ~FFTContext(void);

			// Converter :
			#ifdef __CUDACC__
			__host__ static inline cufftType getCuFFTType(const Operation& op);
			#endif

			// Layout tools :
			__host__ static inline bool isValid(const Operation& _operation, const Layout& input, const Layout& output, const PlaneFlag& _planeFlag);
			template<typename TIn, typename TOut>
			__host__ static Operation getOperation(void);
			template<typename TIn, typename TOut>
			__host__ static bool checkTypes(const Operation& _operation);

			template<typename TIn, typename TOut, Location l>
			__host__ void fft(const Accessor<TIn,l>& input, const Accessor<TOut,l>& output, const bool forward=true);
			template<typename TIn, typename TOut, Location l>
			__host__ void ifft(const Accessor<TIn,l>& input, const Accessor<TOut,l>& output);
	};
} // namespace Kartet

	#include "FFTTools.hpp"

#endif

