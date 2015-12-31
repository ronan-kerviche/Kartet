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


#ifndef __KARTET_FFT_TOOLS__
#define __KARTET_FFT_TOOLS__

namespace Kartet
{
// Type Lists :
	typedef TypeList< float,
		TypeList< double,
		TypeList< cuFloatComplex,
		TypeList< cuDoubleComplex,
		Void
		> > > > FFTKnownTypes;

// FFT :
	__host__ inline FFTContext::FFTContext(const Operation& _operation, const Layout& inputL, const Layout& outputL, const PlaneFlag& _planeFlag)
	 :	operation(_operation),
		planeFlag(_planeFlag),
		inputLayout(inputL),
		outputLayout(outputL)
	{
		setup();
	}

	template<typename TIn, typename TOut, Location l>
	__host__ FFTContext::FFTContext(const Accessor<TIn,l>& input, const Accessor<TOut,l>& output, const PlaneFlag& _planeFlag)
	 :	operation(getOperation<TIn,TOut>()),
		planeFlag(_planeFlag),
		inputLayout(input.layout()),
		outputLayout(output.layout())
	{
		setup();
	}

	__host__ inline void FFTContext::setup(void)
	{
		// Check the feasibility :
		if(!isValid(operation, inputLayout, outputLayout, planeFlag))
			throw IncompatibleLayout;

		#ifdef __CUDACC__
			// Prepare Device Side :
			cufftResult err = CUFFT_SUCCESS;

			if(planeFlag==Single)
			{
				if(inputLayout.numSlices()==1 && inputLayout.numColumns()==1)
					err = cufftPlan1d(&handle, inputLayout.numRows(), getCuFFTType(operation), 1); // 1 transformation.
				else if(inputLayout.numSlices()==1)
					err = cufftPlan2d(&handle, inputLayout.numRows(), inputLayout.numColumns(), getCuFFTType(operation));
				else
					err = cufftPlan3d(&handle, inputLayout.numRows(), inputLayout.numColumns(), inputLayout.numSlices(), getCuFFTType(operation));
			}
			else // Many planes
			{
				if(inputLayout.numColumns()>1 && inputLayout.numSlices()==1) // Many 1D along the rows
				{
					int nI[1] = {static_cast<int>(inputLayout.numRows())},
					    nO[1] = {static_cast<int>(outputLayout.numRows())};
					int idist = static_cast<int>(inputLayout.columnsStride()),
					    odist = static_cast<int>(outputLayout.columnsStride());

					err = cufftPlanMany(&handle, 1, nI, nI, 1, idist, nO, 1, odist, getCuFFTType(operation), inputLayout.numColumns());
				}
				else if(inputLayout.numColumns()>1 && inputLayout.numSlices()>1) // Many 2D along the slices
				{
					int nI[2] = {static_cast<int>(inputLayout.numRows()), static_cast<int>(inputLayout.numColumns())},
					    nO[2] = {static_cast<int>(outputLayout.numRows()), static_cast<int>(outputLayout.numColumns())};
					int idist = static_cast<int>(inputLayout.slicesStride()),
					    odist = static_cast<int>(outputLayout.slicesStride());
					err = cufftPlanMany(&handle, 2, nI, nI, 1, idist, nO, 1, odist, getCuFFTType(operation), inputLayout.numSlices());
				}
				else
					throw NotSupported;
			}

			// Test :
			if(err!=CUFFT_SUCCESS)
				throw static_cast<Exception>(CuFFTExceptionOffset + err);

			// Set the compatibility mode :
			err = cufftSetCompatibilityMode(handle, CUFFT_COMPATIBILITY_FFTW_PADDING);
			if(err!=CUFFT_SUCCESS)
				throw static_cast<Exception>(CuFFTExceptionOffset + err);
		#endif

		#ifdef KARTET_USE_FFTW
			if(inputLayout.numSlices()==1 && inputLayout.numColumns()==1)
			{
				fftwHandleFloat = fftwf_plan_dft_1d(static_cast<int>(inputLayout.numRows()), NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_UNALIGNED | FFTW_PRESERVE_INPUT);
				fftwHandleDouble= fftw_plan_dft_1d(static_cast<int>(inputLayout.numRows()), NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_UNALIGNED | FFTW_PRESERVE_INPUT);
			}
			else if(inputLayout.numSlices()==1)
			{
				fftwHandleFloat = fftwf_plan_dft_2d(inputLayout.numRows(), inputLayout.numColumns(), NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_UNALIGNED | FFTW_PRESERVE_INPUT);
				fftwHandleDouble= fftw_plan_dft_2d(inputLayout.numRows(), inputLayout.numColumns(), NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_UNALIGNED | FFTW_PRESERVE_INPUT);
			}
			else
			{
				fftwHandleFloat = fftwf_plan_dft_3d(inputLayout.numRows(), inputLayout.numColumns(), inputLayout.numSlices(), NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_UNALIGNED | FFTW_PRESERVE_INPUT);
				fftwHandleDouble= fftw_plan_dft_3d(inputLayout.numRows(), inputLayout.numColumns(), inputLayout.numSlices(), NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_UNALIGNED | FFTW_PRESERVE_INPUT);
			}
		#endif
	}

	__host__ FFTContext::~FFTContext(void)
	{
		// Destroy the handles :
		#ifdef __CUDACC__
			cufftDestroy(handle);
		#endif

		#ifdef KARTET_USE_FFTW
			fftwf_destroy_plan(fftwHandleFloat);
			fftw_destroy_plan(fftwHandleDouble);
		#endif
	}

	#ifdef __CUDACC__
	__host__ inline cufftType FFTContext::getCuFFTType(const Operation& op)
	{
		return	((op==R2C) ? CUFFT_R2C : 
			((op==C2R) ? CUFFT_C2R : 
			((op==C2C) ? CUFFT_C2C :
			((op==D2Z) ? CUFFT_D2Z : 
			((op==Z2D) ? CUFFT_Z2D : 	
			/*(op==Z2Z) ?*/ CUFFT_Z2Z)))));
	}
	#endif

	__host__ bool FFTContext::isValid(const Operation& _operation, const Layout& input, const Layout& output, const PlaneFlag& _planeFlag)
	{		
		/*MUST
		input.numRows()>1
		output.numRows()>1*/

		/*if( 	// Check for the dimensions :
			(input.numSlices()!=output.numSlices()) ||
			(input.numColumns()==1 || input.numRows()==1) ||
			(output.numColumns()==1 || output.numRows()==1) ||
			(!input.isSliceMonolithic() || !output.isSliceMonolithic()) ||
			// Check for the types :
			(!TypeInfo<TIn>::isComplex && !TypeInfo<TOut>::isComplex) ||
			// Check for the sizes : 
			(TypeInfo<TIn>::isComplex && TypeInfo<TOut>::isComplex && !(input.numColumns()==output.numColumns() && input.numRows()==output.numRows())) ||
			(TypeInfo<TIn>::isComplex && !TypeInfo<TOut>::isComplex && !(input.numColumns()==output.numColumns() && input.numRows()/2+1==output.numRows())) ||
			(!TypeInfo<TIn>::isComplex && TypeInfo<TOut>::isComplex && !(input.numColumns()==output.numColumns() && input.numRows()==output.numRows()/2+1)) )
			return false;
		else
			return true;*/
		
		return true;
	}

	template<typename TIn, typename TOut>
	__host__ FFTContext::Operation FFTContext::getOperation(void)
	{
		if(SameTypes<TIn,float>::test && SameTypes<TOut,cuFloatComplex>::test)
			return R2C;
		else if(SameTypes<TIn,cuFloatComplex>::test && SameTypes<TOut,float>::test)
			return C2R;
		else if(SameTypes<TIn,cuFloatComplex>::test && SameTypes<TOut,cuFloatComplex>::test)
			return C2C;
		else if(SameTypes<TIn,double>::test && SameTypes<TOut,cuDoubleComplex>::test)
			return D2Z;
		else if(SameTypes<TIn,cuDoubleComplex>::test && SameTypes<TOut,double>::test)
			return Z2D;
		else if(SameTypes<TIn,cuDoubleComplex>::test && SameTypes<TOut,cuDoubleComplex>::test)
			return Z2Z;
		else
			throw InvalidOperation;
	}

	template<typename TIn, typename TOut>
	__host__ bool FFTContext::checkTypes(const Operation& _operation)
	{
		return	((SameTypes<TIn,float>::test && SameTypes<TOut,cuFloatComplex>::test && _operation==R2C) ||
			 (SameTypes<TIn,cuFloatComplex>::test && SameTypes<TOut,float>::test && _operation==C2R) ||
			 (SameTypes<TIn,cuFloatComplex>::test && SameTypes<TOut,cuFloatComplex>::test && _operation==C2C) ||
			 (SameTypes<TIn,double>::test && SameTypes<TOut,cuDoubleComplex>::test && _operation==D2Z) ||
			 (SameTypes<TIn,cuDoubleComplex>::test && SameTypes<TOut,double>::test && _operation==Z2D) ||
			 (SameTypes<TIn,cuDoubleComplex>::test && SameTypes<TOut,cuDoubleComplex>::test && _operation==Z2Z));
	}

	template<typename TIn, typename TOut, Location l>
	__host__ void FFTContext::fft(const Accessor<TIn,l>& input, const Accessor<TOut,l>& output, const bool forward)
	{	
		if(!input.sameLayoutAs(inputLayout) || !output.sameLayoutAs(outputLayout))	
			throw IncompatibleLayout;

		if(!checkTypes<TIn, TOut>(operation))
			throw InvalidOperation;

		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				int direction = CUFFT_FORWARD;
				if(!forward && !(operation==C2C || operation==Z2Z))
					throw InvalidOperation;
				if(!forward)
					direction = CUFFT_INVERSE;

				cufftResult err = CUFFT_SUCCESS;
				switch(operation)
				{
					case R2C : 
						err = cufftExecR2C(handle, reinterpret_cast<float*>(input.getPtr()), reinterpret_cast<cuFloatComplex*>(output.getPtr()));
						break;
					case C2R : 
						err = cufftExecC2R(handle, reinterpret_cast<cuFloatComplex*>(input.getPtr()), reinterpret_cast<float*>(output.getPtr()));
						break;
					case C2C :
						err = cufftExecC2C(handle, reinterpret_cast<cuFloatComplex*>(input.getPtr()), reinterpret_cast<cuFloatComplex*>(output.getPtr()), direction);
						break;
					case D2Z : 
						err = cufftExecD2Z(handle, reinterpret_cast<double*>(input.getPtr()), reinterpret_cast<cuDoubleComplex*>(output.getPtr()));
						break;
					case Z2D : 	
						err = cufftExecZ2D(handle, reinterpret_cast<cuDoubleComplex*>(input.getPtr()), reinterpret_cast<double*>(output.getPtr()));
						break;
					case Z2Z : 
						err = cufftExecZ2Z(handle, reinterpret_cast<cuDoubleComplex*>(input.getPtr()), reinterpret_cast<cuDoubleComplex*>(output.getPtr()), direction);
						break;
					default :
						throw InvalidOperation;
				}
				if(err!=CUFFT_SUCCESS)
					throw static_cast<Exception>(CuFFTExceptionOffset + err);
			#else
				throw NotSupported;
			#endif
		}
		else
		{
			#ifdef KARTET_USE_FFTW
				if(SameTypes<TIn,float>::test || SameTypes<TIn,cuFloatComplex>::test)
					fftwf_execute_dft(fftwHandleFloat, reinterpret_cast<fftwf_complex*>(input.getPtr()), reinterpret_cast<fftwf_complex*>(output.getPtr()));
				else if(SameTypes<TIn,double>::test || SameTypes<TIn,cuDoubleComplex>::test)
					fftw_execute_dft(fftwHandleDouble, reinterpret_cast<fftw_complex*>(input.getPtr()), reinterpret_cast<fftw_complex*>(output.getPtr()));
			#else
				throw NotSupported;
			#endif
		}
	}

	template<typename TIn, typename TOut, Location l>
	__host__ void FFTContext::ifft(const Accessor<TIn,l>& input, const Accessor<TOut,l>& output)
	{
		fft(input, output, false);
	}

} // namespace Kartet

#endif

