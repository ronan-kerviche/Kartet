/* ************************************************************************************************************* */
/*                                                                                                               */
/*     Kartet                                                                                                    */
/*     A Simple C++ Array Library for CUDA                                                                       */
/*                                                                                                               */
/*     LICENSE : The MIT License                                                                                 */
/*     Copyright (c) 2015-2017 Ronan Kerviche                                                                    */
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
	\file    FFTTools.hpp
	\brief   FFT Context implementation.
	\author  R. Kerviche
	\date    November 1st 2009
**/

#ifndef __KARTET_FFT_TOOLS__
#define __KARTET_FFT_TOOLS__

namespace Kartet
{
	/**
	\brief FFTContext constructor.
	\tparam T The base type of the computation (either float or double).
	\tparam l Location of the computation (see Kartet::Location).
	\param _inputLayout Layout of the input data.
	\param _outputLayout Layout of the output data.
	\param _arrayMode Transform mode.
	
	The transform type will be deduced automatically from these parameters.

	\throw IncompatibleLayout If the layouts are not compatible for any transform or if the layout are not (sufficiently) monolithic for a computation with FFTW.
	**/
	template<typename T, Location l>
	__host__ FFTContext<T,l>::FFTContext(const Layout& _inputLayout, const Layout& _outputLayout, const FFTArrayMode& _arrayMode)
	 :
		#ifdef __CUDACC__
		cufftHandle(0),
		#endif
		#ifdef KARTET_USE_FFTW
		fftwHandleFloat(NULL),
		ifftwHandleFloat(NULL),
		fftwHandleDouble(NULL),
		ifftwHandleDouble(NULL),
		#endif
		operation(getOperation(_inputLayout, _outputLayout)),
		arrayMode(_arrayMode),
		inputLayout(_inputLayout),
		outputLayout(_outputLayout)
	{
		initialize();
	}

	/**
	\brief FFTContext constructor.
	\tparam T The base type of the computation (either float or double).
	\tparam l Location of the computation (see Kartet::Location).
	\param _inputOutputLayout Layout of the input and output data.
	\param _arrayMode Transform mode.

	The transform type will be deduced automatically from these parameters.

	\throw IncompatibleLayout If the layouts are not compatible for any transform or if the layout are not (sufficiently) monolithic for a computation with FFTW.
	**/
	template<typename T, Location l>
	__host__ FFTContext<T,l>::FFTContext(const Layout& _inputOutputLayout, const FFTArrayMode& _arrayMode)
	 :
		#ifdef __CUDACC__
		cufftHandle(0),
		#endif
		#ifdef KARTET_USE_FFTW
		fftwHandleFloat(NULL),
		ifftwHandleFloat(NULL),
		fftwHandleDouble(NULL),
		ifftwHandleDouble(NULL),
		#endif
		operation(getOperation(_inputOutputLayout, _inputOutputLayout)),
		arrayMode(_arrayMode),
		inputLayout(_inputOutputLayout),
		outputLayout(_inputOutputLayout)
	{
		initialize();
	}

	template<typename T, Location l>
	__host__ FFTContext<T,l>::~FFTContext(void)
	{
		// Destroy the handles :
		#ifdef __CUDACC__
			cufftDestroy(cufftHandle);
		#endif

		#ifdef KARTET_USE_FFTW
			fftwf_destroy_plan(fftwHandleFloat);
			fftwf_destroy_plan(ifftwHandleFloat);
			fftw_destroy_plan(fftwHandleDouble);
			fftw_destroy_plan(ifftwHandleDouble);
		#endif
	}

	template<typename T, Location l>
	__host__ FFTOperation FFTContext<T,l>::getOperation(const Layout& i, const Layout& o)
	{
		// Test type first :
		STATIC_ASSERT_VERBOSE((Kartet::IsSame<T,float>::value || Kartet::IsSame<T,double>::value), TYPE_NOT_SUPPORTED)
	
		// Test layouts (and get the corresponding operation) :
		FFTOperation op = C2C;
		if(i.sameMonolithicLayoutAs(o))
			op = C2C;
		else if(i.numRows()==o.numRows() && i.numColumns()==o.numColumns() && i.numSlices()==std::floor(o.numSlices()/2)+1)
			op = R2C;
		else if(i.numRows()==o.numRows() && i.numColumns()==o.numColumns() && std::floor(i.numSlices()/2)+1==o.numSlices())
			op = C2R;
		else if(i.numRows()==o.numRows() && i.numColumns()==std::floor(o.numColumns()/2)+1 && i.numSlices()==o.numSlices())
			op = R2C;
		else if(i.numRows()==o.numRows() && std::floor(i.numColumns()/2)+1==o.numColumns() && i.numSlices()==o.numSlices())
			op = C2R;
		else if(i.numRows()==std::floor(o.numRows()/2)+1 && i.numColumns()==o.numColumns() && i.numSlices()==o.numSlices())
			op = R2C;
		else if(std::floor(i.numRows()/2)+1==o.numRows() && i.numColumns()==o.numColumns() && i.numSlices()==o.numSlices())
			op = C2R;
		else
			throw IncompatibleLayout;

		// Promote the operation depending on the type :
		if(Kartet::IsSame<T,double>::value)
		{
			switch(op)
			{
				case R2C : op = D2Z; break;
				case C2R : op = Z2D; break;
				case C2C : op = Z2Z; break;
				default : 
					throw InvalidOperation;
			}
		}
		return op;
	}

	#ifdef __CUDACC__
	template<typename T, Location l>
	__host__ cufftType FFTContext<T,l>::getCuFFTType(const FFTOperation& op)
	{
		return	op==R2C ? CUFFT_R2C : 
			op==C2R ? CUFFT_C2R : 
			op==C2C ? CUFFT_C2C :
			op==D2Z ? CUFFT_D2Z : 
			op==Z2D ? CUFFT_Z2D : 	
			/*op==Z2Z ?*/ CUFFT_Z2Z;
	}
	#endif

	template<typename T, Location l>
	__host__ void FFTContext<T,l>::initialize(void)
	{
		if(l==Kartet::DeviceSide)
		{
			#ifdef __CUDACC__
				cufftResult err = CUFFT_SUCCESS;

				if(arrayMode==Volume)
				{
					if(inputLayout.numSlices()==1 && inputLayout.numColumns()==1)
						err = cufftPlan1d(&cufftHandle, inputLayout.numRows(), getCuFFTType(operation), 1); // 1 transformation.
					else if(inputLayout.numSlices()==1)
						err = cufftPlan2d(&cufftHandle, inputLayout.numRows(), inputLayout.numColumns(), getCuFFTType(operation));
					else
						err = cufftPlan3d(&cufftHandle, inputLayout.numRows(), inputLayout.numColumns(), inputLayout.numSlices(), getCuFFTType(operation));
				}
				else // Many planes
				{
					if(inputLayout.numColumns()>1 && inputLayout.numSlices()==1) // Many 1D along the rows
					{
						int nI[1] = {static_cast<int>(inputLayout.numRows())},
						    nO[1] = {static_cast<int>(outputLayout.numRows())};
						int idist = static_cast<int>(inputLayout.columnsStride()),
						    odist = static_cast<int>(outputLayout.columnsStride());

						err = cufftPlanMany(&cufftHandle, 1, nI, nI, 1, idist, nO, 1, odist, getCuFFTType(operation), inputLayout.numColumns());
					}
					else if(inputLayout.numColumns()>1 && inputLayout.numSlices()>1) // Many 2D along the slices
					{
						int nI[2] = {static_cast<int>(inputLayout.numRows()), static_cast<int>(inputLayout.numColumns())},
						    nO[2] = {static_cast<int>(outputLayout.numRows()), static_cast<int>(outputLayout.numColumns())};
						int idist = static_cast<int>(inputLayout.slicesStride()),
						    odist = static_cast<int>(outputLayout.slicesStride());
						err = cufftPlanMany(&cufftHandle, 2, nI, nI, 1, idist, nO, 1, odist, getCuFFTType(operation), inputLayout.numSlices());
					}
					else
						throw NotSupported;
				}

				// Test :
				if(err!=CUFFT_SUCCESS)
					throw static_cast<Exception>(CuFFTExceptionOffset + err);

				// Set the compatibility mode :
				err = cufftSetCompatibilityMode(cufftHandle, CUFFT_COMPATIBILITY_FFTW_PADDING);
				if(err!=CUFFT_SUCCESS)
					throw static_cast<Exception>(CuFFTExceptionOffset + err);
			#else
				throw NotSupported;
			#endif
		}
		else if(l==Kartet::HostSide)
		{
			#ifdef KARTET_USE_FFTW
				if(Kartet::IsSame<T,float>::value)
				{
					if((inputLayout.numSlices()==1 && inputLayout.numColumns()==1) || (arrayMode==Planes && inputLayout.numSlices()==1))
					{
						fftwHandleFloat = fftwf_plan_dft_1d(static_cast<int>(inputLayout.numRows()), NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_UNALIGNED | FFTW_PRESERVE_INPUT);
						ifftwHandleFloat = fftwf_plan_dft_1d(static_cast<int>(inputLayout.numRows()), NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_UNALIGNED | FFTW_PRESERVE_INPUT);
					}
					else if(inputLayout.numSlices()==1 || arrayMode==Planes)
					{
						if(!inputLayout.isSliceMonolithic() || !outputLayout.isSliceMonolithic())
							throw IncompatibleLayout;
						fftwHandleFloat = fftwf_plan_dft_2d(static_cast<int>(inputLayout.numRows()), static_cast<int>(inputLayout.numColumns()), NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_UNALIGNED | FFTW_PRESERVE_INPUT);
						ifftwHandleFloat = fftwf_plan_dft_2d(static_cast<int>(inputLayout.numRows()), static_cast<int>(inputLayout.numColumns()), NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_UNALIGNED | FFTW_PRESERVE_INPUT);
					}
					else
					{
						if(!inputLayout.isMonolithic() || !outputLayout.isMonolithic())
							throw IncompatibleLayout;
						fftwHandleFloat = fftwf_plan_dft_3d(static_cast<int>(inputLayout.numRows()), static_cast<int>(inputLayout.numColumns()), static_cast<int>(inputLayout.numSlices()), NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_UNALIGNED | FFTW_PRESERVE_INPUT);
						ifftwHandleFloat = fftwf_plan_dft_3d(static_cast<int>(inputLayout.numRows()), static_cast<int>(inputLayout.numColumns()), static_cast<int>(inputLayout.numSlices()), NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_UNALIGNED | FFTW_PRESERVE_INPUT);
					}	
				}
				else if(Kartet::IsSame<T,double>::value)
				{
					if((inputLayout.numSlices()==1 && inputLayout.numColumns()==1) || (arrayMode==Planes && inputLayout.numSlices()==1))
					{
						fftwHandleDouble= fftw_plan_dft_1d(static_cast<int>(inputLayout.numRows()), NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_UNALIGNED | FFTW_PRESERVE_INPUT);
						ifftwHandleDouble= fftw_plan_dft_1d(static_cast<int>(inputLayout.numRows()), NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_UNALIGNED | FFTW_PRESERVE_INPUT);
					}
					else if(inputLayout.numSlices()==1 || arrayMode==Planes)
					{
						if(!inputLayout.isSliceMonolithic() || !outputLayout.isSliceMonolithic())
							throw IncompatibleLayout;
						fftwHandleDouble= fftw_plan_dft_2d(static_cast<int>(inputLayout.numRows()), static_cast<int>(inputLayout.numColumns()), NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_UNALIGNED | FFTW_PRESERVE_INPUT);
						ifftwHandleDouble= fftw_plan_dft_2d(static_cast<int>(inputLayout.numRows()), static_cast<int>(inputLayout.numColumns()), NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_UNALIGNED | FFTW_PRESERVE_INPUT);
					}
					else
					{
						if(!inputLayout.isMonolithic() || !outputLayout.isMonolithic())
							throw IncompatibleLayout;
						fftwHandleDouble= fftw_plan_dft_3d(static_cast<int>(inputLayout.numRows()), static_cast<int>(inputLayout.numColumns()), static_cast<int>(inputLayout.numSlices()), NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_UNALIGNED | FFTW_PRESERVE_INPUT);
						ifftwHandleDouble= fftw_plan_dft_3d(static_cast<int>(inputLayout.numRows()), static_cast<int>(inputLayout.numColumns()), static_cast<int>(inputLayout.numSlices()), NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_UNALIGNED | FFTW_PRESERVE_INPUT);
					}
				}
				else
					throw NotSupported;
			#else
				throw NotSupported;
			#endif
		}
		else
			throw InvalidLocation;
	}

	/**
	\brief Compute the FFT.
	\param input Input accessor.
	\param output Output accesor.
	\param forward If true compute the forward transform, otherwise compute the backward (inverse) transform.
	\return The output accessor.
	**/
	template<typename T, Location l>
	__host__ const Accessor<Complex<T>,l>& FFTContext<T,l>::fft(const Accessor<Complex<T>,l>& input, const Accessor<Complex<T>,l>& output, const bool forward)
	{
		if(operation!=C2C && operation!=Z2Z)
			throw InvalidOperation;
		else if(!inputLayout.sameLayoutAs(input) || !outputLayout.sameLayoutAs(output))
			throw IncompatibleLayout;
		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				int direction = forward ? CUFFT_FORWARD : CUFFT_INVERSE;
				cufftResult err = CUFFT_SUCCESS;
				if(Kartet::IsSame<T,float>::value)
					err = cufftExecC2C(cufftHandle, reinterpret_cast<cuFloatComplex*>(input.dataPtr()), reinterpret_cast<cuFloatComplex*>(output.dataPtr()), direction);
				else
					err = cufftExecZ2Z(cufftHandle, reinterpret_cast<cuDoubleComplex*>(input.dataPtr()), reinterpret_cast<cuDoubleComplex*>(output.dataPtr()), direction);
				if(err!=CUFFT_SUCCESS)
					throw static_cast<Exception>(CuFFTExceptionOffset + err);
			#else
				throw NotSupported;
			#endif
		}
		else
		{
			#ifdef KARTET_USE_FFTW
				if(IsSame<T,float>::value)
				{
					if(arrayMode==Volume)
						fftwf_execute_dft(forward ? fftwHandleFloat : ifftwHandleFloat, reinterpret_cast<fftwf_complex*>(input.dataPtr()), reinterpret_cast<fftwf_complex*>(output.dataPtr()));
					else if(inputLayout.numSlices()==1)
					{
						for(index_t k=0; k<inputLayout.numColumns(); k++)
							fftwf_execute_dft(forward ? fftwHandleFloat : ifftwHandleFloat, reinterpret_cast<fftwf_complex*>(input.column(k).dataPtr()), reinterpret_cast<fftwf_complex*>(output.column(k).dataPtr()));
					}
					else
					{
						for(index_t k=0; k<inputLayout.numSlices(); k++)
							fftwf_execute_dft(forward ? fftwHandleFloat : ifftwHandleFloat, reinterpret_cast<fftwf_complex*>(input.slice(k).dataPtr()), reinterpret_cast<fftwf_complex*>(output.slice(k).dataPtr()));
					}
				}
				else
				{
					if(arrayMode==Volume)
						fftw_execute_dft(forward ? fftwHandleDouble : ifftwHandleDouble, reinterpret_cast<fftw_complex*>(input.dataPtr()), reinterpret_cast<fftw_complex*>(output.dataPtr()));
					else if(inputLayout.numSlices()==1)
					{
						for(index_t k=0; k<inputLayout.numColumns(); k++)
							fftw_execute_dft(forward ? fftwHandleDouble : ifftwHandleDouble, reinterpret_cast<fftw_complex*>(input.column(k).dataPtr()), reinterpret_cast<fftw_complex*>(output.column(k).dataPtr()));
					}
					else
					{
						for(index_t k=0; k<inputLayout.numSlices(); k++)
							fftw_execute_dft(forward ? fftwHandleDouble : ifftwHandleDouble, reinterpret_cast<fftw_complex*>(input.slice(k).dataPtr()), reinterpret_cast<fftw_complex*>(output.slice(k).dataPtr()));
					}
				}
			#else
				throw NotSupported;
			#endif
		}
		return output;
	}

	/**
	\brief Compute the backward FFT.
	\param input Input accessor.
	\param output Output accesor.
	\return The output accessor.
	**/
	template<typename T, Location l>
	__host__ const Accessor<Complex<T>,l>& FFTContext<T,l>::ifft(const Accessor<Complex<T>,l>& input, const Accessor<Complex<T>,l>& output)
	{
		return fft(input, output, false);
	}
	
	/**
	\brief Compute the FFT.
	\param input Input accessor.
	\param output Output accesor.
	\return The output accessor.
	**/
	template<typename T, Location l>
	__host__ const Accessor<T,l>& FFTContext<T,l>::fft(const Accessor<Complex<T>,l>& input, const Accessor<T,l>& output)
	{
		if(operation!=C2R && operation!=Z2D)
			throw InvalidOperation;
		else if(!inputLayout.sameLayoutAs(input) || !outputLayout.sameLayoutAs(output))
			throw IncompatibleLayout;
		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cufftResult err = CUFFT_SUCCESS;
				if(Kartet::IsSame<T,float>::value)
					err = cufftExecC2R(cufftHandle, reinterpret_cast<cuFloatComplex*>(input.dataPtr()), reinterpret_cast<float*>(output.dataPtr()));
				else
					err = cufftExecZ2D(cufftHandle, reinterpret_cast<cuDoubleComplex*>(input.dataPtr()), reinterpret_cast<double*>(output.dataPtr()));
				if(err!=CUFFT_SUCCESS)
					throw static_cast<Exception>(CuFFTExceptionOffset + err);
			#else
				throw NotSupported;
			#endif
		}
		else
		{
			#ifdef KARTET_USE_FFTW
				if(IsSame<T,float>::value)
				{
					if(arrayMode==Volume)
						fftwf_execute_dft(fftwHandleFloat, reinterpret_cast<fftwf_complex*>(input.dataPtr()), reinterpret_cast<fftwf_complex*>(output.dataPtr()));
					else if(inputLayout.numSlices()==1)
					{
						for(index_t k=0; k<inputLayout.numColumns(); k++)
							fftwf_execute_dft(fftwHandleFloat, reinterpret_cast<fftwf_complex*>(input.column(k).dataPtr()), reinterpret_cast<fftwf_complex*>(output.column(k).dataPtr()));
					}
					else
					{
						for(index_t k=0; k<inputLayout.numSlices(); k++)
							fftwf_execute_dft(fftwHandleFloat, reinterpret_cast<fftwf_complex*>(input.slice(k).dataPtr()), reinterpret_cast<fftwf_complex*>(output.slice(k).dataPtr()));
					}
				}
				else
				{
					if(arrayMode==Volume)
						fftw_execute_dft(fftwHandleDouble, reinterpret_cast<fftw_complex*>(input.dataPtr()), reinterpret_cast<fftw_complex*>(output.dataPtr()));
					else if(inputLayout.numSlices()==1)
					{
						for(index_t k=0; k<inputLayout.numColumns(); k++)
							fftw_execute_dft(fftwHandleDouble, reinterpret_cast<fftw_complex*>(input.column(k).dataPtr()), reinterpret_cast<fftw_complex*>(output.column(k).dataPtr()));
					}
					else
					{
						for(index_t k=0; k<inputLayout.numSlices(); k++)
							fftw_execute_dft(fftwHandleDouble, reinterpret_cast<fftw_complex*>(input.slice(k).dataPtr()), reinterpret_cast<fftw_complex*>(output.slice(k).dataPtr()));
					}
				}
			#else
				throw NotSupported;
			#endif
		}
		return output;
	}

	/**
	\brief Compute the FFT.
	\param input Input accessor.
	\param output Output accesor.
	\return The output accessor.
	**/
	template<typename T, Location l>
	__host__ const Accessor<Complex<T>,l>& FFTContext<T,l>::fft(const Accessor<T,l>& input, const Accessor<Complex<T>,l>& output)
	{
		if(operation!=R2C && operation!=D2Z)
			throw InvalidOperation;
		else if(!inputLayout.sameLayoutAs(input) || !outputLayout.sameLayoutAs(output))
			throw IncompatibleLayout;
		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cufftResult err = CUFFT_SUCCESS;
				if(Kartet::IsSame<T,float>::value)
					err = cufftExecR2C(cufftHandle, reinterpret_cast<float*>(input.dataPtr()), reinterpret_cast<cuFloatComplex*>(output.dataPtr()));
				else
					err = cufftExecD2Z(cufftHandle, reinterpret_cast<double*>(input.dataPtr()), reinterpret_cast<cuDoubleComplex*>(output.dataPtr()));
				if(err!=CUFFT_SUCCESS)
					throw static_cast<Exception>(CuFFTExceptionOffset + err);
			#else
				throw NotSupported;
			#endif
		}
		else
		{
			#ifdef KARTET_USE_FFTW
				if(IsSame<T,float>::value)
				{
					if(arrayMode==Volume)
						fftwf_execute_dft(fftwHandleFloat, reinterpret_cast<fftwf_complex*>(input.dataPtr()), reinterpret_cast<fftwf_complex*>(output.dataPtr()));
					else if(inputLayout.numSlices()==1)
					{
						for(index_t k=0; k<inputLayout.numColumns(); k++)
							fftwf_execute_dft(fftwHandleFloat, reinterpret_cast<fftwf_complex*>(input.column(k).dataPtr()), reinterpret_cast<fftwf_complex*>(output.column(k).dataPtr()));
					}
					else
					{
						for(index_t k=0; k<inputLayout.numSlices(); k++)
							fftwf_execute_dft(fftwHandleFloat, reinterpret_cast<fftwf_complex*>(input.slice(k).dataPtr()), reinterpret_cast<fftwf_complex*>(output.slice(k).dataPtr()));
					}
				}
				else
				{
					if(arrayMode==Volume)
						fftw_execute_dft(fftwHandleDouble, reinterpret_cast<fftw_complex*>(input.dataPtr()), reinterpret_cast<fftw_complex*>(output.dataPtr()));
					else if(inputLayout.numSlices()==1)
					{
						for(index_t k=0; k<inputLayout.numColumns(); k++)
							fftw_execute_dft(fftwHandleDouble, reinterpret_cast<fftw_complex*>(input.column(k).dataPtr()), reinterpret_cast<fftw_complex*>(output.column(k).dataPtr()));
					}
					else
					{
						for(index_t k=0; k<inputLayout.numSlices(); k++)
							fftw_execute_dft(fftwHandleDouble, reinterpret_cast<fftw_complex*>(input.slice(k).dataPtr()), reinterpret_cast<fftw_complex*>(output.slice(k).dataPtr()));
					}
				}
			#else
				throw NotSupported;
			#endif
		}
		return output;
	}

} // namespace Kartet

#endif

