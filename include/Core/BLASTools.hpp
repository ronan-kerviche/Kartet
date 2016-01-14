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
	\file    BLASTools.hpp
	\brief   BLAS Context implementation.
	\author  R. Kerviche
	\date    November 1st 2009
**/

#ifndef __KARTET_BLAS_TOOLS__
#define __KARTET_BLAS_TOOLS__

namespace Kartet
{
// Type tools :
	#define ALLOWED_TYPES_VERIFICATION		STATIC_ASSERT_VERBOSE(	(IsSame<T,float>::value || IsSame<T,double>::value || \
										IsSame<T,cuFloatComplex>::value || IsSame<T,cuDoubleComplex>::value || \
										IsSame<T,Complex<float> >::value || IsSame<T,Complex<double> >::value), TYPE_NOT_SUPPORTED )
	#define TYPE_MUST_BE_COMPLEX			STATIC_ASSERT_VERBOSE(Traits<T>::isComplex, TYPE_MUST_BE_COMPLEX)
	#define TEST_MONOLITHIC(x)			{if(!(x).isMonolithic()) throw IncompatibleLayout;}
	#define TEST_SINGLE_SLICE(x)			{if((x).numSlices()>1) throw IncompatibleLayout;} 
	#define TEST_PRODUCT(A, opa, B, opb, C)		{if(!isProductValid(A, opa, B, opb, C)) throw InvalidOperation;}
	#define IF_FLOAT				if(IsSame<T, float>::value)
	#define IF_DOUBLE				if(IsSame<T, double>::value)
	#define IF_CX_FLOAT				if(IsSame<T, cuFloatComplex>::value || IsSame<T, Complex<float> >::value)
	#define IF_CX_DOUBLE				if(IsSame<T, cuDoubleComplex>::value || IsSame<T, Complex<double> >::value)
	#define FCST(x)					const float _##x = real(x);
	#define DCST(x)					const double _##x = real(x);
	#define CCST(x)					const cuFloatComplex _##x = {static_cast<float>(real(x)), static_cast<float>(imag(x))};
	#define ZCST(x)					const cuDoubleComplex _##x = {static_cast<double>(real(x)), static_cast<double>(imag(x))};
	#define FPTR(x)					reinterpret_cast<float*>(x)
	#define DPTR(x)					reinterpret_cast<double*>(x)
	#define CPTR(x)					reinterpret_cast<cuFloatComplex*>(x)
	#define ZPTR(x)					reinterpret_cast<cuDoubleComplex*>(x)
	#define TEST_EXCEPTION(x)			{if(x!=CUBLAS_STATUS_SUCCESS) throw static_cast<Exception>(CuBLASExceptionOffset + x);}

// BLAS :
	/**
	\brief BLASContext constructor.
	\param initDevice If true, will also initialize device.
	**/
	__host__ inline BLASContext::BLASContext(bool initDevice)
	#ifdef __CUDACC__
	 : 	handle(NULL)
	{
		if(initDevice)
		{
			cublasStatus_t err = cublasCreate(&handle);	
			if(err!=CUBLAS_STATUS_SUCCESS)
				throw static_cast<Exception>(CuBLASExceptionOffset + err);
		}
	}
	#else
	{
		UNUSED_PARAMETER(initDevice)
	}
	#endif

	__host__ inline BLASContext::~BLASContext(void)
	{
	#ifdef __CUDACC__
		if(handle!=NULL)
		{
			cublasStatus_t err = cublasDestroy(handle);
			if(err!=CUBLAS_STATUS_SUCCESS)
				throw static_cast<Exception>(CuBLASExceptionOffset + err);
		}
	#endif
	}

	#ifdef __CUDACC__
	/**
	\return The corresponding CuBLAS operation.
	\param op Operation.
	**/
	__host__ inline cublasOperation_t BLASContext::getCuBLASOperation(const MatrixOperation& op)
	{
		return (op==OpTr) ? CUBLAS_OP_T : ((op==OpHr) ? CUBLAS_OP_C : CUBLAS_OP_N);
	}

	/**
	\return The corresponding CuBLAS fill mode.
	\param m Fill mode.
	**/
	__host__ inline cublasFillMode_t BLASContext::getCuBLASFillMode(const MatrixFillMode& m)
	{
		return (m==MatrixFillUp) ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
	}

	/**
	\return The corresponding CuBLAS diagonal type.
	\param t Diagonal type.
	**/
	__host__ inline cublasDiagType_t BLASContext::getCuBLASDiagType(const MatrixDiagType& t)
	{
		return (t==MatrixDiagUnit) ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;
	}

	/**
	\return The corresponding CuBLAS side mode.
	\param s Side mode.
	**/
	__host__ inline cublasSideMode_t BLASContext::getCuBLASSideMode(const MatrixSideMode& s)
	{
		return (s==MatrixRightSide) ? CUBLAS_SIDE_RIGHT : CUBLAS_SIDE_LEFT;
	}
	#endif

	#ifdef KARTET_ADD_CBLAS_INTERFACE
	/**
	\return The corresponding CBLAS operation.
	\param op Operation.
	**/
	__host__ inline CBLAS_TRANSPOSE BLASContext::getCBLASOperation(const MatrixOperation& op)
	{
		return (op==OpTr) ? CblasTrans : ((op==OpHr) ? CblasConjTrans : CblasNoTrans);
	}

	/**
	\return The corresponding CBLAS fill mode.
	\param m Fill mode
	**/
	__host__ inline CBLAS_UPLO BLASContext::getCBLASFillMode(const MatrixFillMode& m)
	{
		return (m==MatrixFillUp) ? CblasUpper : CblasLower;
	}

	/**
	\return The corresponding CBLAS diagonal type.
	\param t Diagonal type.
	**/
	__host__ inline CBLAS_DIAG BLASContext::getCBLASDiagType(const MatrixDiagType& t)
	{
		return (t==MatrixDiagUnit) ? CblasUnit : CblasNonUnit;
	}

	/**
	\return The corresponding CBLAS side mode.
	\param s Side mode.
	**/
	__host__ inline CBLAS_SIDE BLASContext::getCBLASSideMode(const MatrixSideMode& s)
	{
		return (s==MatrixRightSide) ? CblasRight : CblasLeft;
	}
	#endif

	/**
	\brief Test if a product is valid.
	\param A Left side layout. 
	\param opa Left side operation.
	\param B Right side layout.
	\param opb Right side layout.
	\param C Output layout.
	\return True if the sizes are valid.
	**/
	__host__ inline bool BLASContext::isProductValid(const Layout& A, MatrixOperation opa, const Layout& B, MatrixOperation opb, const Layout& C)
	{
		index_t aR = 0,
			aC = 0,
			bR = 0,
			bC = 0,
			cR = C.numRows(),
			cC = C.numColumns();

		if(opa==OpTr || opa==OpHr)
		{
			aR = A.numColumns();
			aC = A.numRows();
		}
		else
		{
			aR = A.numRows();
			aC = A.numColumns();
		}

		if(opb==OpTr || opb==OpHr)
		{
			bR = B.numColumns();
			bC = B.numRows();
		}
		else
		{
			bR = B.numRows();
			bC = B.numColumns();
		}

		return (aR==cR) && (aC==bR) && (bC==cC);
	}

	/**
	\brief Test if a product is valid.
	\param A Left side layout.
	\param B Right side layout.
	\param C Output layout.
	\return True if the sizes are valid.
	**/
	__host__ inline bool BLASContext::isProductValid(const Layout& A, const Layout& B, const Layout& C)
	{
		return BLASContext::isProductValid(A, OpNo, B, OpNo, C);
	}

	/**
	\brief Get the output layout.
	\param A Left side layout.
	\param opa Left side operation.
	\param B Right side layout.
	\param opb Right side layout.
	\return The output layout.
	**/
	__host__ inline Layout BLASContext::getProductLayout(const Layout& A, MatrixOperation opa, const Layout& B, MatrixOperation opb)
	{
		index_t aR = 0,
			aC = 0,
			bR = 0,
			bC = 0;

		if(A.numSlices()!=1 || B.numSlices()!=1)
			throw InvalidOperation;

		if(opa==OpTr || opa==OpHr)
		{
			aR = A.numColumns();
			aC = A.numRows();
		}
		else
		{
			aR = A.numRows();
			aC = A.numColumns();
		}

		if(opb==OpTr || opb==OpHr)
		{
			bR = B.numColumns();
			bC = B.numRows();
		}
		else
		{
			bR = B.numRows();
			bC = B.numColumns();
		}

		if(aC!=bR)
			throw InvalidOperation;
		
		return Layout(aR, bC);
	}

	/**
	\brief Get the output layout.
	\param A Left side layout.
	\param B Right side layout.
	\return The output layout.
	**/
	__host__ inline Layout BLASContext::getProductLayout(const Layout& A, const Layout& B)
	{
		return getProductLayout(A, OpNo, B, OpNo);
	}

// Functions : 
	/**
	\return The index of the absolute maximum value.
	\param x Input data, must be monolithic.

	\f$ = \mbox{argmax}_k\{|x_k|\} \f$
	**/
	template<typename T, Location l>
	__host__ int BLASContext::Iamax(const Accessor<T,l>& x)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_MONOLITHIC(x)
		int res;

		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cublasStatus_t err;
				IF_FLOAT
					err = cublasIsamax(handle, x.numElements(), FPTR(x.dataPtr()), 1, &res);
				else IF_DOUBLE
					err = cublasIdamax(handle, x.numElements(), DPTR(x.dataPtr()), 1, &res);
				else IF_CX_FLOAT
					err = cublasIcamax(handle, x.numElements(), CPTR(x.dataPtr()), 1, &res);
				else IF_CX_DOUBLE
					err = cublasIzamax(handle, x.numElements(), ZPTR(x.dataPtr()), 1, &res);
				TEST_EXCEPTION(err)
			#else
				throw NotSupported;
			#endif
		}
		else
		{
			#ifdef KARTET_ADD_CBLAS_INTERFACE
				IF_FLOAT
					res = cblas_isamax(x.numElements(), FPTR(x.dataPtr()), 1);
				else IF_DOUBLE
					res = cblas_idamax(x.numElements(), DPTR(x.dataPtr()), 1);
				else IF_CX_FLOAT
					res = cblas_icamax(x.numElements(), CPTR(x.dataPtr()), 1);
				else IF_CX_DOUBLE
					res = cblas_izamax(x.numElements(), ZPTR(x.dataPtr()), 1);
			#else
				throw NotSupported;
			#endif
		}
		return res;
	}

	/**
	\return The index of the absolute minimum value.
	\param x Input data, must be monolithic.

	\f$ = \mbox{argmin}_k\{|x_k|\} \f$
	**/
	template<typename T, Location l>
	__host__ int BLASContext::Iamin(const Accessor<T,l>& x)
	{
		STATIC_ASSERT_VERBOSE(l==DeviceSide, LOCATION_NOT_SUPPORTED)
		ALLOWED_TYPES_VERIFICATION
		TEST_MONOLITHIC(x)
		int res;

		#ifdef __CUDACC__
			cublasStatus_t err;
			IF_FLOAT
				err = cublasIsamin(handle, x.numElements(), FPTR(x.dataPtr()), 1, &res);
			else IF_DOUBLE
				err = cublasIdamin(handle, x.numElements(), DPTR(x.dataPtr()), 1, &res);
			else IF_CX_FLOAT
				err = cublasIcamin(handle, x.numElements(), CPTR(x.dataPtr()), 1, &res);
			else IF_CX_DOUBLE
				err = cublasIzamin(handle, x.numElements(), ZPTR(x.dataPtr()), 1, &res);
			TEST_EXCEPTION(err)
		#else
			throw NotSupported;
		#endif
		return res;
	}

	/**
	\return The absolute sum of the data.
	\param x Input data, must be monolithic.

	\f$ = \sum_k |x_k| \f$
	**/
	template<typename T, Location l>
	__host__ typename Traits<T>::BaseType BLASContext::asum(const Accessor<T,l>& x)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_MONOLITHIC(x)
		typename Traits<T>::BaseType res;

		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cublasStatus_t err;
				IF_FLOAT
					err = cublasSasum(handle, x.numElements(), FPTR(x.dataPtr()), 1, FPTR(&res));
				else IF_DOUBLE
					err = cublasDasum(handle, x.numElements(), DPTR(x.dataPtr()), 1, DPTR(&res));
				else IF_CX_FLOAT
					err = cublasScasum(handle, x.numElements(), CPTR(x.dataPtr()), 1, FPTR(&res));
				else IF_CX_DOUBLE
					err = cublasDzasum(handle, x.numElements(), ZPTR(x.dataPtr()), 1, DPTR(&res));
				TEST_EXCEPTION(err)
			#else
				throw NotSupported;
			#endif
		}
		else
		{	
			#ifdef KARTET_ADD_CBLAS_INTERFACE
				IF_FLOAT
					res = cblas_sasum(x.numElements(), FPTR(x.dataPtr()), 1);
				else IF_DOUBLE
					res = cblas_dasum(x.numElements(), DPTR(x.dataPtr()), 1);
				else IF_CX_FLOAT
					res = cblas_scasum(x.numElements(), CPTR(x.dataPtr()), 1);
				else IF_CX_DOUBLE
					res = cblas_dzasum(x.numElements(), ZPTR(x.dataPtr()), 1);
			#else
				throw NotSupported;
			#endif
		}
		return res;
	}

	/**
	\return The dot product between the two vectors.
	\param x Left side, must be monolithic.
	\param y Right side, must be monolithic.
	\param conjugate If true, performs an Hermitian dot product.

	Real : 
	\f$ = \sum_k x_k y_k \f$

	Hermitian :
	\f$ = \sum_k x_k y_k^* \f$
	**/
	template<typename T, Location l>
	__host__ T BLASContext::dot(const Accessor<T,l>& x, const Accessor<T,l>& y, bool conjugate)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_MONOLITHIC(x)
		TEST_MONOLITHIC(y)
		T res;

		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cublasStatus_t err;
				IF_FLOAT
					err = cublasSdot(handle, x.numElements(), FPTR(x.dataPtr()), 1, FPTR(y.dataPtr()), 1, FPTR(&res));
				else IF_DOUBLE
					err = cublasDdot(handle, x.numElements(), DPTR(x.dataPtr()), 1, DPTR(y.dataPtr()), 1, DPTR(&res));
				else IF_CX_FLOAT
				{
					if(!conjugate)
						err = cublasCdotu(handle, x.numElements(), CPTR(x.dataPtr()), 1, CPTR(y.dataPtr()), 1, CPTR(&res));
					else
						err = cublasCdotc(handle, x.numElements(), CPTR(x.dataPtr()), 1, CPTR(y.dataPtr()), 1, CPTR(&res));
				}
				else IF_CX_DOUBLE
				{
					if(!conjugate)
						err = cublasZdotu(handle, x.numElements(), ZPTR(x.dataPtr()), 1, ZPTR(y.dataPtr()), 1, ZPTR(&res));
					else
						err = cublasZdotc(handle, x.numElements(), ZPTR(x.dataPtr()), 1, ZPTR(y.dataPtr()), 1, ZPTR(&res));
				}
				TEST_EXCEPTION(err)
			#else
				throw NotSupported;
			#endif
		}
		else
		{
			#ifdef KARTET_ADD_CBLAS_INTERFACE
				IF_FLOAT
					res = cblas_sdot(x.numElements(), FPTR(x.dataPtr()), 1, FPTR(y.dataPtr()), 1);
				else IF_DOUBLE
					res = cblas_ddot(x.numElements(), DPTR(x.dataPtr()), 1, DPTR(y.dataPtr()), 1);
				else
					throw NotSupported;
			#else
				throw NotSupported;
			#endif
		}
		return res;
	}

	/**
	\return The L2 norm of the vector.
	\param x Input data, must be monolithic.

	\f$ = \sqrt{\sum_k x_k^2} \f$
	**/
	template<typename T, Location l>
	__host__ typename Traits<T>::BaseType BLASContext::nrm2(const Accessor<T,l>& x)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_MONOLITHIC(x)
		typename Traits<T>::BaseType res;

		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cublasStatus_t err;
				IF_FLOAT
					err = cublasSnrm2(handle, x.numElements(), FPTR(x.dataPtr()), 1, FPTR(&res));
				else IF_DOUBLE
					err = cublasDnrm2(handle, x.numElements(), DPTR(x.dataPtr()), 1, DPTR(&res));
				else IF_CX_FLOAT
					err = cublasScnrm2(handle, x.numElements(), CPTR(x.dataPtr()), 1, FPTR(&res));
				else IF_CX_DOUBLE
					err = cublasDznrm2(handle, x.numElements(), ZPTR(x.dataPtr()), 1, DPTR(&res));
				TEST_EXCEPTION(err)
			#else
				throw NotSupported;
			#endif
		}
		else
		{
			#ifdef KARTET_ADD_CBLAS_INTERFACE
				IF_FLOAT
					res = cblas_snrm2(x.numElements(), FPTR(x.dataPtr()), 1);
				else IF_DOUBLE
					res = cblas_dnrm2(x.numElements(), DPTR(x.dataPtr()), 1);
				else IF_CX_FLOAT
					res = cblas_scnrm2(x.numElements(), CPTR(x.dataPtr()), 1);
				else IF_CX_DOUBLE
					res = cblas_dznrm2(x.numElements(), ZPTR(x.dataPtr()), 1);
			#else
				throw NotSupported;
			#endif
		}
		return res;
	}

	/**
	\brief Compute the generic matrix-vector multiplication.
	\param alpha Scalar.
	\param A matrix.
	\param op Operation.
	\param x Vector.
	\param beta Scalar.
	\param y Output vector.
	\return Accessor to result.

	\f$ y = \alpha A^{\mbox{op}} x + \beta y \f$
	**/
	template<typename T, Location l, typename TAlpha, typename TBeta>
	__host__ const Accessor<T,l>& BLASContext::gemv(const TAlpha& alpha, const Accessor<T,l>& A, MatrixOperation op, const Accessor<T,l>& x, const TBeta& beta, const Accessor<T,l>& y)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_MONOLITHIC(x)
		TEST_MONOLITHIC(y)
		TEST_SINGLE_SLICE(A)
		TEST_PRODUCT(A, op, x, OpNo, y)
		
		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cublasStatus_t err;
				IF_FLOAT
				{
					FCST(alpha)
					FCST(beta)
					err = cublasSgemv(handle, getCuBLASOperation(op), A.numRows(), A.numColumns(), &_alpha, FPTR(A.dataPtr()), A.columnsStride(), FPTR(x.dataPtr()), 1, &_beta, FPTR(y.dataPtr()), 1);
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					DCST(beta)
					err = cublasDgemv(handle, getCuBLASOperation(op), A.numRows(), A.numColumns(), &_alpha, DPTR(A.dataPtr()), A.columnsStride(), DPTR(x.dataPtr()), 1, &_beta, DPTR(y.dataPtr()), 1);
				}		
				else IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					err = cublasCgemv(handle, getCuBLASOperation(op), A.numRows(), A.numColumns(), &_alpha, CPTR(A.dataPtr()), A.columnsStride(), CPTR(x.dataPtr()), 1, &_beta, CPTR(y.dataPtr()), 1);
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					err = cublasZgemv(handle, getCuBLASOperation(op), A.numRows(), A.numColumns(), &_alpha, ZPTR(A.dataPtr()), A.columnsStride(), ZPTR(x.dataPtr()), 1, &_beta, ZPTR(y.dataPtr()), 1);
				}
				TEST_EXCEPTION(err)
			#else
				throw NotSupported;
			#endif
		}
		else
		{
			#ifdef KARTET_ADD_CBLAS_INTERFACE
				IF_FLOAT
				{
					FCST(alpha)
					FCST(beta)
					cblas_sgemv(CblasColMajor, getCBLASOperation(op), A.numRows(), A.numColumns(), _alpha, FPTR(A.dataPtr()), A.columnsStride(), FPTR(x.dataPtr()), 1, _beta, FPTR(y.dataPtr()), 1);
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					DCST(beta)
					cblas_dgemv(CblasColMajor, getCBLASOperation(op), A.numRows(), A.numColumns(), _alpha, DPTR(A.dataPtr()), A.columnsStride(), DPTR(x.dataPtr()), 1, _beta, DPTR(y.dataPtr()), 1);
				}		
				else IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					cblas_cgemv(CblasColMajor, getCBLASOperation(op), A.numRows(), A.numColumns(), &_alpha, CPTR(A.dataPtr()), A.columnsStride(), CPTR(x.dataPtr()), 1, &_beta, CPTR(y.dataPtr()), 1);
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					cblas_zgemv(CblasColMajor, getCBLASOperation(op), A.numRows(), A.numColumns(), &_alpha, ZPTR(A.dataPtr()), A.columnsStride(), ZPTR(x.dataPtr()), 1, &_beta, ZPTR(y.dataPtr()), 1);
				}
			#else
				throw NotSupported;
			#endif
		}
		return y;
	}

		/**
		\brief Compute the generic matrix-vector multiplication.
		\param A matrix.
		\param op Operation.
		\param x Vector.
		\param y Output vector.
		\return Accessor to result.

		\f$ y = A^{\mbox{op}} x\f$
		**/
		template<typename T, Location l>
		__host__ const Accessor<T,l>& BLASContext::gemv(const Accessor<T,l>& A, MatrixOperation op, const Accessor<T,l>& x, const Accessor<T,l>& y)
		{
			const T alpha = 1,
				beta = 0;
			return gemv(alpha, A, op, x, beta, y);
		}

		/**
		\brief Compute the generic matrix-vector multiplication.
		\param A matrix.
		\param x Vector.
		\param y Output vector.
		\return Accessor to result.

		\f$ y = A x \f$
		**/
		template<typename T, Location l>
		__host__ const Accessor<T,l>& BLASContext::gemv(const Accessor<T,l>& A, const Accessor<T,l>& x, const Accessor<T,l>& y)
		{
			const T alpha = 1,
				beta = 0;
			return gemv(alpha, A, OpNo, x, beta, y);
		}

	/**
	\brief Compute the rank one update.
	\param alpha Scalar.
	\param x Left side vector.
	\param y Right side vector.
	\param A output matrix.
	\param conjugate If true then performs hermitian rank one update.
	\return Accessor to result.

	If conjugate is false : 
	\f$ A = \alpha x y^{\intercal} + A \f$
	
	If conjugate is true : 
	\f$ A = \alpha x y^{H} + A \f$
	**/
	template<typename T, Location l, typename TAlpha>
	__host__ const Accessor<T,l>& BLASContext::ger(const TAlpha& alpha, const Accessor<T,l>& x, const Accessor<T,l>& y, const Accessor<T,l>& A, bool conjugate)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_MONOLITHIC(x)
		TEST_MONOLITHIC(y)
		TEST_SINGLE_SLICE(A)
		TEST_PRODUCT(x, OpNo, y, OpTr, A)
		
		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cublasStatus_t err;
				IF_FLOAT
				{
					FCST(alpha)
					err = cublasSger(handle, A.numRows(), A.numColumns(), &_alpha, FPTR(x.dataPtr()), 1, FPTR(y.dataPtr()), 1, FPTR(A.dataPtr()), A.columnsStride());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					err = cublasDger(handle, A.numRows(), A.numColumns(), &_alpha, DPTR(x.dataPtr()), 1, DPTR(y.dataPtr()), 1, DPTR(A.dataPtr()), A.columnsStride());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					if(!conjugate)
						err = cublasCgeru(handle, A.numRows(), A.numColumns(), &_alpha, CPTR(x.dataPtr()), 1, CPTR(y.dataPtr()), 1, CPTR(A.dataPtr()), A.columnsStride());
					else
						err = cublasCgerc(handle, A.numRows(), A.numColumns(), &_alpha, CPTR(x.dataPtr()), 1, CPTR(y.dataPtr()), 1, CPTR(A.dataPtr()), A.columnsStride());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					if(!conjugate)
						err = cublasZgeru(handle, A.numRows(), A.numColumns(), &_alpha, ZPTR(x.dataPtr()), 1, ZPTR(y.dataPtr()), 1, ZPTR(A.dataPtr()), A.columnsStride());
					else
						err = cublasZgerc(handle, A.numRows(), A.numColumns(), &_alpha, ZPTR(x.dataPtr()), 1, ZPTR(y.dataPtr()), 1, ZPTR(A.dataPtr()), A.columnsStride());
				}
				TEST_EXCEPTION(err)
			#else
				throw NotSupported;
			#endif
		}
		else
		{
			#ifdef KARTET_ADD_CBLAS_INTERFACE
				IF_FLOAT
				{
					FCST(alpha)
					cblas_sger(CblasColMajor, A.numRows(), A.numColumns(), _alpha, FPTR(x.dataPtr()), 1, FPTR(y.dataPtr()), 1, FPTR(A.dataPtr()), A.columnsStride());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					cblas_dger(CblasColMajor, A.numRows(), A.numColumns(), _alpha, DPTR(x.dataPtr()), 1, DPTR(y.dataPtr()), 1, DPTR(A.dataPtr()), A.columnsStride());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					if(!conjugate)
						cblas_cgeru(CblasColMajor, A.numRows(), A.numColumns(), _alpha, CPTR(x.dataPtr()), 1, CPTR(y.dataPtr()), 1, CPTR(A.dataPtr()), A.columnsStride());
					else
						cblas_cgerc(CblasColMajor, A.numRows(), A.numColumns(), _alpha, CPTR(x.dataPtr()), 1, CPTR(y.dataPtr()), 1, CPTR(A.dataPtr()), A.columnsStride());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					if(!conjugate)
						cblas_zgeru(CblasColMajor, A.numRows(), A.numColumns(), _alpha, ZPTR(x.dataPtr()), 1, ZPTR(y.dataPtr()), 1, ZPTR(A.dataPtr()), A.columnsStride());
					else
						cblas_zgerc(CblasColMajor, A.numRows(), A.numColumns(), _alpha, ZPTR(x.dataPtr()), 1, ZPTR(y.dataPtr()), 1, ZPTR(A.dataPtr()), A.columnsStride());
				}
			#else
				throw NotSupported;
			#endif
		}
		return A;
	}

		/**
		\brief Compute the rank one update.
		\param x Left side vector.
		\param y Right side vector.
		\param A output matrix.
		\param conjugate If true then performs hermitian rank one update.
		\return Accessor to result.

		If conjugate is false : 
		\f$ A = \alpha x y^{\intercal} + A \f$
	
		If conjugate is true : 
		\f$ A = \alpha x y^{H} + A \f$
		**/
		template<typename T, Location l>
		__host__ const Accessor<T,l>& BLASContext::ger(const Accessor<T,l>& x, const Accessor<T,l>& y, const Accessor<T,l>& A, bool conjugate)
		{
			const T alpha = 1;
			return ger(alpha, x, y, A, conjugate);
		}

	/**
	\brief Symmetric matrix-vector product.
	\param alpha Scalar.
	\param uplo Fill mode.
	\param A Symmetric matrix.
	\param x Vector.
	\param beta Scalar.
	\param y Output vector.
	\return Accessor to result.

	\f$ y = \alpha A x + \beta y \f$
	**/
	template<typename T, Location l, typename TAlpha, typename TBeta>
	__host__ const Accessor<T,l>& BLASContext::symv(const TAlpha& alpha, MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x, const TBeta& beta, const Accessor<T,l>& y)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_MONOLITHIC(x)
		TEST_MONOLITHIC(y)
		TEST_SINGLE_SLICE(A)
		TEST_PRODUCT(A, OpNo, x, OpNo, y)

		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cublasStatus_t err;
				IF_FLOAT
				{
					FCST(alpha)
					FCST(beta)
					err = cublasSsymv(handle, getCuBLASFillMode(uplo), A.numRows(), &_alpha, FPTR(A.dataPtr()), A.columnsStride(), FPTR(x.dataPtr()), 1, &_beta, FPTR(y.dataPtr()), 1);
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					DCST(beta)
					err = cublasDsymv(handle, getCuBLASFillMode(uplo), A.numRows(), &_alpha, DPTR(A.dataPtr()), A.columnsStride(), DPTR(x.dataPtr()), 1, &_beta, DPTR(y.dataPtr()), 1);
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					err = cublasCsymv(handle, getCuBLASFillMode(uplo), A.numRows(), &_alpha, CPTR(A.dataPtr()), A.columnsStride(), CPTR(x.dataPtr()), 1, &_beta, CPTR(y.dataPtr()), 1);
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					err = cublasZsymv(handle, getCuBLASFillMode(uplo), A.numRows(), &_alpha, ZPTR(A.dataPtr()), A.columnsStride(), ZPTR(x.dataPtr()), 1, &_beta, ZPTR(y.dataPtr()), 1);
				}
				TEST_EXCEPTION(err)
			#else
				throw NotSupported;
			#endif
		}
		else
		{
			#ifdef KARTET_ADD_CBLAS_INTERFACE
				IF_FLOAT
				{
					FCST(alpha)
					FCST(beta)
					cblas_ssymv(CblasColMajor, getCBLASFillMode(uplo), A.numRows(), _alpha, FPTR(A.dataPtr()), A.columnsStride(), FPTR(x.dataPtr()), 1, _beta, FPTR(y.dataPtr()), 1);
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					DCST(beta)
					cblas_dsymv(CblasColMajor, getCBLASFillMode(uplo), A.numRows(), _alpha, DPTR(A.dataPtr()), A.columnsStride(), DPTR(x.dataPtr()), 1, _beta, DPTR(y.dataPtr()), 1);
				}
				else
					throw NotSupported;
			#else
				throw NotSupported;
			#endif
		}
		return y;
	}

		/**
		\brief Symmetric matrix-vector product.
		\param uplo Fill mode.
		\param A Symmetric matrix.
		\param x Vector.
		\param y Output vector.
		\return Accessor to result.

		\f$ y =  A x \f$
		**/
		template<typename T, Location l>
		__host__ const Accessor<T,l>& BLASContext::symv(MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x, const Accessor<T,l>& y)
		{
			const T alpha = 1,
				beta = 0;
			return symv(alpha, uplo, A, x, beta, y);
		}
	
	/**
	\brief Symmetric rank one update.
	\param alpha Scalar.
	\param uplo Fill mode.
	\param A Symmetric matrix (input and output).
	\param x Vector.
	\return Accessor to result.

	\f$ A = \alpha x x^\intercal + A \f$
	**/
	template<typename T, Location l, typename TAlpha>
	__host__ const Accessor<T,l>& BLASContext::syr(const TAlpha& alpha, MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_MONOLITHIC(x)
		TEST_SINGLE_SLICE(A)
		TEST_PRODUCT(x, OpNo, x, OpTr, A)
		
		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cublasStatus_t err;
				IF_FLOAT
				{
					FCST(alpha)
					err = cublasSsyr(handle, getCuBLASFillMode(uplo), A.numRows(), &_alpha, FPTR(x.dataPtr()), 1, FPTR(A.gePtr()), A.columnsStride());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					err = cublasDsyr(handle, getCuBLASFillMode(uplo), A.numRows(), &_alpha, DPTR(x.dataPtr()), 1, DPTR(A.gePtr()), A.columnsStride());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					err = cublasCsyr(handle, getCuBLASFillMode(uplo), A.numRows(), &_alpha, CPTR(x.dataPtr()), 1, CPTR(A.gePtr()), A.columnsStride());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					err = cublasZsyr(handle, getCuBLASFillMode(uplo), A.numRows(), &_alpha, ZPTR(x.dataPtr()), 1, ZPTR(A.gePtr()), A.columnsStride());
				}
				TEST_EXCEPTION(err)
			#else
				throw NotSupported;
			#endif
		}
		else
		{
			#ifdef KARTET_ADD_CBLAS_INTERFACE
				IF_FLOAT
				{
					FCST(alpha)
					cblas_ssyr(CblasColMajor, getCBLASFillMode(uplo), A.numRows(), _alpha, FPTR(x.dataPtr()), 1, FPTR(A.gePtr()), A.columnsStride());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					cblas_dsyr(CblasColMajor, getCBLASFillMode(uplo), A.numRows(), _alpha, DPTR(x.dataPtr()), 1, DPTR(A.gePtr()), A.columnsStride());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					cblas_csyr(CblasColMajor, getCBLASFillMode(uplo), A.numRows(), &_alpha, CPTR(x.dataPtr()), 1, CPTR(A.gePtr()), A.columnsStride());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					cblas_zsyr(CblasColMajor, getCBLASFillMode(uplo), A.numRows(), &_alpha, ZPTR(x.dataPtr()), 1, ZPTR(A.gePtr()), A.columnsStride());
				}
			#else
				throw NotSupported;
			#endif
		}
		return A;
	}

		/**
		\brief Symmetric rank one update.
		\param uplo Fill mode.
		\param A Symmetric matrix (input and output).
		\param x Vector.
		\return Accessor to result.

		\f$ A = x x^\intercal + A \f$
		**/
		template<typename T, Location l>
		__host__ const Accessor<T,l>& BLASContext::syr(MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x)
		{
			const T alpha = 1;
			return syr(alpha, uplo, A, x);
		}

	/**
	\brief Symmetric rank two update.
	\param alpha Scalar.
	\param uplo Fill mode.
	\param A Symmetric matrix (and output).
	\param x First vector.
	\param y Second vector.
	\return Accessor to result.

	\f$ A = \alpha(xy^\intercal + yx^\intercal) + A \f$
	**/
	template<typename T, Location l, typename TAlpha>
	__host__ const Accessor<T,l>& BLASContext::syr2(const TAlpha& alpha, MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x, const Accessor<T,l>& y)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_MONOLITHIC(x)
		TEST_MONOLITHIC(y)
		TEST_SINGLE_SLICE(A)
		TEST_PRODUCT(x, OpNo, y, OpTr, A)
		TEST_PRODUCT(y, OpNo, x, OpTr, A)

		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cublasStatus_t err;
				IF_FLOAT
				{
					FCST(alpha)
					err = cublasSsyr2(handle, getCuBLASFillMode(uplo), A.numRows(), &_alpha, FPTR(x.dataPtr()), 1, FPTR(y.dataPtr()), 1, FPTR(A.gePtr()), A.columnsStride());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					err = cublasDsyr2(handle, getCuBLASFillMode(uplo), A.numRows(), &_alpha, DPTR(x.dataPtr()), 1, DPTR(y.dataPtr()), 1, DPTR(A.gePtr()), A.columnsStride());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					err = cublasCsyr2(handle, getCuBLASFillMode(uplo), A.numRows(), &_alpha, CPTR(x.dataPtr()), 1, CPTR(y.dataPtr()), 1, CPTR(A.gePtr()), A.columnsStride());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					err = cublasZsyr2(handle, getCuBLASFillMode(uplo), A.numRows(), &_alpha, ZPTR(x.dataPtr()), 1, ZPTR(y.dataPtr()), 1, ZPTR(A.gePtr()), A.columnsStride());
				}
				TEST_EXCEPTION(err)
			#else
				throw NotSupported;
			#endif
		}
		else
		{
			#ifdef KARTET_ADD_CBLAS_INTERFACE
				IF_FLOAT
				{
					FCST(alpha)
					cblas_ssyr2(CblasColMajor, getCBLASFillMode(uplo), A.numRows(), _alpha, FPTR(x.dataPtr()), 1, FPTR(y.dataPtr()), 1, FPTR(A.gePtr()), A.columnsStride());
				}
				else IF_DOUBLE
				{
					FCST(alpha)
					cblas_ssyr2(CblasColMajor, getCBLASFillMode(uplo), A.numRows(), _alpha, DPTR(x.dataPtr()), 1, DPTR(y.dataPtr()), 1, DPTR(A.gePtr()), A.columnsStride());
				}
				else
					throw NotSupported;
			#else
				throw NotSupported;
			#endif
		}
		return A;
	}

		/**
		\brief Symmetric rank two update.
		\param uplo Fill mode.
		\param A Symmetric matrix (and output).
		\param x First vector.
		\param y Second vector.
		\return Accessor to result.

		\f$ A = (xy^\intercal + yx^\intercal) + A \f$
		**/
		template<typename T, Location l>
		__host__ const Accessor<T,l>& BLASContext::syr2(MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x, const Accessor<T,l>& y)
		{
			const T alpha = 1;
			return syr2(alpha, uplo, A, x, y);
		}

	/**
	\brief Triangular matrix-vector product.
	\param uplo Fill mode.
	\param diag Diagonal type.
	\param A Triangular matrix.
	\param op Operation.
	\param x Input/output vector.
	\return Accessor to result.

	\f$ x' = A^{\mbox{op}} x \f$
	**/
	template<typename T, Location l>
	__host__ const Accessor<T,l>& BLASContext::trmv(MatrixFillMode uplo, MatrixDiagType diag, const Accessor<T,l>& A, MatrixOperation op, const Accessor<T,l>& x)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_MONOLITHIC(x)
		TEST_SINGLE_SLICE(A)
		TEST_PRODUCT(A, op, x, OpNo, x)

		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cublasStatus_t err;
				IF_FLOAT
					err = cublasStrmv(handle, getCuBLASFillMode(uplo), getCuBLASOperation(op), getCuBLASDiagType(diag), A.numRows(), FPTR(A.gePtr()), A.columnsStride(), FPTR(x.dataPtr()), 1);
				else IF_DOUBLE
					err = cublasStrmv(handle, getCuBLASFillMode(uplo), getCuBLASOperation(op), getCuBLASDiagType(diag), A.numRows(), DPTR(A.gePtr()), A.columnsStride(), DPTR(x.dataPtr()), 1);
				else IF_CX_FLOAT
					err = cublasStrmv(handle, getCuBLASFillMode(uplo), getCuBLASOperation(op), getCuBLASDiagType(diag), A.numRows(), CPTR(A.gePtr()), A.columnsStride(), CPTR(x.dataPtr()), 1);
				else IF_CX_DOUBLE
					err = cublasStrmv(handle, getCuBLASFillMode(uplo), getCuBLASOperation(op), getCuBLASDiagType(diag), A.numRows(), ZPTR(A.gePtr()), A.columnsStride(), ZPTR(x.dataPtr()), 1);
				TEST_EXCEPTION(err)
			#else
				throw NotSupported;
			#endif
		}
		else
		{
			#ifdef KARTET_ADD_CBLAS_INTERFACE
				IF_FLOAT
					cblas_strmv(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(op), getCBLASDiagType(diag), A.numRows(), FPTR(A.gePtr()), A.columnsStride(), FPTR(x.dataPtr()), 1);
				else IF_DOUBLE
					cblas_dtrmv(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(op), getCBLASDiagType(diag), A.numRows(), DPTR(A.gePtr()), A.columnsStride(), DPTR(x.dataPtr()), 1);
				else IF_CX_FLOAT
					cblas_ctrmv(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(op), getCBLASDiagType(diag), A.numRows(), CPTR(A.gePtr()), A.columnsStride(), CPTR(x.dataPtr()), 1);
				else IF_CX_DOUBLE
					cblas_ztrmv(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(op), getCBLASDiagType(diag), A.numRows(), ZPTR(A.gePtr()), A.columnsStride(), ZPTR(x.dataPtr()), 1);
			#else
				throw NotSupported;
			#endif
		}
		return x;
	}

	/**
	\brief Solve triangular system.
	\param uplo Fill mode.
	\param diag Diagonal type.
	\param A Triangular matrix.
	\param op Operation.
	\param x Input/output vector.
	\return Accessor to result.

	\f$ x = A^{\mbox{op}} x' \f$
	**/
	template<typename T, Location l>
	__host__ const Accessor<T,l>& BLASContext::trsv(MatrixFillMode uplo, MatrixDiagType diag, const Accessor<T,l>& A, MatrixOperation op, const Accessor<T,l>& x)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_MONOLITHIC(x)
		TEST_SINGLE_SLICE(A)
		TEST_PRODUCT(A, op, x, OpNo, x)

		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cublasStatus_t err;
				IF_FLOAT
					err = cublasStrsv(handle, getCuBLASFillMode(uplo), getCuBLASOperation(op), getCuBLASDiagType(diag), A.numRows(), FPTR(A.gePtr()), A.columnsStride(), FPTR(x.dataPtr()), 1);
				else IF_DOUBLE
					err = cublasStrsv(handle, getCuBLASFillMode(uplo), getCuBLASOperation(op), getCuBLASDiagType(diag), A.numRows(), DPTR(A.gePtr()), A.columnsStride(), DPTR(x.dataPtr()), 1);
				else IF_CX_FLOAT
					err = cublasStrsv(handle, getCuBLASFillMode(uplo), getCuBLASOperation(op), getCuBLASDiagType(diag), A.numRows(), CPTR(A.gePtr()), A.columnsStride(), CPTR(x.dataPtr()), 1);
				else IF_CX_DOUBLE
					err = cublasStrsv(handle, getCuBLASFillMode(uplo), getCuBLASOperation(op), getCuBLASDiagType(diag), A.numRows(), ZPTR(A.gePtr()), A.columnsStride(), ZPTR(x.dataPtr()), 1);
				TEST_EXCEPTION(err)
			#else
				throw NotSupported;
			#endif
		}
		else
		{
			#ifdef KARTET_ADD_CBLAS_INTERFACE
				IF_FLOAT
					cblas_strsv(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(op), getCBLASDiagType(diag), A.numRows(), FPTR(A.gePtr()), A.columnsStride(), FPTR(x.dataPtr()), 1);
				else IF_DOUBLE
					cblas_dtrsv(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(op), getCBLASDiagType(diag), A.numRows(), DPTR(A.gePtr()), A.columnsStride(), DPTR(x.dataPtr()), 1);
				else IF_CX_FLOAT
					cblas_ctrsv(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(op), getCBLASDiagType(diag), A.numRows(), CPTR(A.gePtr()), A.columnsStride(), CPTR(x.dataPtr()), 1);
				else IF_CX_DOUBLE
					cblas_ztrsv(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(op), getCBLASDiagType(diag), A.numRows(), ZPTR(A.gePtr()), A.columnsStride(), ZPTR(x.dataPtr()), 1);
			#else
				throw NotSupported;
			#endif
		}
		return x;
	}

	/**
	\brief Hermitian matrix-vector multiplication.
	\param alpha Scalar.
	\param uplo Fill mode.
	\param A Hermitian matrix.
	\param x Vector.
	\param beta Scalar.
	\param y Output vector.
	\return Accessor to result.

	\f$ y = \alpha A x + \beta y \f$
	**/
	template<typename T, Location l, typename TAlpha, typename TBeta>
	__host__ const Accessor<T,l>& BLASContext::hemv(const TAlpha& alpha, MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x, const TBeta& beta, const Accessor<T,l>& y)
	{
		ALLOWED_TYPES_VERIFICATION
		TYPE_MUST_BE_COMPLEX
		TEST_MONOLITHIC(x)
		TEST_SINGLE_SLICE(A)
		TEST_PRODUCT(A, OpNo, x, OpNo, y)

		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cublasStatus_t err;
				IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					err = cublasChemv(handle, getCuBLASFillMode(uplo), A.numRows(), &_alpha, CPTR(A.gePtr()), A.columnsStride(), CPTR(x.dataPtr()), 1, &_beta, CPTR(y.dataPtr()), 1);
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					err = cublasZhemv(handle, getCuBLASFillMode(uplo), A.numRows(), &_alpha, ZPTR(A.gePtr()), A.columnsStride(), ZPTR(x.dataPtr()), 1, &_beta, ZPTR(y.dataPtr()), 1);
				}
				TEST_EXCEPTION(err)
			#else
				throw NotSupported;
			#endif
		}
		else
		{
			#ifdef KARTET_ADD_CBLAS_INTERFACE
				IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					cblas_chemv(CblasColMajor, getCBLASFillMode(uplo), A.numRows(), &_alpha, CPTR(A.gePtr()), A.columnsStride(), CPTR(x.dataPtr()), 1, &_beta, CPTR(y.dataPtr()), 1);
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					cblas_zhemv(CblasColMajor, getCBLASFillMode(uplo), A.numRows(), &_alpha, ZPTR(A.gePtr()), A.columnsStride(), ZPTR(x.dataPtr()), 1, &_beta, ZPTR(y.dataPtr()), 1);
				}
			#else
				throw NotSupported;
			#endif
		}
		return y;
	}

		/**
		\brief Hermitian matrix-vector multiplication.
		\param uplo Fill mode.
		\param A Hermitian matrix.
		\param x Vector.
		\param y Output vector.
		\return Accessor to result.

		\f$ y = A x + y \f$
		**/
		template<typename T, Location l>
		__host__ const Accessor<T,l>& BLASContext::hemv(MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x, const Accessor<T,l>& y)
		{
			const T alpha = 1,
				beta = 0;
			return hemv(alpha, uplo, A, x, beta, y);
		}

	/**
	\brief Hermitian rank one update.
	\param alpha Scalar.
	\param uplo Fill mode.
	\param A An hermitian matrix (input and output).
	\param x Vector.
	\return Accessor to result.

	\f$ A = \alpha x x^{\mbox{H}} + A \f$
	**/
	template<typename T, Location l, typename TAlpha>
	__host__ const Accessor<T,l>& BLASContext::her(const TAlpha& alpha, MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x)
	{
		ALLOWED_TYPES_VERIFICATION
		TYPE_MUST_BE_COMPLEX
		TEST_MONOLITHIC(x)
		TEST_SINGLE_SLICE(A)
		TEST_PRODUCT(x, OpNo, x, OpHr, A)

		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cublasStatus_t err;
				IF_CX_FLOAT
				{
					CCST(alpha)
					err = cublasCher(handle, getCuBLASFillMode(uplo), A.numRows(), &_alpha, CPTR(x.dataPtr()), 1, CPTR(A.gePtr()), A.columnsStride());
				}                
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					err = cublasZher(handle, getCuBLASFillMode(uplo), A.numRows(), &_alpha, ZPTR(x.dataPtr()), 1, ZPTR(A.gePtr()), A.columnsStride());
				}
				TEST_EXCEPTION(err)
			#else
				throw NotSupported;
			#endif
		}
		else
		{
			#ifdef KARTET_ADD_CBLAS_INTERFACE
				IF_CX_FLOAT
				{
					CCST(alpha)
					cblas_cher(CblasColMajor, getCBLASFillMode(uplo), A.numRows(), &_alpha, CPTR(x.dataPtr()), 1, CPTR(A.gePtr()), A.columnsStride());
				}                
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					cblas_zher(CblasColMajor, getCBLASFillMode(uplo), A.numRows(), &_alpha, ZPTR(x.dataPtr()), 1, ZPTR(A.gePtr()), A.columnsStride());
				}
			#else
				throw NotSupported;
			#endif
		}
		return A;
	}

		/**
		\brief Hermitian rank one update.
		\param uplo Fill mode.
		\param A An hermitian matrix (input and output).
		\param x Vector.
		\return Accessor to result.

		\f$ A = x x^{\mbox{H}} + A \f$
		**/
		template<typename T, Location l>
		__host__ const Accessor<T,l>& BLASContext::her(MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x)
		{
			const T alpha = 1;
			return her(alpha, uplo, A, x);
		}

	/**
	\brief Hermitian rank two update.
	\param alpha Scalar.
	\param uplo Fill mode.
	\param A An hermitian matrix (input and output).
	\param x First vector.
	\param y Second vector.
	\return Accessor to result.

	\f$ A = \alpha x y^{\mbox{H}} + \alpha^* y x^{\mbox{H}} + A \f$
	**/
	template<typename T, Location l, typename TAlpha>
	__host__ const Accessor<T,l>& BLASContext::her2(const TAlpha& alpha, MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x, const Accessor<T,l>& y)
	{
		ALLOWED_TYPES_VERIFICATION
		TYPE_MUST_BE_COMPLEX
		TEST_MONOLITHIC(x)
		TEST_SINGLE_SLICE(A)
		TEST_PRODUCT(x, OpNo, y, OpHr, A)
		TEST_PRODUCT(y, OpNo, x, OpHr, A)

		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cublasStatus_t err;
				IF_CX_FLOAT
				{
					CCST(alpha)
					err = cublasCher2(handle, getCuBLASFillMode(uplo), A.numRows(), &_alpha, CPTR(x.dataPtr()), 1, CPTR(y.dataPtr()), 1, CPTR(A.gePtr()), A.columnsStride());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					err = cublasCher2(handle, getCuBLASFillMode(uplo), A.numRows(), &_alpha, ZPTR(x.dataPtr()), 1, ZPTR(y.dataPtr()), 1, CPTR(A.gePtr()), A.columnsStride());
				}
				TEST_EXCEPTION(err)
			#else
				throw NotSupported;
			#endif
		}
		else
		{
			#ifdef KARTET_ADD_CBLAS_INTERFACE
				IF_CX_FLOAT
				{
					CCST(alpha)
					cblas_cher2(CblasColMajor, getCBLASFillMode(uplo), A.numRows(), &_alpha, CPTR(x.dataPtr()), 1, CPTR(y.dataPtr()), 1, CPTR(A.gePtr()), A.columnsStride());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					cblas_zher2(CblasColMajor, getCBLASFillMode(uplo), A.numRows(), &_alpha, ZPTR(x.dataPtr()), 1, ZPTR(y.dataPtr()), 1, ZPTR(A.gePtr()), A.columnsStride());
				}
			#else
				throw NotSupported;
			#endif
		}
		return A;
	}

		/**
		\brief Hermitian rank two update.
		\param uplo Fill mode.
		\param A An hermitian matrix (input and output).
		\param x First vector.
		\param y Second vector.
		\return Accessor to result.

		\f$ A = x y^{\mbox{H}} + \alpha^* y x^{\mbox{H}} + A \f$
		**/
		template<typename T, Location l>
		__host__ const Accessor<T,l>& BLASContext::her2(MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x, const Accessor<T,l>& y)
		{
			const T alpha = 1;
			return her2(alpha, uplo, A, x, y);
		}

	/**
	\brief Generic matrix-matrix product.
	\param alpha Scalar.
	\param A Left side matrix.
	\param opa Operation on left side.
	\param B Right side matrix.
	\param opb Operation on right side.
	\param beta Scalar.
	\param C Output matrix.
	\return Accessor to result.
	
	\f$ C = \alpha A^{\mbox{opa}} B^{\mbox{opb}} + \beta C \f$
	**/
	template<typename T, Location l, typename TAlpha, typename TBeta>
	__host__ const Accessor<T,l>& BLASContext::gemm(const TAlpha& alpha, const Accessor<T,l>& A, MatrixOperation opa, const Accessor<T,l>& B, MatrixOperation opb, const TBeta& beta, const Accessor<T,l>& C)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(B)
		TEST_SINGLE_SLICE(C)
		TEST_PRODUCT(A, opa, B, opb, C)
		const int k = (opa==OpTr || opa==OpHr) ? A.numRows() : A.numColumns();

		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cublasStatus_t err;			
				IF_FLOAT
				{
					FCST(alpha)
					FCST(beta)
					err = cublasSgemm(handle, getCuBLASOperation(opa), getCuBLASOperation(opb), C.numRows(), C.numColumns(), k, &_alpha, FPTR(A.dataPtr()), A.columnsStride(), FPTR(B.dataPtr()), B.columnsStride(), &_beta, FPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					DCST(beta)
					err = cublasDgemm(handle, getCuBLASOperation(opa), getCuBLASOperation(opb), C.numRows(), C.numColumns(), k, &_alpha, DPTR(A.dataPtr()), A.columnsStride(), DPTR(B.dataPtr()), B.columnsStride(), &_beta, DPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					err = cublasCgemm(handle, getCuBLASOperation(opa), getCuBLASOperation(opb), C.numRows(), C.numColumns(), k, &_alpha, CPTR(A.dataPtr()), A.columnsStride(), CPTR(B.dataPtr()), B.columnsStride(), &_beta, CPTR(C.dataPtr()), C.columnsStride());
				}		
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					err = cublasZgemm(handle, getCuBLASOperation(opa), getCuBLASOperation(opb), C.numRows(), C.numColumns(), k, &_alpha, ZPTR(A.dataPtr()), A.columnsStride(), ZPTR(B.dataPtr()), B.columnsStride(), &_beta, ZPTR(C.dataPtr()), C.columnsStride());
				}
				TEST_EXCEPTION(err)
			#else
				throw NotSupported;
			#endif
		}
		else
		{
			#ifdef KARTET_ADD_CBLAS_INTERFACE
				IF_FLOAT
				{
					FCST(alpha)
					FCST(beta)
					cblas_sgemm(CblasColMajor, getCBLASOperation(opa), getCBLASOperation(opb), C.numRows(), C.numColumns(), k, _alpha, FPTR(A.dataPtr()), A.columnsStride(), FPTR(B.dataPtr()), B.columnsStride(), _beta, FPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					DCST(beta)
					cblas_dgemm(CblasColMajor, getCBLASOperation(opa), getCBLASOperation(opb), C.numRows(), C.numColumns(), k, _alpha, DPTR(A.dataPtr()), A.columnsStride(), DPTR(B.dataPtr()), B.columnsStride(), _beta, DPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					cblas_cgemm(CblasColMajor, getCBLASOperation(opa), getCBLASOperation(opb), C.numRows(), C.numColumns(), k, &_alpha, CPTR(A.dataPtr()), A.columnsStride(), CPTR(B.dataPtr()), B.columnsStride(), &_beta, CPTR(C.dataPtr()), C.columnsStride());
				}		
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					cblas_zgemm(CblasColMajor, getCBLASOperation(opa), getCBLASOperation(opb), C.numRows(), C.numColumns(), k, &_alpha, ZPTR(A.dataPtr()), A.columnsStride(), ZPTR(B.dataPtr()), B.columnsStride(), &_beta, ZPTR(C.dataPtr()), C.columnsStride());
				}
			#else
				throw NotSupported;
			#endif
		}
		return C;
	}

		/**
		\brief Generic matrix-matrix product.
		\param A Left side matrix.
		\param opa Operation on left side.
		\param B Right side matrix.
		\param opb Operation on right side.
		\param C Output matrix.
		\return Accessor to result.
	
		\f$ C = A^{\mbox{opa}} B^{\mbox{opb}}\f$
		**/
		template<typename T, Location l>
		__host__ const Accessor<T,l>& BLASContext::gemm(const Accessor<T,l>& A, MatrixOperation opa, const Accessor<T,l>& B, MatrixOperation opb, const Accessor<T,l>& C)
		{
			const T alpha = 1,
				beta = 0;
			return gemm(alpha, A, opa, B, opb, beta, C);
		}

		/**
		\brief Generic matrix-matrix product.
		\param A Left side matrix.
		\param B Right side matrix.
		\param C Output matrix.
		\return Accessor to result.
	
		\f$ C = A B \f$
		**/
		template<typename T, Location l>
		__host__ const Accessor<T,l>& BLASContext::gemm(const Accessor<T,l>& A, const Accessor<T,l>& B, const Accessor<T,l>& C)
		{
			const T alpha = 1,
				beta = 0;
			return gemm(alpha, A, OpNo, B, OpNo, beta, C);
		}

	/**
	\brief Symmetric matrix product.
	\param side Operation side.
	\param alpha Scalar.
	\param uplo Fill mode.
	\param A Symmetric matrix.
	\param B Matrix.
	\param beta Scalar.
	\param C Output matrix.
	\return Accessor to result.

	If side is left : 
	\f$ C = \alpha A B + \beta C \f$
	If side is right : 
	\f$ C = \alpha B A + \beta C \f$
	**/
	template<typename T, Location l, typename TAlpha, typename TBeta>
	__host__ const Accessor<T,l>& BLASContext::symm(MatrixSideMode side, const TAlpha& alpha, MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& B, const TBeta& beta, const Accessor<T,l>& C)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(B)
		TEST_SINGLE_SLICE(C)
		if(side==MatrixLeftSide)
			TEST_PRODUCT(A, OpNo, B, OpNo, C)
		else // MatrixRightSide
			TEST_PRODUCT(B, OpNo, A, OpNo, C)
	
		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cublasStatus_t err;
				IF_FLOAT
				{
					FCST(alpha)
					FCST(beta)
					err = cublasSsymm(handle, getCuBLASSideMode(side), getCuBLASFillMode(uplo), B.numRows(), C.numColumns(), &_alpha, FPTR(A.dataPtr()), A.columnsStride(), FPTR(B.dataPtr()), B.columnsStride(), &_beta, FPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					DCST(beta)
					err = cublasDsymm(handle, getCuBLASSideMode(side), getCuBLASFillMode(uplo), B.numRows(), C.numColumns(), &_alpha, DPTR(A.dataPtr()), A.columnsStride(), DPTR(B.dataPtr()), B.columnsStride(), &_beta, DPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					err = cublasCsymm(handle, getCuBLASSideMode(side), getCuBLASFillMode(uplo), B.numRows(), C.numColumns(), &_alpha, CPTR(A.dataPtr()), A.columnsStride(), CPTR(B.dataPtr()), B.columnsStride(), &_beta, CPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					err = cublasZsymm(handle, getCuBLASSideMode(side), getCuBLASFillMode(uplo), B.numRows(), C.numColumns(), &_alpha, ZPTR(A.dataPtr()), A.columnsStride(), ZPTR(B.dataPtr()), B.columnsStride(), &_beta, ZPTR(C.dataPtr()), C.columnsStride());
				}
				TEST_EXCEPTION(err)
			#else
				throw NotSupported;
			#endif
		}
		else
		{
			#ifdef KARTET_ADD_CBLAS_INTERFACE
				IF_FLOAT
				{
					FCST(alpha)
					FCST(beta)
					cblas_ssymm(CblasColMajor, getCBLASSideMode(side), getCBLASFillMode(uplo), B.numRows(), C.numColumns(), _alpha, FPTR(A.dataPtr()), A.columnsStride(), FPTR(B.dataPtr()), B.columnsStride(), _beta, FPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					DCST(beta)
					cblas_dsymm(CblasColMajor, getCBLASSideMode(side), getCBLASFillMode(uplo), B.numRows(), C.numColumns(), _alpha, DPTR(A.dataPtr()), A.columnsStride(), DPTR(B.dataPtr()), B.columnsStride(), _beta, DPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					cblas_csymm(CblasColMajor, getCBLASSideMode(side), getCBLASFillMode(uplo), B.numRows(), C.numColumns(), &_alpha, CPTR(A.dataPtr()), A.columnsStride(), CPTR(B.dataPtr()), B.columnsStride(), &_beta, CPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					cblas_zsymm(CblasColMajor, getCBLASSideMode(side), getCBLASFillMode(uplo), B.numRows(), C.numColumns(), &_alpha, ZPTR(A.dataPtr()), A.columnsStride(), ZPTR(B.dataPtr()), B.columnsStride(), &_beta, ZPTR(C.dataPtr()), C.columnsStride());
				}
			#else
				throw NotSupported;
			#endif
		}
		return C;
	}

		/**
		\brief Symmetric matrix product.
		\param side Operation side.
		\param uplo Fill mode.
		\param A Symmetric matrix.
		\param B Matrix.
		\param C Output matrix.
		\return Accessor to result.

		If side is left : 
		\f$ C = A B \f$
		If side is right : 
		\f$ C = B A \f$
		**/
		template<typename T, Location l>
		__host__ const Accessor<T,l>& BLASContext::symm(MatrixSideMode side, MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& B, const Accessor<T,l>& C)
		{
			const T alpha = 1,
				beta = 0;
			return symm(side, alpha, uplo, A, B, beta, C);
		}

	/**
	\brief Symmetric rank-k update.
	\param alpha Scalar.
	\param A Matrix.
	\param opa Operation on matrix A.
	\param beta Scalar.
	\param uplo Output fill mode.
	\param C Output matrix.
	\return Accessor to result.

	\f$ C = \alpha A^{\mbox{opa}} {A^{\mbox{opa}}}^\intercal + \beta C \f$
	**/
	template<typename T, Location l, typename TAlpha, typename TBeta>
	__host__ const Accessor<T,l>& BLASContext::syrk(const TAlpha& alpha, const Accessor<T,l>& A, MatrixOperation opa, const TBeta& beta, MatrixFillMode uplo, const Accessor<T,l>& C)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(C)
		if(opa==OpNo)
			TEST_PRODUCT(A, OpNo, A, OpTr, C)
		else
			TEST_PRODUCT(A, OpTr, A, OpNo, C)
		const int k = (opa==OpTr || opa==OpHr) ? A.numRows() : A.numColumns();

		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cublasStatus_t err;		
				IF_FLOAT
				{
					FCST(alpha)
					FCST(beta)
					err = cublasSsyrk(handle, getCuBLASFillMode(uplo), getCuBLASOperation(opa), C.numRows(), k, &_alpha, FPTR(A.dataPtr()), A.columnsStride(), &_beta, FPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					DCST(beta)
					err = cublasDsyrk(handle, getCuBLASFillMode(uplo), getCuBLASOperation(opa), C.numRows(), k, &_alpha, DPTR(A.dataPtr()), A.columnsStride(), &_beta, DPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					err = cublasCsyrk(handle, getCuBLASFillMode(uplo), getCuBLASOperation(opa), C.numRows(), k, &_alpha, CPTR(A.dataPtr()), A.columnsStride(), &_beta, CPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					err = cublasZsyrk(handle, getCuBLASFillMode(uplo), getCuBLASOperation(opa), C.numRows(), k, &_alpha, ZPTR(A.dataPtr()), A.columnsStride(), &_beta, ZPTR(C.dataPtr()), C.columnsStride());
				}
				TEST_EXCEPTION(err)
			#else
				throw NotSupported;
			#endif
		}
		else
		{
			#ifdef KARTET_ADD_CBLAS_INTERFACE
				IF_FLOAT
				{
					FCST(alpha)
					FCST(beta)
					cblas_ssyrk(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(opa), C.numRows(), k, _alpha, FPTR(A.dataPtr()), A.columnsStride(), _beta, FPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					DCST(beta)
					cblas_dsyrk(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(opa), C.numRows(), k, _alpha, DPTR(A.dataPtr()), A.columnsStride(), _beta, DPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					cblas_csyrk(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(opa), C.numRows(), k, &_alpha, CPTR(A.dataPtr()), A.columnsStride(), &_beta, CPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					cblas_zsyrk(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(opa), C.numRows(), k, &_alpha, ZPTR(A.dataPtr()), A.columnsStride(), &_beta, ZPTR(C.dataPtr()), C.columnsStride());
				}
			#else
				throw NotSupported;
			#endif
		}
		return C;
	}

		/**
		\brief Symmetric rank-k update.
		\param A Matrix.
		\param opa Operation on matrix A.
		\param uplo Output fill mode.
		\param C Output matrix.
		\return Accessor to result.

		\f$ C = A^{\mbox{opa}} {A^{\mbox{opa}}}^\intercal \f$
		**/
		template<typename T, Location l>
		__host__ const Accessor<T,l>& BLASContext::syrk(const Accessor<T,l>& A, MatrixOperation opa, MatrixFillMode uplo, const Accessor<T,l>& C)
		{
			const T alpha = 1,
				beta = 0;
			return syrk(alpha, A, opa, beta, uplo, C);
		}

		/**
		\brief Symmetric rank-k update.
		\param A Matrix.
		\param uplo Output fill mode.
		\param C Output matrix.
		\return Accessor to result.

		\f$ C = A A^\intercal \f$
		**/
		template<typename T, Location l>
		__host__ const Accessor<T,l>& BLASContext::syrk(const Accessor<T,l>& A, MatrixFillMode uplo, const Accessor<T,l>& C)
		{
			const T alpha = 1,
				beta = 0;
			return syrk(alpha, A, OpNo, beta, uplo, C);
		}

	/**
	\brief Symmetric rank-2k update.
	\param op Operation on both input.
	\param alpha Scalar.
	\param A Left side matrix.
	\param B Right side matrix.
	\param beta Scalar.
	\param uplo Output fill mode.
	\param C Output matrix.
	\return Accessor to result.
	
	\f$ C = \alpha (A^{\mbox{op}} {B^{\mbox{op}}}^\intercal + B^{\mbox{op}} {A^{\mbox{op}}}^\intercal) + \beta C \f$
	**/
	template<typename T, Location l, typename TAlpha>
	__host__ const Accessor<T,l>& BLASContext::syr2k(MatrixOperation op, const TAlpha& alpha, const Accessor<T,l>& A, const Accessor<T,l>& B, const T& beta, MatrixFillMode uplo, const Accessor<T,l>& C)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(C)
		if(op==OpNo)
		{
			TEST_PRODUCT(A, OpNo, B, OpTr, C)
			TEST_PRODUCT(B, OpNo, A, OpTr, C)
		}		
		else
		{
			TEST_PRODUCT(A, OpTr, B, OpNo, C)
			TEST_PRODUCT(B, OpTr, A, OpNo, C)
		}
		const int k = (op==OpTr || op==OpHr) ? A.numRows() : A.numColumns();

		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cublasStatus_t err;		
				IF_FLOAT
				{
					FCST(alpha)
					FCST(beta)
					err = cublasSsyr2k(handle, getCuBLASFillMode(uplo), getCuBLASOperation(op), C.numRows(), k, &_alpha, FPTR(A.dataPtr()), A.columnsStride(), FPTR(B.dataPtr()), B.columnsStride(), &_beta, FPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					DCST(beta)
					err = cublasDsyr2k(handle, getCuBLASFillMode(uplo), getCuBLASOperation(op), C.numRows(), k, &_alpha, DPTR(A.dataPtr()), A.columnsStride(), DPTR(B.dataPtr()), B.columnsStride(), &_beta, DPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					err = cublasCsyr2k(handle, getCuBLASFillMode(uplo), getCuBLASOperation(op), C.numRows(), k, &_alpha, CPTR(A.dataPtr()), A.columnsStride(), CPTR(B.dataPtr()), B.columnsStride(), &_beta, CPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					err = cublasZsyr2k(handle, getCuBLASFillMode(uplo), getCuBLASOperation(op), C.numRows(), k, &_alpha, ZPTR(A.dataPtr()), A.columnsStride(), ZPTR(B.dataPtr()), B.columnsStride(), &_beta, ZPTR(C.dataPtr()), C.columnsStride());
				}
				TEST_EXCEPTION(err)
			#else
				throw NotSupported;
			#endif
		}
		else
		{
			#ifdef KARTET_ADD_CBLAS_INTERFACE
				IF_FLOAT
				{
					FCST(alpha)
					FCST(beta)
					cblas_ssyr2k(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(op), C.numRows(), k, _alpha, FPTR(A.dataPtr()), A.columnsStride(), FPTR(B.dataPtr()), B.columnsStride(), _beta, FPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					DCST(beta)
					cblas_dsyr2k(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(op), C.numRows(), k, _alpha, DPTR(A.dataPtr()), A.columnsStride(), DPTR(B.dataPtr()), B.columnsStride(), _beta, DPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					cblas_csyr2k(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(op), C.numRows(), k, &_alpha, CPTR(A.dataPtr()), A.columnsStride(), CPTR(B.dataPtr()), B.columnsStride(), &_beta, CPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					cblas_zsyr2k(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(op), C.numRows(), k, &_alpha, ZPTR(A.dataPtr()), A.columnsStride(), ZPTR(B.dataPtr()), B.columnsStride(), &_beta, ZPTR(C.dataPtr()), C.columnsStride());
				}
			#else
				throw NotSupported;
			#endif
		}
		return C;
	}

		/**
		\brief Symmetric rank-2k update.
		\param op Operation on both input.
		\param A Left side matrix.
		\param B Right side matrix.
		\param uplo Output fill mode.
		\param C Output matrix.
		\return Accessor to result.
	
		\f$ C = A^{\mbox{op}} {B^{\mbox{op}}}^\intercal + B^{\mbox{op}} {A^{\mbox{op}}}^\intercal \f$
		**/
		template<typename T, Location l>
		__host__ const Accessor<T,l>& BLASContext::syr2k(MatrixOperation op, const Accessor<T,l>& A, const Accessor<T,l>& B, MatrixFillMode uplo, const Accessor<T,l>& C)
		{
			const T alpha = 1,
				beta = 0;
			return syr2k(op, alpha, A, B, beta, uplo, C);
		}

		/**
		\brief Symmetric rank-2k update.
		\param A Left side matrix.
		\param B Right side matrix.
		\param uplo Output fill mode.
		\param C Output matrix.
		\return Accessor to result.
	
		\f$ C = A B^\intercal + B A^\intercal \f$
		**/
		template<typename T, Location l>
		__host__ const Accessor<T,l>& BLASContext::syr2k(const Accessor<T,l>& A, const Accessor<T,l>& B, MatrixFillMode uplo, const Accessor<T,l>& C)
		{
			const T alpha = 1,
				beta = 0;
			return syr2k(OpNo, alpha, A, B, beta, uplo, C);
		}

	/**
	\brief Triangular matrix-matrix multiplication.
	\param side Operation side.
	\param alpha Scalar.
	\param uplo Side mode of the triangular matrix.
	\param diag Diagonal mode of the triangular matrix.
	\param A Triangular matrix.
	\param opa Operation on matrix A.
	\param B Second matrix.
	\param C Output matrix.
	\return Accessor to result.

	If side is set to left : 
	\f$ C = \alpha A^{\mbox{opa}} B \f$
	If side is set to right : 
	\f$ C = \alpha B A^{\mbox{opa}} \f$
	**/
	template<typename T, Location l, typename TAlpha>
	__host__ const Accessor<T,l>& BLASContext::trmm(MatrixSideMode side, const TAlpha& alpha, MatrixFillMode uplo, MatrixDiagType diag, const Accessor<T,l>& A, MatrixOperation opa, const Accessor<T,l>& B, const Accessor<T,l>& C)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(B)
		TEST_SINGLE_SLICE(C)
		if(side==MatrixLeftSide)
			TEST_PRODUCT(A, opa, B, OpNo, C)
		else // MatrixRightSide
			TEST_PRODUCT(B, OpNo, A, opa, C)
		TEST_PRODUCT(A, OpNo, B, OpTr, C)

		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cublasStatus_t err;
				IF_FLOAT
				{
					FCST(alpha)
					err = cublasStrmm(handle, getCuBLASSideMode(side), getCuBLASFillMode(uplo), getCuBLASOperation(opa), getCuBLASDiagType(diag), B.numRows(), B.numColumns(), &_alpha, FPTR(A.dataPtr()), A.columnsStride(), FPTR(B.dataPtr()), B.columnsStride(), FPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					err = cublasDtrmm(handle, getCuBLASSideMode(side), getCuBLASFillMode(uplo), getCuBLASOperation(opa), getCuBLASDiagType(diag), B.numRows(), B.numColumns(), &_alpha, DPTR(A.dataPtr()), A.columnsStride(), DPTR(B.dataPtr()), B.columnsStride(), DPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					err = cublasCtrmm(handle, getCuBLASSideMode(side), getCuBLASFillMode(uplo), getCuBLASOperation(opa), getCuBLASDiagType(diag), B.numRows(), B.numColumns(), &_alpha, CPTR(A.dataPtr()), A.columnsStride(), CPTR(B.dataPtr()), B.columnsStride(), CPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					err = cublasZtrmm(handle, getCuBLASSideMode(side), getCuBLASFillMode(uplo), getCuBLASOperation(opa), getCuBLASDiagType(diag), B.numRows(), B.numColumns(), &_alpha, ZPTR(A.dataPtr()), A.columnsStride(), ZPTR(B.dataPtr()), B.columnsStride(), ZPTR(C.dataPtr()), C.columnsStride());
				}
				TEST_EXCEPTION(err)
			#else
				throw NotSupported;
			#endif
		}
		else
		{
			#ifdef KARTET_ADD_CBLAS_INTERFACE
				IF_FLOAT
				{
					FCST(alpha)
					cblas_strmm(CblasColMajor, getCBLASSideMode(side), getCBLASFillMode(uplo), getCBLASOperation(opa), getCBLASDiagType(diag), B.numRows(), B.numColumns(), _alpha, FPTR(A.dataPtr()), A.columnsStride(), FPTR(B.dataPtr()), B.columnsStride(), FPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					cblas_dtrmm(CblasColMajor, getCBLASSideMode(side), getCBLASFillMode(uplo), getCBLASOperation(opa), getCBLASDiagType(diag), B.numRows(), B.numColumns(), _alpha, DPTR(A.dataPtr()), A.columnsStride(), DPTR(B.dataPtr()), B.columnsStride(), DPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					cblas_ctrmm(CblasColMajor, getCBLASSideMode(side), getCBLASFillMode(uplo), getCBLASOperation(opa), getCBLASDiagType(diag), B.numRows(), B.numColumns(), &_alpha, CPTR(A.dataPtr()), A.columnsStride(), CPTR(B.dataPtr()), B.columnsStride(), CPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					cblas_ztrmm(CblasColMajor, getCBLASSideMode(side), getCBLASFillMode(uplo), getCBLASOperation(opa), getCBLASDiagType(diag), B.numRows(), B.numColumns(), &_alpha, ZPTR(A.dataPtr()), A.columnsStride(), ZPTR(B.dataPtr()), B.columnsStride(), ZPTR(C.dataPtr()), C.columnsStride());
				}
			#else
				throw NotSupported;
			#endif
		}
		return C;
	}

		/**
		\brief Triangular matrix-matrix multiplication.
		\param side Operation side.
		\param uplo Side mode of the triangular matrix.
		\param diag Diagonal mode of the triangular matrix.
		\param A Triangular matrix.
		\param opa Operation on matrix A.
		\param B Second matrix.
		\param C Output matrix.
		\return Accessor to result.

		If side is set to left : 
		\f$ C = A^{\mbox{opa}} B \f$
		If side is set to right : 
		\f$ C = B A^{\mbox{opa}} \f$
		**/
		template<typename T, Location l>
		__host__ const Accessor<T,l>& BLASContext::trmm(MatrixSideMode side, MatrixFillMode uplo, MatrixDiagType diag, const Accessor<T,l>& A, MatrixOperation opa, const Accessor<T,l>& B, const Accessor<T,l>& C)
		{
			const T alpha = 1;
			return trmm(side, alpha, uplo, diag, A, opa, B, C);
		}
		
		/**
		\brief Triangular matrix-matrix multiplication.
		\param side Operation side.
		\param uplo Side mode of the triangular matrix.
		\param diag Diagonal mode of the triangular matrix.
		\param A Triangular matrix.
		\param B Second matrix.
		\param C Output matrix.
		\return Accessor to result.

		If side is set to left : 
		\f$ C = A B \f$
		If side is set to right : 
		\f$ C = B A \f$
		**/
		template<typename T, Location l>
		__host__ const Accessor<T,l>& BLASContext::trmm(MatrixSideMode side, MatrixFillMode uplo, MatrixDiagType diag, const Accessor<T,l>& A, const Accessor<T,l>& B, const Accessor<T,l>& C)
		{
			const T alpha = 1;
			return trmm(side, alpha, uplo, diag, A, OpNo, B, C);
		}
		
	/**
	\brief Solve triangular system with multiple left hand side.
	\param side Side mode.
	\param uplo Fill mode.
	\param diag Diagonal type.
	\param A Triangular matrix.
	\param opa Operation on the matrix A.
	\param alpha Scalar.
	\param B Input left hand side, output right hand side.
	\return Accessor to result.

	If side is set to left : 
	\f$ \alpha B = A^{\mbox{op}} B' \f$
	If side is set to right : 
	\f$ \alpha B = B' A^{\mbox{op}} \f$
	**/
	template<typename T, Location l, typename TAlpha>
	__host__ const Accessor<T,l>& BLASContext::trsm(MatrixSideMode side, MatrixFillMode uplo, MatrixDiagType diag, const Accessor<T,l>& A, MatrixOperation opa, const TAlpha& alpha, const Accessor<T,l>& B)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(B)
		if(side==MatrixLeftSide)
			TEST_PRODUCT(A, opa, B, OpNo, B)
		else // MatrixRightSide
			TEST_PRODUCT(B, OpNo, A, opa, B)
	
		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cublasStatus_t err;
				IF_FLOAT
				{
					FCST(alpha)
					err = cublasStrsm(handle, getCuBLASSideMode(side), getCuBLASFillMode(uplo), getCuBLASOperation(opa), getCuBLASDiagType(diag), B.numRows(), B.numColumns(), &_alpha, FPTR(A.dataPtr()), A.columnsStride(), FPTR(B.dataPtr()), B.columnsStride());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					err = cublasDtrsm(handle, getCuBLASSideMode(side), getCuBLASFillMode(uplo), getCuBLASOperation(opa), getCuBLASDiagType(diag), B.numRows(), B.numColumns(), &_alpha, DPTR(A.dataPtr()), A.columnsStride(), DPTR(B.dataPtr()), B.columnsStride());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					err = cublasCtrsm(handle, getCuBLASSideMode(side), getCuBLASFillMode(uplo), getCuBLASOperation(opa), getCuBLASDiagType(diag), B.numRows(), B.numColumns(), &_alpha, CPTR(A.dataPtr()), A.columnsStride(), CPTR(B.dataPtr()), B.columnsStride());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					err = cublasZtrsm(handle, getCuBLASSideMode(side), getCuBLASFillMode(uplo), getCuBLASOperation(opa), getCuBLASDiagType(diag), B.numRows(), B.numColumns(), &_alpha, ZPTR(A.dataPtr()), A.columnsStride(), ZPTR(B.dataPtr()), B.columnsStride());
				}
				TEST_EXCEPTION(err)
			#else
				throw NotSupported;
			#endif
		}
		else
		{
			#ifdef KARTET_ADD_CBLAS_INTERFACE
				IF_FLOAT
				{
					FCST(alpha)
					cblas_strsm(CblasColMajor, getCBLASSideMode(side), getCBLASFillMode(uplo), getCBLASOperation(opa), getCBLASDiagType(diag), B.numRows(), B.numColumns(), _alpha, FPTR(A.dataPtr()), A.columnsStride(), FPTR(B.dataPtr()), B.columnsStride());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					cblas_dtrsm(CblasColMajor, getCBLASSideMode(side), getCBLASFillMode(uplo), getCBLASOperation(opa), getCBLASDiagType(diag), B.numRows(), B.numColumns(), _alpha, DPTR(A.dataPtr()), A.columnsStride(), DPTR(B.dataPtr()), B.columnsStride());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					cblas_ctrsm(CblasColMajor, getCBLASSideMode(side), getCBLASFillMode(uplo), getCBLASOperation(opa), getCBLASDiagType(diag), B.numRows(), B.numColumns(), &_alpha, CPTR(A.dataPtr()), A.columnsStride(), CPTR(B.dataPtr()), B.columnsStride());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					cblas_ztrsm(CblasColMajor, getCBLASSideMode(side), getCBLASFillMode(uplo), getCBLASOperation(opa), getCBLASDiagType(diag), B.numRows(), B.numColumns(), &_alpha, ZPTR(A.dataPtr()), A.columnsStride(), ZPTR(B.dataPtr()), B.columnsStride());
				}
			#else
				throw NotSupported;
			#endif
		}
		return B;
	}

		/**
		\brief Solve triangular system with multiple left hand side.
		\param side Side mode.
		\param uplo Fill mode.
		\param diag Diagonal type.
		\param A Triangular matrix.
		\param opa Operation on the matrix A.
		\param B Input left hand side, output right hand side.
		\return Accessor to result.

		If side is set to left : 
		\f$ B = A^{\mbox{op}} B' \f$
		If side is set to right : 
		\f$ B = B' A^{\mbox{op}} \f$
		**/
		template<typename T, Location l>
		__host__ const Accessor<T,l>& BLASContext::trsm(MatrixSideMode side, MatrixFillMode uplo, MatrixDiagType diag, const Accessor<T,l>& A, MatrixOperation opa, const Accessor<T,l>& B)
		{
			const T alpha = 1;
			return trsm(side, uplo, diag, A, opa, alpha, B);
		}

		/**
		\brief Solve triangular system with multiple left hand side.
		\param side Side mode.
		\param uplo Fill mode.
		\param diag Diagonal type.
		\param A Triangular matrix.
		\param B Input left hand side, output right hand side.
		\return Accessor to result.

		If side is set to left : 
		\f$ B = A B' \f$
		If side is set to right : 
		\f$ B = B' A \f$
		**/
		template<typename T, Location l>
		__host__ const Accessor<T,l>& BLASContext::trsm(MatrixSideMode side, MatrixFillMode uplo, MatrixDiagType diag, const Accessor<T,l>& A, const Accessor<T,l>& B)
		{
			const T alpha = 1;
			return trsm(side, uplo, diag, A, OpNo, alpha, B);
		}

	/**
	\brief Hermitian matrix-matrix product.
	\param side Side of the operation.
	\param alpha Scalar.
	\param uplo Fill mode.
	\param A Hermitian matrix.
	\param B Matrix.
	\param beta Scalar.
	\param C Output matrix.
	\return Accessor to result.

	If side is set to left : 
	\f$ C = \alpha A B + \beta C \f$
	If side is set to right : 
	\f$ C = \alpha B A + \beta C \f$
	**/
	template<typename T, Location l, typename TAlpha, typename TBeta>
	__host__ const Accessor<T,l>& BLASContext::hemm(MatrixSideMode side, const TAlpha& alpha, MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& B, const TBeta& beta, const Accessor<T,l>& C)
	{
		ALLOWED_TYPES_VERIFICATION
		TYPE_MUST_BE_COMPLEX
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(B)
		TEST_SINGLE_SLICE(C)
		if(side==MatrixLeftSide)
			TEST_PRODUCT(A, OpNo, B, OpNo, C)
		else // MatrixRightSide
			TEST_PRODUCT(B, OpNo, A, OpNo, C)

		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cublasStatus_t err;
				IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					err = cublasChemm(handle, getCuBLASSideMode(side), getCuBLASFillMode(uplo), C.numRows(), C.numColumns(), &_alpha, CPTR(A.dataPtr()), A.columnsStride(), CPTR(B.dataPtr()), B.columnsStride(), &_beta, CPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					err = cublasZhemm(handle, getCuBLASSideMode(side), getCuBLASFillMode(uplo), C.numRows(), C.numColumns(), &_alpha, ZPTR(A.dataPtr()), A.columnsStride(), ZPTR(B.dataPtr()), B.columnsStride(), &_beta, ZPTR(C.dataPtr()), C.columnsStride());
				}
				TEST_EXCEPTION(err)
			#else
				throw NotSupported;
			#endif
		}
		else
		{
			#ifdef KARTET_ADD_CBLAS_INTERFACE
				IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					cblas_chemm(CblasColMajor, getCBLASSideMode(side), getCBLASFillMode(uplo), C.numRows(), C.numColumns(), &_alpha, CPTR(A.dataPtr()), A.columnsStride(), CPTR(B.dataPtr()), B.columnsStride(), &_beta, CPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					cblas_zhemm(CblasColMajor, getCBLASSideMode(side), getCBLASFillMode(uplo), C.numRows(), C.numColumns(), &_alpha, ZPTR(A.dataPtr()), A.columnsStride(), ZPTR(B.dataPtr()), B.columnsStride(), &_beta, ZPTR(C.dataPtr()), C.columnsStride());
				}
			#else
				throw NotSupported;
			#endif
		}
		return C;
	}

		/**
		\brief Hermitian matrix-matrix product.
		\param side Side of the operation.
		\param uplo Fill mode.
		\param A Hermitian matrix.
		\param B Matrix.
		\param C Output matrix.
		\return Accessor to result.

		If side is set to left : 
		\f$ C = A B \f$
		If side is set to right : 
		\f$ C = B A \f$
		**/
		template<typename T, Location l>
		__host__ const Accessor<T,l>& BLASContext::hemm(MatrixSideMode side, MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& B, const Accessor<T,l>& C)
		{
			const T alpha = 1,
				beta = 0;
			return hemm(side, alpha, uplo, A, B, beta, C);
		}

	/**
	\brief Hermitian rank-k update.
	\param alpha Scalar.
	\param A Matrix.
	\param opa Operation applied to matrix A.
	\param beta Scalar.
	\param uplo Fill mode.
	\param C Output matrix.
	\return Accessor to result.

	\f$ C = \alpha A^{\mbox{opa}} {A^{\mbox{opa}}}^+ + \beta C \f$
	**/
	template<typename T, Location l, typename TAlpha, typename TBeta>
	__host__ const Accessor<T,l>& BLASContext::herk(const TAlpha& alpha, const Accessor<T,l>& A, MatrixOperation opa, const TBeta& beta, MatrixFillMode uplo, const Accessor<T,l>& C)
	{
		ALLOWED_TYPES_VERIFICATION
		TYPE_MUST_BE_COMPLEX
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(C)
		if(opa==OpNo)
			TEST_PRODUCT(A, OpNo, A, OpHr, C)
		else
			TEST_PRODUCT(A, OpHr, A, OpNo, C)
		int k = (opa==OpTr || opa==OpHr) ? A.numRows() : A.numColumns();

		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cublasStatus_t err;
				IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					err = cublasCherk(handle, getCuBLASFillMode(uplo), getCuBLASOperation(opa), C.numRows(), k, &_alpha, CPTR(A.dataPtr()), A.columnsStride(), &_beta, CPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					err = cublasZherk(handle, getCuBLASFillMode(uplo), getCuBLASOperation(opa), C.numRows(), k, &_alpha, ZPTR(A.dataPtr()), A.columnsStride(), &_beta, ZPTR(C.dataPtr()), C.columnsStride());
				}
				TEST_EXCEPTION(err)
			#else
				throw NotSupported;
			#endif
		}
		else
		{
			#ifdef KARTET_ADD_CBLAS_INTERFACE
				IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					cblas_cherk(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(opa), C.numRows(), k, &_alpha, CPTR(A.dataPtr()), A.columnsStride(), &_beta, CPTR(C.dataPtr()), C.columnsStride());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					cblas_zherk(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(opa), C.numRows(), k, &_alpha, ZPTR(A.dataPtr()), A.columnsStride(), &_beta, ZPTR(C.dataPtr()), C.columnsStride());
				}
			#else
				throw NotSupported;
			#endif
		}
		return C;
	}

		/**
		\brief Hermitian rank-k update.
		\param A Matrix.
		\param opa Operation applied to matrix A.
		\param uplo Fill mode.
		\param C Output matrix.
		\return Accessor to result.

		\f$ C = A^{\mbox{opa}} {A^{\mbox{opa}}}^+ \f$
		**/
		template<typename T, Location l>
		__host__ const Accessor<T,l>& BLASContext::herk(const Accessor<T,l>& A, MatrixOperation opa, MatrixFillMode uplo, const Accessor<T,l>& C)
		{
			const T alpha = 1,
				beta = 0;
			return herk(alpha, A, opa, beta, uplo, C);
		}

		/**
		\brief Hermitian rank-k update.
		\param A Matrix.
		\param uplo Fill mode.
		\param C Output matrix.
		\return Accessor to result.

		\f$ C = A A^+ \f$
		**/
		template<typename T, Location l>
		__host__ const Accessor<T,l>& BLASContext::herk(const Accessor<T,l>& A, MatrixFillMode uplo, const Accessor<T,l>& C)
		{
			const T alpha = 1,
				beta = 0;
			return herk(alpha, A, OpNo, beta, uplo, C);
		}		

	/**
	\brief Hermitian rank-2 update.
	\param op Operation.
	\param alpha Scalar.
	\param A First matrix.
	\param B Second matrix.
	\param beta Scalar.
	\param uplo Fill mode.
	\param C Output matrix.
	\return Accessor to result.

	\f$ C = \alpha A^{\mbox{op}} {B^{\mbox{op}}}^+ + \alpha^* B^{\mbox{op}} {A^{\mbox{op}}}^+ + \beta C \f$
	**/
	template<typename T, Location l, typename TAlpha, typename TBeta>
	__host__ const Accessor<T,l>& BLASContext::her2k(MatrixOperation op, const TAlpha& alpha, const Accessor<T,l>& A, const Accessor<T,l>& B, const TBeta& beta, MatrixFillMode uplo, const Accessor<T,l>& C)
	{
		ALLOWED_TYPES_VERIFICATION
		TYPE_MUST_BE_COMPLEX
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(B)
		TEST_SINGLE_SLICE(C)
		if(op==OpNo)
		{
			TEST_PRODUCT(A, OpNo, B, OpHr, C)
			TEST_PRODUCT(B, OpNo, A, OpHr, C)
		}		
		else
		{
			TEST_PRODUCT(A, OpHr, B, OpNo, C)
			TEST_PRODUCT(B, OpHr, A, OpNo, C)
		}	
		const int k = (op==OpTr || op==OpHr) ? A.numRows() : A.numColumns();

		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cublasStatus_t err;
				IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					err = cublasCher2k(handle, getCuBLASFillMode(uplo), getCuBLASOperation(op), C.numRows(), k, &_alpha, CPTR(A.dataPtr()), A.columnsStride(), CPTR(B.dataPtr()), B.columnsStride(), &_beta, CPTR(C.dataPtr()), C.columnsStride());
				}
				IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					err = cublasZher2k(handle, getCuBLASFillMode(uplo), getCuBLASOperation(op), C.numRows(), k, &_alpha, ZPTR(A.dataPtr()), A.columnsStride(), ZPTR(B.dataPtr()), B.columnsStride(), &_beta, ZPTR(C.dataPtr()), C.columnsStride());
				}
				TEST_EXCEPTION(err)
			#else
				throw NotSupported;
			#endif
		}
		else
		{
			#ifdef KARTET_ADD_CBLAS_INTERFACE
				IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					cblas_cher2k(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(op), C.numRows(), k, &_alpha, CPTR(A.dataPtr()), A.columnsStride(), CPTR(B.dataPtr()), B.columnsStride(), &_beta, CPTR(C.dataPtr()), C.columnsStride());
				}
				IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					cblas_zher2k(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(op), C.numRows(), k, &_alpha, ZPTR(A.dataPtr()), A.columnsStride(), ZPTR(B.dataPtr()), B.columnsStride(), &_beta, ZPTR(C.dataPtr()), C.columnsStride());
				}
			#else
				throw NotSupported;
			#endif
		}
		return C;
	}

		/**
		\brief Hermitian rank-2 update.
		\param op Operation.
		\param A First matrix.
		\param B Second matrix.
		\param uplo Fill mode.
		\param C Output matrix.
		\return Accessor to result.

		\f$ C = A^{\mbox{op}} {B^{\mbox{op}}}^+ + \alpha^* B^{\mbox{op}} {A^{\mbox{op}}}^+ \f$
		**/
		template<typename T, Location l>
		__host__ const Accessor<T,l>& BLASContext::her2k(MatrixOperation op, const Accessor<T,l>& A, const Accessor<T,l>& B, MatrixFillMode uplo, const Accessor<T,l>& C)
		{
			const T alpha = 1,
				beta = 0;
			return her2k(op, alpha, A, B, beta, uplo, C);
		}

		/**
		\brief Hermitian rank-2 update.
		\param A First matrix.
		\param B Second matrix.
		\param uplo Fill mode.
		\param C Output matrix.
		\return Accessor to result.

		\f$ C = A B^+ + B A^+ \f$
		**/
		template<typename T, Location l>
		__host__ const Accessor<T,l>& BLASContext::her2k(const Accessor<T,l>& A, const Accessor<T,l>& B, MatrixFillMode uplo, const Accessor<T,l>& C)
		{
			const T alpha = 1,
				beta = 0;
			return her2k(OpNo, alpha, A, B, beta, uplo, C);
		}

	/**
	\brief Matrix-matrix addition transposition.
	\param alpha Scalar.
	\param A Left side matrix.
	\param opa Operation on matrix A.
	\param beta Scalar.
	\param B Right side matrix.
	\param opb Operation on Matrix B.
	\param C Output matrix.
	\return Accessor to result.

	\f$ C = \alpha A^{\mbox{opa}} + \beta B^{\mbox{opb}} \f$
	**/
	template<typename T, Location l, typename TAlpha, typename TBeta>
	__host__ const Accessor<T,l>& BLASContext::geam(const TAlpha& alpha, const Accessor<T,l>& A, MatrixOperation opa, const TBeta& beta, const Accessor<T,l>& B, MatrixOperation opb, const Accessor<T,l>& C)
	{
		STATIC_ASSERT_VERBOSE(l==DeviceSide, LOCATION_NOT_SUPPORTED)
		ALLOWED_TYPES_VERIFICATION
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(B)
		TEST_SINGLE_SLICE(C)

		#ifdef __CUDACC__
			cublasStatus_t err;
			IF_FLOAT
			{
				FCST(alpha)
				FCST(beta)
				err = cublasSgeam(handle, getCuBLASOperation(opa), getCuBLASOperation(opb), C.numRows(), C.numColumns(), FPTR(&alpha), FPTR(A.dataPtr()), A.columnsStride(), FPTR(&beta), FPTR(B.dataPtr()), B.columnsStride(), FPTR(C.dataPtr()), C.columnsStride());
			}
			else IF_DOUBLE
			{
				DCST(alpha)
				DCST(beta)
				err = cublasDgeam(handle, getCuBLASOperation(opa), getCuBLASOperation(opb), C.numRows(), C.numColumns(), DPTR(&alpha), DPTR(A.dataPtr()), A.columnsStride(), DPTR(&beta), DPTR(B.dataPtr()), B.columnsStride(), DPTR(C.dataPtr()), C.columnsStride());
			}
			else IF_CX_FLOAT
			{
				CCST(alpha)
				CCST(beta)
				err = cublasCgeam(handle, getCuBLASOperation(opa), getCuBLASOperation(opb), C.numRows(), C.numColumns(), CPTR(&alpha), CPTR(A.dataPtr()), A.columnsStride(), CPTR(&beta), CPTR(B.dataPtr()), B.columnsStride(), CPTR(C.dataPtr()), C.columnsStride());
			}
			else IF_CX_DOUBLE
			{
				ZCST(alpha)
				ZCST(beta)
				err = cublasZgeam(handle, getCuBLASOperation(opa), getCuBLASOperation(opb), C.numRows(), C.numColumns(), ZPTR(&alpha), ZPTR(A.dataPtr()), A.columnsStride(), ZPTR(&beta), ZPTR(B.dataPtr()), B.columnsStride(), ZPTR(C.dataPtr()), C.columnsStride());
			}
			TEST_EXCEPTION(err)
		#else
			throw NotSupported;
		#endif
		return C;
	}

		/**
		\brief Matrix-matrix addition transposition.
		\param A Left side matrix.
		\param opa Operation on matrix A.
		\param B Right side matrix.
		\param opb Operation on Matrix B.
		\param C Output matrix.
		\return Accessor to result.

		\f$ C = A^{\mbox{opa}} + B^{\mbox{opb}} \f$
		**/
		template<typename T, Location l>
		__host__ const Accessor<T,l>& BLASContext::geam(const Accessor<T,l>& A, MatrixOperation opa, const Accessor<T,l>& B, MatrixOperation opb, const Accessor<T,l>& C)
		{
			const T alpha = 1,
				beta = 1;
			return geam(alpha, A, opa, beta, B, opb, C);
		}

		/**
		\brief Matrix-matrix addition transposition.
		\param A Left side matrix.
		\param B Right side matrix.
		\param C Output matrix.
		\return Accessor to result.

		\f$ C = A + B \f$
		**/
		template<typename T, Location l>
		__host__ const Accessor<T,l>& BLASContext::geam(const Accessor<T,l>& A, const Accessor<T,l>& B, const Accessor<T,l>& C)
		{
			const T alpha = 1,
				beta = 1;
			return geam(alpha, A, OpNo, beta, B, OpNo, C);
		}

	/**
	\brief Diagonal matrix-matrix product.
	\param side Side mode.
	\param A First matrix.
	\param x Vector used as diagonal matrix.
	\param C Output matrix.
	\return Accessor to result.
	
	If side is set to left : 
	\f$ C = A \mbox{diag}(x) \f$
	If side is set to right : 
	\f$ C = \mbox{diag}(x) A \f$
	**/
	template<typename T, Location l>
	__host__ const Accessor<T,l>& BLASContext::dgmm(MatrixSideMode side, const Accessor<T,l>& A, const Accessor<T,l>& x, const Accessor<T,l>& C)
	{
		STATIC_ASSERT_VERBOSE(l==DeviceSide, LOCATION_NOT_SUPPORTED)
		ALLOWED_TYPES_VERIFICATION
		TEST_MONOLITHIC(x)
		TEST_SINGLE_SLICE(A)
	
		#ifdef __CUDACC__
			cublasStatus_t err;
			IF_FLOAT
				err = cublasSdgmm(handle, getCuBLASSideMode(side), C.numRows(), C.numColumns(), FPTR(A.dataPtr()), A.columnsStride(), FPTR(x.dataPtr()), 1, FPTR(C.dataPtr()), C.columnsStride());
			else IF_DOUBLE
				err = cublasDdgmm(handle, getCuBLASSideMode(side), C.numRows(), C.numColumns(), DPTR(A.dataPtr()), A.columnsStride(), DPTR(x.dataPtr()), 1, DPTR(C.dataPtr()), C.columnsStride()); 
			else IF_CX_FLOAT
				err = cublasCdgmm(handle, getCuBLASSideMode(side), C.numRows(), C.numColumns(), CPTR(A.dataPtr()), A.columnsStride(), CPTR(x.dataPtr()), 1, CPTR(C.dataPtr()), C.columnsStride());
			else IF_CX_DOUBLE
				err = cublasZdgmm(handle, getCuBLASSideMode(side), C.numRows(), C.numColumns(), ZPTR(A.dataPtr()), A.columnsStride(), ZPTR(x.dataPtr()), 1, ZPTR(C.dataPtr()), C.columnsStride());
			TEST_EXCEPTION(err)
		#else
			throw NotSupported;
		#endif
		return C;
	}

// Clean :
	#undef ALLOWED_TYPES_VERIFICATION
	#undef TYPE_MUST_BE_COMPLEX
	#undef TEST_CONTEXT
	#undef TEST_MONOLITHIC
	#undef IF_FLOAT
	#undef IF_DOUBLE
	#undef IF_CX_FLOAT
	#undef IF_CX_DOUBLE
	#undef handle
	#undef FCST
	#undef DCST
	#undef CCST
	#undef ZCST
	#undef FPTR
	#undef DPTR
	#undef CPTR
	#undef ZPTR
	#undef TEST_EXCEPTION

} // namespace Kartet

#endif

