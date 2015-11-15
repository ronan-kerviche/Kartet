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

#ifndef __KARTET_BLAS_TOOLS__
#define __KARTET_BLAS_TOOLS__

namespace Kartet
{
// Type Lists :
	typedef TypeList< float,
		TypeList< double,
		TypeList< cuFloatComplex,
		TypeList< cuDoubleComplex,
		Void
		> > > > BLASKnownTypes;

// Type tools :
	#define ALLOWED_TYPES_VERIFICATION		StaticAssert<Belongs<BLASKnownTypes, T>::value>();
	#define TYPE_MUST_BE_COMPLEX			StaticAssert<TypeInfo<T>::isComplex>();
	#define TEST_MONOLITHIC(x)			{if(!(x).isMonolithic()) throw IncompatibleLayout;}
	#define TEST_SINGLE_SLICE(x)			{if((x).getNumSlices()>1) throw IncompatibleLayout;} 
	#define TEST_PRODUCT(A, opa, B, opb, C)		{if(!isProductValid(A, opa, B, opb, C)) throw InvalidOperation;}
	#define IF_FLOAT				if(SameTypes<T, float>::test)
	#define IF_DOUBLE				if(SameTypes<T, double>::test)
	#define IF_CX_FLOAT				if(SameTypes<T, cuFloatComplex>::test)
	#define IF_CX_DOUBLE				if(SameTypes<T, cuDoubleComplex>::test)
	#define FCST(x)					const float _##x = static_cast<float>(x);
	#define DCST(x)					const double _##x = static_cast<double>(x);
	#define CCST(x)					const cuFloatComplex _##x = toFloatComplex(x);
	#define ZCST(x)					const cuDoubleComplex _##x = toDoubleComplex(x);
	#define FPTR(x)					reinterpret_cast<float*>(x)
	#define DPTR(x)					reinterpret_cast<double*>(x)
	#define CPTR(x)					reinterpret_cast<cuFloatComplex*>(x)
	#define ZPTR(x)					reinterpret_cast<cuDoubleComplex*>(x)
	#define TEST_EXCEPTION(x)			{if(x!=CUBLAS_STATUS_SUCCESS) throw static_cast<Exception>(CuBLASExceptionOffset + x);}

// BLAS :
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
	__host__ inline cublasOperation_t BLASContext::getCuBLASOperation(const MatrixOperation& op)
	{
		return (op==OpTr) ? CUBLAS_OP_T : ((op==OpHr) ? CUBLAS_OP_C : CUBLAS_OP_N);
	}

	__host__ inline cublasFillMode_t BLASContext::getCuBLASFillMode(const MatrixFillMode& m)
	{
		return (m==MatrixFillUp) ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
	}

	__host__ inline cublasDiagType_t BLASContext::getCuBLASDiagType(const MatrixDiagType& t)
	{
		return (t==MatrixDiagUnit) ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;
	}

	__host__ inline cublasSideMode_t BLASContext::getCuBLASSideMode(const MatrixSideMode& s)
	{
		return (s==MatrixRightSide) ? CUBLAS_SIDE_RIGHT : CUBLAS_SIDE_LEFT;
	}
	#endif

	#ifdef KARTET_ADD_CBLAS_INTERFACE
	__host__ inline CBLAS_TRANSPOSE BLASContext::getCBLASOperation(const MatrixOperation& op)
	{
		return (op==OpTr) ? CblasTrans : ((op==OpHr) ? CblasConjTrans : CblasNoTrans);
	}

	__host__ inline CBLAS_UPLO BLASContext::getCBLASFillMode(const MatrixFillMode& m)
	{
		return (m==MatrixFillUp) ? CblasUpper : CblasLower;
	}

	__host__ inline CBLAS_DIAG BLASContext::getCBLASDiagType(const MatrixDiagType& t)
	{
		return (t==MatrixDiagUnit) ? CblasUnit : CblasNonUnit;
	}

	__host__ inline CBLAS_SIDE BLASContext::getCBLASSideMode(const MatrixSideMode& s)
	{
		return (s==MatrixRightSide) ? CblasRight : CblasLeft;
	}
	#endif

	__host__ inline bool BLASContext::isProductValid(const Layout& A, MatrixOperation opa, const Layout& B, MatrixOperation opb, const Layout& C)
	{
		index_t aR = 0,
			aC = 0,
			bR = 0,
			bC = 0,
			cR = C.getNumRows(),
			cC = C.getNumColumns();

		if(opa==OpTr || opa==OpHr)
		{
			aR = A.getNumColumns();
			aC = A.getNumRows();
		}
		else
		{
			aR = A.getNumRows();
			aC = A.getNumColumns();
		}

		if(opb==OpTr || opb==OpHr)
		{
			bR = B.getNumColumns();
			bC = B.getNumRows();
		}
		else
		{
			bR = B.getNumRows();
			bC = B.getNumColumns();
		}

		return (aR==cR) && (aC==bR) && (bC==cC);
	}

	__host__ inline bool BLASContext::isProductValid(const Layout& A, const Layout& B, const Layout& C)
	{
		return BLASContext::isProductValid(A, OpNo, B, OpNo, C);
	}

	__host__ inline Layout BLASContext::getProductLayout(const Layout& A, MatrixOperation opa, const Layout& B, MatrixOperation opb)
	{
		index_t aR = 0,
			aC = 0,
			bR = 0,
			bC = 0;

		if(A.getNumSlices()!=1 || B.getNumSlices()!=1)
			throw InvalidOperation;

		if(opa==OpTr || opa==OpHr)
		{
			aR = A.getNumColumns();
			aC = A.getNumRows();
		}
		else
		{
			aR = A.getNumRows();
			aC = A.getNumColumns();
		}

		if(opb==OpTr || opb==OpHr)
		{
			bR = B.getNumColumns();
			bC = B.getNumRows();
		}
		else
		{
			bR = B.getNumRows();
			bC = B.getNumColumns();
		}

		if(aC!=bR)
			throw InvalidOperation;
		
		return Layout(aR, bC);
	}

	__host__ inline Layout BLASContext::getProductLayout(const Layout& A, const Layout& B)
	{
		return getProductLayout(A, OpNo, B, OpNo);
	}

// Functions : 
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
					err = cublasIsamax(handle, x.getNumElements(), FPTR(x.getPtr()), 1, &res);
				else IF_DOUBLE
					err = cublasIdamax(handle, x.getNumElements(), DPTR(x.getPtr()), 1, &res);
				else IF_CX_FLOAT
					err = cublasIcamax(handle, x.getNumElements(), CPTR(x.getPtr()), 1, &res);
				else IF_CX_DOUBLE
					err = cublasIzamax(handle, x.getNumElements(), ZPTR(x.getPtr()), 1, &res);
				TEST_EXCEPTION(err)
			#else
				throw NotSupported;
			#endif
		}
		else
		{
			#ifdef KARTET_ADD_CBLAS_INTERFACE
				IF_FLOAT
					res = cblas_isamax(x.getNumElements(), FPTR(x.getPtr()), 1);
				else IF_DOUBLE
					res = cblas_idamax(x.getNumElements(), DPTR(x.getPtr()), 1);
				else IF_CX_FLOAT
					res = cblas_icamax(x.getNumElements(), CPTR(x.getPtr()), 1);
				else IF_CX_DOUBLE
					res = cblas_izamax(x.getNumElements(), ZPTR(x.getPtr()), 1);
			#else
				throw NotSupported;
			#endif
		}

		return res;
	}

	template<typename T, Location l>
	__host__ int BLASContext::Iamin(const Accessor<T,l>& x)
	{
		StaticAssert<l==DeviceSide>();
		ALLOWED_TYPES_VERIFICATION
		TEST_MONOLITHIC(x)
		int res;

		#ifdef __CUDACC__
			cublasStatus_t err;
			IF_FLOAT
				err = cublasIsamin(handle, x.getNumElements(), FPTR(x.getPtr()), 1, &res);
			else IF_DOUBLE
				err = cublasIdamin(handle, x.getNumElements(), DPTR(x.getPtr()), 1, &res);
			else IF_CX_FLOAT
				err = cublasIcamin(handle, x.getNumElements(), CPTR(x.getPtr()), 1, &res);
			else IF_CX_DOUBLE
				err = cublasIzamin(handle, x.getNumElements(), ZPTR(x.getPtr()), 1, &res);
			TEST_EXCEPTION(err)
		#else
			throw NotSupported;
		#endif

		return res;
	}

	template<typename T, Location l>
	__host__ typename TypeInfo<T>::BaseType BLASContext::asum(const Accessor<T,l>& x)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_MONOLITHIC(x)
		typename TypeInfo<T>::BaseType res;

		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cublasStatus_t err;
				IF_FLOAT
					err = cublasSasum(handle, x.getNumElements(), FPTR(x.getPtr()), 1, FPTR(&res));
				else IF_DOUBLE
					err = cublasDasum(handle, x.getNumElements(), DPTR(x.getPtr()), 1, DPTR(&res));
				else IF_CX_FLOAT
					err = cublasScasum(handle, x.getNumElements(), CPTR(x.getPtr()), 1, FPTR(&res));
				else IF_CX_DOUBLE
					err = cublasDzasum(handle, x.getNumElements(), ZPTR(x.getPtr()), 1, DPTR(&res));
				TEST_EXCEPTION(err)
			#else
				throw NotSupported;
			#endif
		}
		else
		{	
			#ifdef KARTET_ADD_CBLAS_INTERFACE
				IF_FLOAT
					res = cblas_sasum(x.getNumElements(), FPTR(x.getPtr()), 1);
				else IF_DOUBLE
					res = cblas_dasum(x.getNumElements(), DPTR(x.getPtr()), 1);
				else IF_CX_FLOAT
					res = cblas_scasum(x.getNumElements(), CPTR(x.getPtr()), 1);
				else IF_CX_DOUBLE
					res = cblas_dzasum(x.getNumElements(), ZPTR(x.getPtr()), 1);
			#else
				throw NotSupported;
			#endif
		}
		return res;
	}

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
					err = cublasSdot(handle, x.getNumElements(), FPTR(x.getPtr()), 1, FPTR(y.getPtr()), 1, FPTR(&res));
				else IF_DOUBLE
					err = cublasDdot(handle, x.getNumElements(), DPTR(x.getPtr()), 1, DPTR(y.getPtr()), 1, DPTR(&res));
				else IF_CX_FLOAT
				{
					if(!conjugate)
						err = cublasCdotu(handle, x.getNumElements(), CPTR(x.getPtr()), 1, CPTR(y.getPtr()), 1, CPTR(&res));
					else
						err = cublasCdotc(handle, x.getNumElements(), CPTR(x.getPtr()), 1, CPTR(y.getPtr()), 1, CPTR(&res));
				}
				else IF_CX_DOUBLE
				{
					if(!conjugate)
						err = cublasZdotu(handle, x.getNumElements(), ZPTR(x.getPtr()), 1, ZPTR(y.getPtr()), 1, ZPTR(&res));
					else
						err = cublasZdotc(handle, x.getNumElements(), ZPTR(x.getPtr()), 1, ZPTR(y.getPtr()), 1, ZPTR(&res));
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
					res = cblas_sdot(x.getNumElements(), FPTR(x.getPtr()), 1, FPTR(y.getPtr()), 1);
				else IF_DOUBLE
					res = cblas_ddot(x.getNumElements(), DPTR(x.getPtr()), 1, DPTR(y.getPtr()), 1);
				else
					throw NotSupported;
			#else
				throw NotSupported;
			#endif
		}

		return res;
	}

	template<typename T, Location l>
	__host__ typename TypeInfo<T>::BaseType BLASContext::nrm2(const Accessor<T,l>& x)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_MONOLITHIC(x)
		typename TypeInfo<T>::BaseType res;

		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cublasStatus_t err;
				IF_FLOAT
					err = cublasSnrm2(handle, x.getNumElements(), FPTR(x.getPtr()), 1, FPTR(&res));
				else IF_DOUBLE
					err = cublasDnrm2(handle, x.getNumElements(), DPTR(x.getPtr()), 1, DPTR(&res));
				else IF_CX_FLOAT
					err = cublasScnrm2(handle, x.getNumElements(), CPTR(x.getPtr()), 1, FPTR(&res));
				else IF_CX_DOUBLE
					err = cublasDznrm2(handle, x.getNumElements(), ZPTR(x.getPtr()), 1, DPTR(&res));
				TEST_EXCEPTION(err)
			#else
				throw NotSupported;
			#endif
		}
		else
		{
			#ifdef KARTET_ADD_CBLAS_INTERFACE
				IF_FLOAT
					res = cblas_snrm2(x.getNumElements(), FPTR(x.getPtr()), 1);
				else IF_DOUBLE
					res = cblas_dnrm2(x.getNumElements(), DPTR(x.getPtr()), 1);
				else IF_CX_FLOAT
					res = cblas_scnrm2(x.getNumElements(), CPTR(x.getPtr()), 1);
				else IF_CX_DOUBLE
					res = cblas_dznrm2(x.getNumElements(), ZPTR(x.getPtr()), 1);
			#else
				throw NotSupported;
			#endif
		}

		return res;
	}

	template<typename T, Location l, typename TAlpha, typename TBeta>
	__host__ void BLASContext::gemv(const TAlpha& alpha, const Accessor<T,l>& A, MatrixOperation op, const Accessor<T,l>& x, const TBeta& beta, const Accessor<T,l>& y)
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
					err = cublasSgemv(handle, getCuBLASOperation(op), A.getNumRows(), A.getNumColumns(), &_alpha, FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(x.getPtr()), 1, &_beta, FPTR(y.getPtr()), 1);
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					DCST(beta)
					err = cublasDgemv(handle, getCuBLASOperation(op), A.getNumRows(), A.getNumColumns(), &_alpha, DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(x.getPtr()), 1, &_beta, DPTR(y.getPtr()), 1);
				}		
				else IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					err = cublasCgemv(handle, getCuBLASOperation(op), A.getNumRows(), A.getNumColumns(), &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(x.getPtr()), 1, &_beta, CPTR(y.getPtr()), 1);
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					err = cublasZgemv(handle, getCuBLASOperation(op), A.getNumRows(), A.getNumColumns(), &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(x.getPtr()), 1, &_beta, ZPTR(y.getPtr()), 1);
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
					cblas_sgemv(CblasColMajor, getCBLASOperation(op), A.getNumRows(), A.getNumColumns(), _alpha, FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(x.getPtr()), 1, _beta, FPTR(y.getPtr()), 1);
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					DCST(beta)
					cblas_dgemv(CblasColMajor, getCBLASOperation(op), A.getNumRows(), A.getNumColumns(), _alpha, DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(x.getPtr()), 1, _beta, DPTR(y.getPtr()), 1);
				}		
				else IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					cblas_cgemv(CblasColMajor, getCBLASOperation(op), A.getNumRows(), A.getNumColumns(), &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(x.getPtr()), 1, &_beta, CPTR(y.getPtr()), 1);
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					cblas_zgemv(CblasColMajor, getCBLASOperation(op), A.getNumRows(), A.getNumColumns(), &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(x.getPtr()), 1, &_beta, ZPTR(y.getPtr()), 1);
				}
			#else
				throw NotSupported;
			#endif
		}
	}

		template<typename T, Location l>
		__host__ void BLASContext::gemv(const Accessor<T,l>& A, MatrixOperation op, const Accessor<T,l>& x, const Accessor<T,l>& y)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			gemv(alpha, A, op, x, beta, y);
		}

		template<typename T, Location l>
		__host__ void BLASContext::gemv(const Accessor<T,l>& A, const Accessor<T,l>& x, const Accessor<T,l>& y)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			gemv(alpha, A, OpNo, x, beta, y);
		}

	template<typename T, Location l, typename TAlpha>
	void BLASContext::ger(const TAlpha& alpha, const Accessor<T,l>& x, const Accessor<T,l>& y, const Accessor<T,l>& A, bool conjugate)
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
					err = cublasSger(handle, A.getNumRows(), A.getNumColumns(), &_alpha, FPTR(x.getPtr()), 1, FPTR(y.getPtr()), 1, FPTR(A.getPtr()), A.getLeadingColumns());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					err = cublasDger(handle, A.getNumRows(), A.getNumColumns(), &_alpha, DPTR(x.getPtr()), 1, DPTR(y.getPtr()), 1, DPTR(A.getPtr()), A.getLeadingColumns());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					if(!conjugate)
						err = cublasCgeru(handle, A.getNumRows(), A.getNumColumns(), &_alpha, CPTR(x.getPtr()), 1, CPTR(y.getPtr()), 1, CPTR(A.getPtr()), A.getLeadingColumns());
					else
						err = cublasCgerc(handle, A.getNumRows(), A.getNumColumns(), &_alpha, CPTR(x.getPtr()), 1, CPTR(y.getPtr()), 1, CPTR(A.getPtr()), A.getLeadingColumns());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					if(!conjugate)
						err = cublasZgeru(handle, A.getNumRows(), A.getNumColumns(), &_alpha, ZPTR(x.getPtr()), 1, ZPTR(y.getPtr()), 1, ZPTR(A.getPtr()), A.getLeadingColumns());
					else
						err = cublasZgerc(handle, A.getNumRows(), A.getNumColumns(), &_alpha, ZPTR(x.getPtr()), 1, ZPTR(y.getPtr()), 1, ZPTR(A.getPtr()), A.getLeadingColumns());
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
					cblas_sger(CblasColMajor, A.getNumRows(), A.getNumColumns(), _alpha, FPTR(x.getPtr()), 1, FPTR(y.getPtr()), 1, FPTR(A.getPtr()), A.getLeadingColumns());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					cblas_dger(CblasColMajor, A.getNumRows(), A.getNumColumns(), _alpha, DPTR(x.getPtr()), 1, DPTR(y.getPtr()), 1, DPTR(A.getPtr()), A.getLeadingColumns());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					if(!conjugate)
						cblas_cgeru(CblasColMajor, A.getNumRows(), A.getNumColumns(), _alpha, CPTR(x.getPtr()), 1, CPTR(y.getPtr()), 1, CPTR(A.getPtr()), A.getLeadingColumns());
					else
						cblas_cgerc(CblasColMajor, A.getNumRows(), A.getNumColumns(), _alpha, CPTR(x.getPtr()), 1, CPTR(y.getPtr()), 1, CPTR(A.getPtr()), A.getLeadingColumns());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					if(!conjugate)
						cblas_zgeru(CblasColMajor, A.getNumRows(), A.getNumColumns(), _alpha, ZPTR(x.getPtr()), 1, ZPTR(y.getPtr()), 1, ZPTR(A.getPtr()), A.getLeadingColumns());
					else
						cblas_zgerc(CblasColMajor, A.getNumRows(), A.getNumColumns(), _alpha, ZPTR(x.getPtr()), 1, ZPTR(y.getPtr()), 1, ZPTR(A.getPtr()), A.getLeadingColumns());
				}
			#else
				throw NotSupported;
			#endif
		}
	}

		template<typename T, Location l>
		__host__ void ger(const Accessor<T,l>& x, const Accessor<T,l>& y, const Accessor<T,l>& A, bool conjugate)
		{
			const T alpha = complexCopy<T>(1);
			ger(alpha, x, y, A, conjugate);
		}

	template<typename T, Location l, typename TAlpha, typename TBeta>
	__host__ void BLASContext::symv(const TAlpha& alpha, MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x, const TBeta& beta, const Accessor<T,l>& y)
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
					err = cublasSsymv(handle, getCuBLASFillMode(uplo), A.getNumRows(), &_alpha, FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(x.getPtr()), 1, &_beta, FPTR(y.getPtr()), 1);
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					DCST(beta)
					err = cublasDsymv(handle, getCuBLASFillMode(uplo), A.getNumRows(), &_alpha, DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(x.getPtr()), 1, &_beta, DPTR(y.getPtr()), 1);
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					err = cublasCsymv(handle, getCuBLASFillMode(uplo), A.getNumRows(), &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(x.getPtr()), 1, &_beta, CPTR(y.getPtr()), 1);
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					err = cublasZsymv(handle, getCuBLASFillMode(uplo), A.getNumRows(), &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(x.getPtr()), 1, &_beta, ZPTR(y.getPtr()), 1);
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
					cblas_ssymv(CblasColMajor, getCBLASFillMode(uplo), A.getNumRows(), _alpha, FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(x.getPtr()), 1, _beta, FPTR(y.getPtr()), 1);
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					DCST(beta)
					cblas_dsymv(CblasColMajor, getCBLASFillMode(uplo), A.getNumRows(), _alpha, DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(x.getPtr()), 1, _beta, DPTR(y.getPtr()), 1);
				}
				else
					throw NotSupported;
			#else
				throw NotSupported;
			#endif
		}
	}

		template<typename T, Location l>
		__host__ void BLASContext::symv(MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x, const Accessor<T,l>& y)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			symv(alpha, uplo, A, x, beta, y);
		}
	
	template<typename T, Location l, typename TAlpha>
	__host__ void BLASContext::syr(const TAlpha& alpha, MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x)
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
					err = cublasSsyr(handle, getCuBLASFillMode(uplo), A.getNumRows(), &_alpha, FPTR(x.getPtr()), 1, FPTR(A.gePtr()), A.getLeadingColumns());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					err = cublasDsyr(handle, getCuBLASFillMode(uplo), A.getNumRows(), &_alpha, DPTR(x.getPtr()), 1, DPTR(A.gePtr()), A.getLeadingColumns());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					err = cublasCsyr(handle, getCuBLASFillMode(uplo), A.getNumRows(), &_alpha, CPTR(x.getPtr()), 1, CPTR(A.gePtr()), A.getLeadingColumns());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					err = cublasZsyr(handle, getCuBLASFillMode(uplo), A.getNumRows(), &_alpha, ZPTR(x.getPtr()), 1, ZPTR(A.gePtr()), A.getLeadingColumns());
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
					cblas_ssyr(CblasColMajor, getCBLASFillMode(uplo), A.getNumRows(), _alpha, FPTR(x.getPtr()), 1, FPTR(A.gePtr()), A.getLeadingColumns());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					cblas_dsyr(CblasColMajor, getCBLASFillMode(uplo), A.getNumRows(), _alpha, DPTR(x.getPtr()), 1, DPTR(A.gePtr()), A.getLeadingColumns());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					cblas_csyr(CblasColMajor, getCBLASFillMode(uplo), A.getNumRows(), &_alpha, CPTR(x.getPtr()), 1, CPTR(A.gePtr()), A.getLeadingColumns());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					cblas_zsyr(CblasColMajor, getCBLASFillMode(uplo), A.getNumRows(), &_alpha, ZPTR(x.getPtr()), 1, ZPTR(A.gePtr()), A.getLeadingColumns());
				}
			#else
				throw NotSupported;
			#endif
		}
	}

		template<typename T, Location l>
		__host__ void BLASContext::syr(MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x)
		{
			const T alpha = complexCopy<T>(1);
			syr(alpha, uplo, A, x);
		}

	template<typename T, Location l, typename TAlpha>
	__host__ void BLASContext::syr2(const TAlpha& alpha, MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x, const Accessor<T,l>& y)
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
					err = cublasSsyr2(handle, getCuBLASFillMode(uplo), A.getNumRows(), &_alpha, FPTR(x.getPtr()), 1, FPTR(y.getPtr()), 1, FPTR(A.gePtr()), A.getLeadingColumns());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					err = cublasDsyr2(handle, getCuBLASFillMode(uplo), A.getNumRows(), &_alpha, DPTR(x.getPtr()), 1, DPTR(y.getPtr()), 1, DPTR(A.gePtr()), A.getLeadingColumns());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					err = cublasCsyr2(handle, getCuBLASFillMode(uplo), A.getNumRows(), &_alpha, CPTR(x.getPtr()), 1, CPTR(y.getPtr()), 1, CPTR(A.gePtr()), A.getLeadingColumns());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					err = cublasZsyr2(handle, getCuBLASFillMode(uplo), A.getNumRows(), &_alpha, ZPTR(x.getPtr()), 1, ZPTR(y.getPtr()), 1, ZPTR(A.gePtr()), A.getLeadingColumns());
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
					cblas_ssyr2(CblasColMajor, getCBLASFillMode(uplo), A.getNumRows(), _alpha, FPTR(x.getPtr()), 1, FPTR(y.getPtr()), 1, FPTR(A.gePtr()), A.getLeadingColumns());
				}
				else IF_DOUBLE
				{
					FCST(alpha)
					cblas_ssyr2(CblasColMajor, getCBLASFillMode(uplo), A.getNumRows(), _alpha, DPTR(x.getPtr()), 1, DPTR(y.getPtr()), 1, DPTR(A.gePtr()), A.getLeadingColumns());
				}
				else
					throw NotSupported;
			#else
				throw NotSupported;
			#endif
		}
	}

		template<typename T, Location l>
		__host__ void BLASContext::syr2(MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x, const Accessor<T,l>& y)
		{
			const T alpha = complexCopy<T>(1);
			syr2(alpha, uplo, A, x, y);
		}

	template<typename T, Location l>
	__host__ void BLASContext::trmv(MatrixFillMode uplo, MatrixDiagType diag, const Accessor<T,l>& A, MatrixOperation op, const Accessor<T,l>& x)
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
					err = cublasStrmv(handle, getCuBLASFillMode(uplo), getCuBLASOperation(op), getCuBLASDiagType(diag), A.getNumRows(), FPTR(A.gePtr()), A.getLeadingColumns(), FPTR(x.getPtr()), 1);
				else IF_DOUBLE
					err = cublasStrmv(handle, getCuBLASFillMode(uplo), getCuBLASOperation(op), getCuBLASDiagType(diag), A.getNumRows(), DPTR(A.gePtr()), A.getLeadingColumns(), DPTR(x.getPtr()), 1);
				else IF_CX_FLOAT
					err = cublasStrmv(handle, getCuBLASFillMode(uplo), getCuBLASOperation(op), getCuBLASDiagType(diag), A.getNumRows(), CPTR(A.gePtr()), A.getLeadingColumns(), CPTR(x.getPtr()), 1);
				else IF_CX_DOUBLE
					err = cublasStrmv(handle, getCuBLASFillMode(uplo), getCuBLASOperation(op), getCuBLASDiagType(diag), A.getNumRows(), ZPTR(A.gePtr()), A.getLeadingColumns(), ZPTR(x.getPtr()), 1);
				TEST_EXCEPTION(err)
			#else
				throw NotSupported;
			#endif
		}
		else
		{
			#ifdef KARTET_ADD_CBLAS_INTERFACE
				IF_FLOAT
					cblas_strmv(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(op), getCBLASDiagType(diag), A.getNumRows(), FPTR(A.gePtr()), A.getLeadingColumns(), FPTR(x.getPtr()), 1);
				else IF_DOUBLE
					cblas_dtrmv(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(op), getCBLASDiagType(diag), A.getNumRows(), DPTR(A.gePtr()), A.getLeadingColumns(), DPTR(x.getPtr()), 1);
				else IF_CX_FLOAT
					cblas_ctrmv(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(op), getCBLASDiagType(diag), A.getNumRows(), CPTR(A.gePtr()), A.getLeadingColumns(), CPTR(x.getPtr()), 1);
				else IF_CX_DOUBLE
					cblas_ztrmv(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(op), getCBLASDiagType(diag), A.getNumRows(), ZPTR(A.gePtr()), A.getLeadingColumns(), ZPTR(x.getPtr()), 1);
			#else
				throw NotSupported;
			#endif
		}
	}

	template<typename T, Location l>
	__host__ void BLASContext::trsv(MatrixFillMode uplo, MatrixDiagType diag, const Accessor<T,l>& A, MatrixOperation op, const Accessor<T,l>& x)
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
					err = cublasStrsv(handle, getCuBLASFillMode(uplo), getCuBLASOperation(op), getCuBLASDiagType(diag), A.getNumRows(), FPTR(A.gePtr()), A.getLeadingColumns(), FPTR(x.getPtr()), 1);
				else IF_DOUBLE
					err = cublasStrsv(handle, getCuBLASFillMode(uplo), getCuBLASOperation(op), getCuBLASDiagType(diag), A.getNumRows(), DPTR(A.gePtr()), A.getLeadingColumns(), DPTR(x.getPtr()), 1);
				else IF_CX_FLOAT
					err = cublasStrsv(handle, getCuBLASFillMode(uplo), getCuBLASOperation(op), getCuBLASDiagType(diag), A.getNumRows(), CPTR(A.gePtr()), A.getLeadingColumns(), CPTR(x.getPtr()), 1);
				else IF_CX_DOUBLE
					err = cublasStrsv(handle, getCuBLASFillMode(uplo), getCuBLASOperation(op), getCuBLASDiagType(diag), A.getNumRows(), ZPTR(A.gePtr()), A.getLeadingColumns(), ZPTR(x.getPtr()), 1);
				TEST_EXCEPTION(err)
			#else
				throw NotSupported;
			#endif
		}
		else
		{
			#ifdef KARTET_ADD_CBLAS_INTERFACE
				IF_FLOAT
					cblas_strsv(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(op), getCBLASDiagType(diag), A.getNumRows(), FPTR(A.gePtr()), A.getLeadingColumns(), FPTR(x.getPtr()), 1);
				else IF_DOUBLE
					cblas_dtrsv(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(op), getCBLASDiagType(diag), A.getNumRows(), DPTR(A.gePtr()), A.getLeadingColumns(), DPTR(x.getPtr()), 1);
				else IF_CX_FLOAT
					cblas_ctrsv(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(op), getCBLASDiagType(diag), A.getNumRows(), CPTR(A.gePtr()), A.getLeadingColumns(), CPTR(x.getPtr()), 1);
				else IF_CX_DOUBLE
					cblas_ztrsv(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(op), getCBLASDiagType(diag), A.getNumRows(), ZPTR(A.gePtr()), A.getLeadingColumns(), ZPTR(x.getPtr()), 1);
			#else
				throw NotSupported;
			#endif
		}
	}

	template<typename T, Location l, typename TAlpha, typename TBeta>
	__host__ void BLASContext::hemv(const TAlpha& alpha, MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x, const TBeta& beta, const Accessor<T,l>& y)
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
					err = cublasChemv(handle, getCuBLASFillMode(uplo), A.getNumRows(), &_alpha, CPTR(A.gePtr()), A.getLeadingColumns(), CPTR(x.getPtr()), 1, &_beta, CPTR(y.getPtr()), 1);
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					err = cublasZhemv(handle, getCuBLASFillMode(uplo), A.getNumRows(), &_alpha, ZPTR(A.gePtr()), A.getLeadingColumns(), ZPTR(x.getPtr()), 1, &_beta, ZPTR(y.getPtr()), 1);
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
					cblas_chemv(CblasColMajor, getCBLASFillMode(uplo), A.getNumRows(), &_alpha, CPTR(A.gePtr()), A.getLeadingColumns(), CPTR(x.getPtr()), 1, &_beta, CPTR(y.getPtr()), 1);
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					cblas_zhemv(CblasColMajor, getCBLASFillMode(uplo), A.getNumRows(), &_alpha, ZPTR(A.gePtr()), A.getLeadingColumns(), ZPTR(x.getPtr()), 1, &_beta, ZPTR(y.getPtr()), 1);
				}
			#else
				throw NotSupported;
			#endif
		}
	}

		template<typename T, Location l>
		__host__ void BLASContext::hemv(MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x, const Accessor<T,l>& y)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			hemv(alpha, uplo, A, x, beta, y);
		}

	template<typename T, Location l, typename TAlpha>
	__host__ void BLASContext::her(const TAlpha& alpha, MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x)
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
					err = cublasCher(handle, getCuBLASFillMode(uplo), A.getNumRows(), &_alpha, CPTR(x.getPtr()), 1, CPTR(A.gePtr()), A.getLeadingColumns());
				}                
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					err = cublasZher(handle, getCuBLASFillMode(uplo), A.getNumRows(), &_alpha, ZPTR(x.getPtr()), 1, ZPTR(A.gePtr()), A.getLeadingColumns());
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
					cblas_cher(CblasColMajor, getCBLASFillMode(uplo), A.getNumRows(), &_alpha, CPTR(x.getPtr()), 1, CPTR(A.gePtr()), A.getLeadingColumns());
				}                
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					cblas_zher(CblasColMajor, getCBLASFillMode(uplo), A.getNumRows(), &_alpha, ZPTR(x.getPtr()), 1, ZPTR(A.gePtr()), A.getLeadingColumns());
				}
			#else
				throw NotSupported;
			#endif
		}
	}

		template<typename T, Location l>
		__host__ void BLASContext::her(MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x)
		{
			const T alpha = complexCopy<T>(1);
			her(alpha, uplo, A, x);
		}

	template<typename T, Location l, typename TAlpha>
	__host__ void BLASContext::her2(const TAlpha& alpha, MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x, const Accessor<T,l>& y)
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
					err = cublasCher2(handle, getCuBLASFillMode(uplo), A.getNumRows(), &_alpha, CPTR(x.getPtr()), 1, CPTR(y.getPtr()), 1, CPTR(A.gePtr()), A.getLeadingColumns());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					err = cublasCher2(handle, getCuBLASFillMode(uplo), A.getNumRows(), &_alpha, ZPTR(x.getPtr()), 1, ZPTR(y.getPtr()), 1, CPTR(A.gePtr()), A.getLeadingColumns());
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
					cblas_cher2(CblasColMajor, getCBLASFillMode(uplo), A.getNumRows(), &_alpha, CPTR(x.getPtr()), 1, CPTR(y.getPtr()), 1, CPTR(A.gePtr()), A.getLeadingColumns());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					cblas_zher2(CblasColMajor, getCBLASFillMode(uplo), A.getNumRows(), &_alpha, ZPTR(x.getPtr()), 1, ZPTR(y.getPtr()), 1, ZPTR(A.gePtr()), A.getLeadingColumns());
				}
			#else
				throw NotSupported;
			#endif
		}
	}

		template<typename T, Location l>
		__host__ void BLASContext::her2(MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x, const Accessor<T,l>& y)
		{
			const T alpha = complexCopy<T>(1);
			her2(alpha, uplo, A, x, y);
		}

	template<typename T, Location l, typename TAlpha, typename TBeta>
	__host__ void BLASContext::gemm(const TAlpha& alpha, const Accessor<T,l>& A, MatrixOperation opa, const Accessor<T,l>& B, MatrixOperation opb, const TBeta& beta, const Accessor<T,l>& C)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(B)
		TEST_SINGLE_SLICE(C)
		TEST_PRODUCT(A, opa, B, opb, C)
		const int k = (opa==OpTr || opa==OpHr) ? A.getNumRows() : A.getNumColumns();

		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cublasStatus_t err;			
				IF_FLOAT
				{
					FCST(alpha)
					FCST(beta)
					err = cublasSgemm(handle, getCuBLASOperation(opa), getCuBLASOperation(opb), C.getNumRows(), C.getNumColumns(), k, &_alpha, FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(B.getPtr()), B.getLeadingColumns(), &_beta, FPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					DCST(beta)
					err = cublasDgemm(handle, getCuBLASOperation(opa), getCuBLASOperation(opb), C.getNumRows(), C.getNumColumns(), k, &_alpha, DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(B.getPtr()), B.getLeadingColumns(), &_beta, DPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					err = cublasCgemm(handle, getCuBLASOperation(opa), getCuBLASOperation(opb), C.getNumRows(), C.getNumColumns(), k, &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(B.getPtr()), B.getLeadingColumns(), &_beta, CPTR(C.getPtr()), C.getLeadingColumns());
				}		
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					err = cublasZgemm(handle, getCuBLASOperation(opa), getCuBLASOperation(opb), C.getNumRows(), C.getNumColumns(), k, &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(B.getPtr()), B.getLeadingColumns(), &_beta, ZPTR(C.getPtr()), C.getLeadingColumns());
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
					cblas_sgemm(CblasColMajor, getCBLASOperation(opa), getCBLASOperation(opb), C.getNumRows(), C.getNumColumns(), k, _alpha, FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(B.getPtr()), B.getLeadingColumns(), _beta, FPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					DCST(beta)
					cblas_dgemm(CblasColMajor, getCBLASOperation(opa), getCBLASOperation(opb), C.getNumRows(), C.getNumColumns(), k, _alpha, DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(B.getPtr()), B.getLeadingColumns(), _beta, DPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					cblas_cgemm(CblasColMajor, getCBLASOperation(opa), getCBLASOperation(opb), C.getNumRows(), C.getNumColumns(), k, &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(B.getPtr()), B.getLeadingColumns(), &_beta, CPTR(C.getPtr()), C.getLeadingColumns());
				}		
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					cblas_zgemm(CblasColMajor, getCBLASOperation(opa), getCBLASOperation(opb), C.getNumRows(), C.getNumColumns(), k, &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(B.getPtr()), B.getLeadingColumns(), &_beta, ZPTR(C.getPtr()), C.getLeadingColumns());
				}
			#else
				throw NotSupported;
			#endif
		}
	}

		template<typename T, Location l>
		__host__ void BLASContext::gemm(const Accessor<T,l>& A, MatrixOperation opa, const Accessor<T,l>& B, MatrixOperation opb, const Accessor<T,l>& C)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			gemm(alpha, A, opa, B, opb, beta, C);
		}

		template<typename T, Location l>
		__host__ void BLASContext::gemm(const Accessor<T,l>& A, const Accessor<T,l>& B, const Accessor<T,l>& C)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			gemm(alpha, A, OpNo, B, OpNo, beta, C);
		}

	template<typename T, Location l, typename TAlpha, typename TBeta>
	__host__ void BLASContext::symm(MatrixSideMode side, const TAlpha& alpha, MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& B, const TBeta& beta, const Accessor<T,l>& C)
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
					err = cublasSsymm(handle, getCuBLASSideMode(side), getCuBLASFillMode(uplo), B.getNumRows(), C.getNumColumns(), &_alpha, FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(B.getPtr()), B.getLeadingColumns(), &_beta, FPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					DCST(beta)
					err = cublasDsymm(handle, getCuBLASSideMode(side), getCuBLASFillMode(uplo), B.getNumRows(), C.getNumColumns(), &_alpha, DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(B.getPtr()), B.getLeadingColumns(), &_beta, DPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					err = cublasCsymm(handle, getCuBLASSideMode(side), getCuBLASFillMode(uplo), B.getNumRows(), C.getNumColumns(), &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(B.getPtr()), B.getLeadingColumns(), &_beta, CPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					err = cublasZsymm(handle, getCuBLASSideMode(side), getCuBLASFillMode(uplo), B.getNumRows(), C.getNumColumns(), &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(B.getPtr()), B.getLeadingColumns(), &_beta, ZPTR(C.getPtr()), C.getLeadingColumns());
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
					cblas_ssymm(CblasColMajor, getCBLASSideMode(side), getCBLASFillMode(uplo), B.getNumRows(), C.getNumColumns(), _alpha, FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(B.getPtr()), B.getLeadingColumns(), _beta, FPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					DCST(beta)
					cblas_dsymm(CblasColMajor, getCBLASSideMode(side), getCBLASFillMode(uplo), B.getNumRows(), C.getNumColumns(), _alpha, DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(B.getPtr()), B.getLeadingColumns(), _beta, DPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					cblas_csymm(CblasColMajor, getCBLASSideMode(side), getCBLASFillMode(uplo), B.getNumRows(), C.getNumColumns(), &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(B.getPtr()), B.getLeadingColumns(), &_beta, CPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					cblas_zsymm(CblasColMajor, getCBLASSideMode(side), getCBLASFillMode(uplo), B.getNumRows(), C.getNumColumns(), &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(B.getPtr()), B.getLeadingColumns(), &_beta, ZPTR(C.getPtr()), C.getLeadingColumns());
				}
			#else
				throw NotSupported;
			#endif
		}
	}

		template<typename T, Location l>
		__host__ void BLASContext::symm(MatrixSideMode side, MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& B, const Accessor<T,l>& C)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			symm(side, alpha, uplo, A, B, beta, C);
		}

	template<typename T, Location l, typename TAlpha, typename TBeta>
	__host__ void BLASContext::syrk(const TAlpha& alpha, const Accessor<T,l>& A, MatrixOperation opa, const TBeta& beta, MatrixFillMode uplo, const Accessor<T,l>& C)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(C)
		if(opa==OpNo)
			TEST_PRODUCT(A, OpNo, A, OpTr, C)
		else
			TEST_PRODUCT(A, OpTr, A, OpNo, C)
		const int k = (opa==OpTr || opa==OpHr) ? A.getNumRows() : A.getNumColumns();

		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cublasStatus_t err;		
				IF_FLOAT
				{
					FCST(alpha)
					FCST(beta)
					err = cublasSsyrk(handle, getCuBLASFillMode(uplo), getCuBLASOperation(opa), C.getNumRows(), k, &_alpha, FPTR(A.getPtr()), A.getLeadingColumns(), &_beta, FPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					DCST(beta)
					err = cublasDsyrk(handle, getCuBLASFillMode(uplo), getCuBLASOperation(opa), C.getNumRows(), k, &_alpha, DPTR(A.getPtr()), A.getLeadingColumns(), &_beta, DPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					err = cublasCsyrk(handle, getCuBLASFillMode(uplo), getCuBLASOperation(opa), C.getNumRows(), k, &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), &_beta, CPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					err = cublasZsyrk(handle, getCuBLASFillMode(uplo), getCuBLASOperation(opa), C.getNumRows(), k, &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), &_beta, ZPTR(C.getPtr()), C.getLeadingColumns());
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
					cblas_ssyrk(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(opa), C.getNumRows(), k, _alpha, FPTR(A.getPtr()), A.getLeadingColumns(), _beta, FPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					DCST(beta)
					cblas_dsyrk(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(opa), C.getNumRows(), k, _alpha, DPTR(A.getPtr()), A.getLeadingColumns(), _beta, DPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					cblas_csyrk(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(opa), C.getNumRows(), k, &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), &_beta, CPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					cblas_zsyrk(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(opa), C.getNumRows(), k, &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), &_beta, ZPTR(C.getPtr()), C.getLeadingColumns());
				}
			#else
				throw NotSupported;
			#endif
		}
	}

		template<typename T, Location l>
		__host__ void BLASContext::syrk(const Accessor<T,l>& A, MatrixOperation opa, MatrixFillMode uplo, const Accessor<T,l>& C)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			syrk(alpha, A, opa, beta, uplo, C);
		}

		template<typename T, Location l>
		__host__ void BLASContext::syrk(const Accessor<T,l>& A, MatrixFillMode uplo, const Accessor<T,l>& C)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			syrk(alpha, A, OpNo, beta, uplo, C);
		}

	template<typename T, Location l, typename TAlpha>
	__host__ void BLASContext::syr2k(MatrixOperation op, const TAlpha& alpha, const Accessor<T,l>& A, const Accessor<T,l>& B, const T& beta, MatrixFillMode uplo, const Accessor<T,l>& C)
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
		const int k = (op==OpTr || op==OpHr) ? A.getNumRows() : A.getNumColumns();

		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cublasStatus_t err;		
				IF_FLOAT
				{
					FCST(alpha)
					FCST(beta)
					err = cublasSsyr2k(handle, getCuBLASFillMode(uplo), getCuBLASOperation(op), C.getNumRows(), k, &_alpha, FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(B.getPtr()), B.getLeadingColumns(), &_beta, FPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					DCST(beta)
					err = cublasDsyr2k(handle, getCuBLASFillMode(uplo), getCuBLASOperation(op), C.getNumRows(), k, &_alpha, DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(B.getPtr()), B.getLeadingColumns(), &_beta, DPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					err = cublasCsyr2k(handle, getCuBLASFillMode(uplo), getCuBLASOperation(op), C.getNumRows(), k, &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(B.getPtr()), B.getLeadingColumns(), &_beta, CPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					err = cublasZsyr2k(handle, getCuBLASFillMode(uplo), getCuBLASOperation(op), C.getNumRows(), k, &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(B.getPtr()), B.getLeadingColumns(), &_beta, ZPTR(C.getPtr()), C.getLeadingColumns());
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
					cblas_ssyr2k(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(op), C.getNumRows(), k, _alpha, FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(B.getPtr()), B.getLeadingColumns(), _beta, FPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					DCST(beta)
					cblas_dsyr2k(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(op), C.getNumRows(), k, _alpha, DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(B.getPtr()), B.getLeadingColumns(), _beta, DPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					cblas_csyr2k(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(op), C.getNumRows(), k, &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(B.getPtr()), B.getLeadingColumns(), &_beta, CPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					cblas_zsyr2k(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(op), C.getNumRows(), k, &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(B.getPtr()), B.getLeadingColumns(), &_beta, ZPTR(C.getPtr()), C.getLeadingColumns());
				}
			#else
				throw NotSupported;
			#endif
		}
	}

		template<typename T, Location l>
		__host__ void BLASContext::syr2k(MatrixOperation op, const Accessor<T,l>& A, const Accessor<T,l>& B, MatrixFillMode uplo, const Accessor<T,l>& C)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			syr2k(op, alpha, A, B, beta, uplo, C);
		}

		template<typename T, Location l>
		__host__ void BLASContext::syr2k(const Accessor<T,l>& A, const Accessor<T,l>& B, MatrixFillMode uplo, const Accessor<T,l>& C)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			syr2k(OpNo, alpha, A, B, beta, uplo, C);
		}

	template<typename T, Location l, typename TAlpha>
	__host__ void BLASContext::trmm(MatrixSideMode side, const TAlpha& alpha, MatrixFillMode uplo, MatrixDiagType diag, const Accessor<T,l>& A, MatrixOperation opa, const Accessor<T,l>& B, const Accessor<T,l>& C)
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
					err = cublasStrmm(handle, getCuBLASSideMode(side), getCuBLASFillMode(uplo), getCuBLASOperation(opa), getCuBLASDiagType(diag), B.getNumRows(), B.getNumColumns(), &_alpha, FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(B.getPtr()), B.getLeadingColumns(), FPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					err = cublasDtrmm(handle, getCuBLASSideMode(side), getCuBLASFillMode(uplo), getCuBLASOperation(opa), getCuBLASDiagType(diag), B.getNumRows(), B.getNumColumns(), &_alpha, DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(B.getPtr()), B.getLeadingColumns(), DPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					err = cublasCtrmm(handle, getCuBLASSideMode(side), getCuBLASFillMode(uplo), getCuBLASOperation(opa), getCuBLASDiagType(diag), B.getNumRows(), B.getNumColumns(), &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(B.getPtr()), B.getLeadingColumns(), CPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					err = cublasZtrmm(handle, getCuBLASSideMode(side), getCuBLASFillMode(uplo), getCuBLASOperation(opa), getCuBLASDiagType(diag), B.getNumRows(), B.getNumColumns(), &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(B.getPtr()), B.getLeadingColumns(), ZPTR(C.getPtr()), C.getLeadingColumns());
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
					cblas_strmm(CblasColMajor, getCBLASSideMode(side), getCBLASFillMode(uplo), getCBLASOperation(opa), getCBLASDiagType(diag), B.getNumRows(), B.getNumColumns(), _alpha, FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(B.getPtr()), B.getLeadingColumns(), FPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					cblas_dtrmm(CblasColMajor, getCBLASSideMode(side), getCBLASFillMode(uplo), getCBLASOperation(opa), getCBLASDiagType(diag), B.getNumRows(), B.getNumColumns(), _alpha, DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(B.getPtr()), B.getLeadingColumns(), DPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					cblas_ctrmm(CblasColMajor, getCBLASSideMode(side), getCBLASFillMode(uplo), getCBLASOperation(opa), getCBLASDiagType(diag), B.getNumRows(), B.getNumColumns(), &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(B.getPtr()), B.getLeadingColumns(), CPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					cblas_ztrmm(CblasColMajor, getCBLASSideMode(side), getCBLASFillMode(uplo), getCBLASOperation(opa), getCBLASDiagType(diag), B.getNumRows(), B.getNumColumns(), &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(B.getPtr()), B.getLeadingColumns(), ZPTR(C.getPtr()), C.getLeadingColumns());
				}
			#else
				throw NotSupported;
			#endif
		}
	}

		template<typename T, Location l>
		__host__ void BLASContext::trmm(MatrixSideMode side, MatrixFillMode uplo, MatrixDiagType diag, const Accessor<T,l>& A, MatrixOperation opa, const Accessor<T,l>& B, const Accessor<T,l>& C)
		{
			const T alpha = complexCopy<T>(1);
			trmm(side, alpha, uplo, diag, A, opa, B, C);
		}
		
		template<typename T, Location l>
		__host__ void BLASContext::trmm(MatrixSideMode side, MatrixFillMode uplo, MatrixDiagType diag, const Accessor<T,l>& A, const Accessor<T,l>& B, const Accessor<T,l>& C)
		{
			const T alpha = complexCopy<T>(1);
			trmm(side, alpha, uplo, diag, A, OpNo, B, C);
		}
		
	template<typename T, Location l, typename TAlpha>
	__host__ void BLASContext::trsm(MatrixSideMode side, MatrixFillMode uplo, MatrixDiagType diag, const Accessor<T,l>& A, MatrixOperation opa, const TAlpha& alpha, const Accessor<T,l>& B)
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
					err = cublasStrsm(handle, getCuBLASSideMode(side), getCuBLASFillMode(uplo), getCuBLASOperation(opa), getCuBLASDiagType(diag), B.getNumRows(), B.getNumColumns(), &_alpha, FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(B.getPtr()), B.getLeadingColumns());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					err = cublasDtrsm(handle, getCuBLASSideMode(side), getCuBLASFillMode(uplo), getCuBLASOperation(opa), getCuBLASDiagType(diag), B.getNumRows(), B.getNumColumns(), &_alpha, DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(B.getPtr()), B.getLeadingColumns());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					err = cublasCtrsm(handle, getCuBLASSideMode(side), getCuBLASFillMode(uplo), getCuBLASOperation(opa), getCuBLASDiagType(diag), B.getNumRows(), B.getNumColumns(), &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(B.getPtr()), B.getLeadingColumns());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					err = cublasZtrsm(handle, getCuBLASSideMode(side), getCuBLASFillMode(uplo), getCuBLASOperation(opa), getCuBLASDiagType(diag), B.getNumRows(), B.getNumColumns(), &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(B.getPtr()), B.getLeadingColumns());
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
					cblas_strsm(CblasColMajor, getCBLASSideMode(side), getCBLASFillMode(uplo), getCBLASOperation(opa), getCBLASDiagType(diag), B.getNumRows(), B.getNumColumns(), _alpha, FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(B.getPtr()), B.getLeadingColumns());
				}
				else IF_DOUBLE
				{
					DCST(alpha)
					cblas_dtrsm(CblasColMajor, getCBLASSideMode(side), getCBLASFillMode(uplo), getCBLASOperation(opa), getCBLASDiagType(diag), B.getNumRows(), B.getNumColumns(), _alpha, DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(B.getPtr()), B.getLeadingColumns());
				}
				else IF_CX_FLOAT
				{
					CCST(alpha)
					cblas_ctrsm(CblasColMajor, getCBLASSideMode(side), getCBLASFillMode(uplo), getCBLASOperation(opa), getCBLASDiagType(diag), B.getNumRows(), B.getNumColumns(), &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(B.getPtr()), B.getLeadingColumns());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					cblas_ztrsm(CblasColMajor, getCBLASSideMode(side), getCBLASFillMode(uplo), getCBLASOperation(opa), getCBLASDiagType(diag), B.getNumRows(), B.getNumColumns(), &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(B.getPtr()), B.getLeadingColumns());
				}
			#else
				throw NotSupported;
			#endif
		}
	}

		template<typename T, Location l>
		__host__ void BLASContext::trsm(MatrixSideMode side, MatrixFillMode uplo, MatrixDiagType diag, const Accessor<T,l>& A, MatrixOperation opa, const Accessor<T,l>& B)
		{
			const T alpha = complexCopy<T>(1);
			trsm(side, uplo, diag, A, opa, alpha, B);
		}

		template<typename T, Location l>
		__host__ void BLASContext::trsm(MatrixSideMode side, MatrixFillMode uplo, MatrixDiagType diag, const Accessor<T,l>& A, const Accessor<T,l>& B)
		{
			const T alpha = complexCopy<T>(1);
			trsm(side, uplo, diag, A, OpNo, alpha, B);
		}

	template<typename T, Location l, typename TAlpha, typename TBeta>
	__host__ void BLASContext::hemm(MatrixSideMode side, const TAlpha& alpha, MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& B, const TBeta& beta, const Accessor<T,l>& C)
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
					err = cublasChemm(handle, getCuBLASSideMode(side), getCuBLASFillMode(uplo), C.getNumRows(), C.getNumColumns(), &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(B.getPtr()), B.getLeadingColumns(), &_beta, CPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					err = cublasZhemm(handle, getCuBLASSideMode(side), getCuBLASFillMode(uplo), C.getNumRows(), C.getNumColumns(), &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(B.getPtr()), B.getLeadingColumns(), &_beta, ZPTR(C.getPtr()), C.getLeadingColumns());
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
					cblas_chemm(CblasColMajor, getCBLASSideMode(side), getCBLASFillMode(uplo), C.getNumRows(), C.getNumColumns(), &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(B.getPtr()), B.getLeadingColumns(), &_beta, CPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					cblas_zhemm(CblasColMajor, getCBLASSideMode(side), getCBLASFillMode(uplo), C.getNumRows(), C.getNumColumns(), &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(B.getPtr()), B.getLeadingColumns(), &_beta, ZPTR(C.getPtr()), C.getLeadingColumns());
				}
			#else
				throw NotSupported;
			#endif
		}
	}

		template<typename T, Location l>
		__host__ void BLASContext::hemm(MatrixSideMode side, MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& B, const Accessor<T,l>& C)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			hemm(side, alpha, uplo, A, B, beta, C);
		}

	template<typename T, Location l, typename TAlpha, typename TBeta>
	__host__ void BLASContext::herk(const TAlpha& alpha, const Accessor<T,l>& A, MatrixOperation opa, const TBeta& beta, MatrixFillMode uplo, const Accessor<T,l>& C)
	{
		ALLOWED_TYPES_VERIFICATION
		TYPE_MUST_BE_COMPLEX
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(C)
		if(opa==OpNo)
			TEST_PRODUCT(A, OpNo, A, OpHr, C)
		else
			TEST_PRODUCT(A, OpHr, A, OpNo, C)
		int k = (opa==OpTr || opa==OpHr) ? A.getNumRows() : A.getNumColumns();

		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cublasStatus_t err;
				IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					err = cublasCherk(handle, getCuBLASFillMode(uplo), getCuBLASOperation(opa), C.getNumRows(), k, &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), &_beta, CPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					err = cublasZherk(handle, getCuBLASFillMode(uplo), getCuBLASOperation(opa), C.getNumRows(), k, &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), &_beta, ZPTR(C.getPtr()), C.getLeadingColumns());
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
					cblas_cherk(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(opa), C.getNumRows(), k, &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), &_beta, CPTR(C.getPtr()), C.getLeadingColumns());
				}
				else IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					cblas_zherk(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(opa), C.getNumRows(), k, &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), &_beta, ZPTR(C.getPtr()), C.getLeadingColumns());
				}
			#else
				throw NotSupported;
			#endif
		}
	}

		template<typename T, Location l>
		__host__ void BLASContext::herk(const Accessor<T,l>& A, MatrixOperation opa, MatrixFillMode uplo, const Accessor<T,l>& C)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			herk(alpha, A, opa, beta, uplo, C);
		}

		template<typename T, Location l>
		__host__ void BLASContext::herk(const Accessor<T,l>& A, MatrixFillMode uplo, const Accessor<T,l>& C)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			herk(alpha, A, OpNo, beta, uplo, C);
		}		

	template<typename T, Location l, typename TAlpha, typename TBeta>
	__host__ void BLASContext::her2k(MatrixOperation op, const TAlpha& alpha, const Accessor<T,l>& A, const Accessor<T,l>& B, const TBeta& beta, MatrixFillMode uplo, const Accessor<T,l>& C)
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
		const int k = (op==OpTr || op==OpHr) ? A.getNumRows() : A.getNumColumns();

		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				cublasStatus_t err;
				IF_CX_FLOAT
				{
					CCST(alpha)
					CCST(beta)
					err = cublasCher2k(handle, getCuBLASFillMode(uplo), getCuBLASOperation(op), C.getNumRows(), k, &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(B.getPtr()), B.getLeadingColumns(), &_beta, CPTR(C.getPtr()), C.getLeadingColumns());
				}
				IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					err = cublasZher2k(handle, getCuBLASFillMode(uplo), getCuBLASOperation(op), C.getNumRows(), k, &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(B.getPtr()), B.getLeadingColumns(), &_beta, ZPTR(C.getPtr()), C.getLeadingColumns());
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
					cblas_cher2k(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(op), C.getNumRows(), k, &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(B.getPtr()), B.getLeadingColumns(), &_beta, CPTR(C.getPtr()), C.getLeadingColumns());
				}
				IF_CX_DOUBLE
				{
					ZCST(alpha)
					ZCST(beta)
					cblas_zher2k(CblasColMajor, getCBLASFillMode(uplo), getCBLASOperation(op), C.getNumRows(), k, &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(B.getPtr()), B.getLeadingColumns(), &_beta, ZPTR(C.getPtr()), C.getLeadingColumns());
				}
			#else
				throw NotSupported;
			#endif
		}
	}

		template<typename T, Location l>
		__host__ void BLASContext::her2k(MatrixOperation op, const Accessor<T,l>& A, const Accessor<T,l>& B, MatrixFillMode uplo, const Accessor<T,l>& C)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			her2k(op, alpha, A, B, beta, uplo, C);
		}

		template<typename T, Location l>
		__host__ void BLASContext::her2k(const Accessor<T,l>& A, const Accessor<T,l>& B, MatrixFillMode uplo, const Accessor<T,l>& C)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			her2k(OpNo, alpha, A, B, beta, uplo, C);
		}

	template<typename T, Location l, typename TAlpha, typename TBeta>
	__host__ void BLASContext::geam(const TAlpha& alpha, const Accessor<T,l>& A, MatrixOperation opa, const TBeta& beta, const Accessor<T,l>& B, MatrixOperation opb, const Accessor<T,l>& C)
	{
		StaticAssert<l==DeviceSide>();
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
				err = cublasSgeam(handle, getCuBLASOperation(opa), getCuBLASOperation(opb), C.getNumRows(), C.getNumColumns(), FPTR(&alpha), FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(&beta), FPTR(B.getPtr()), B.getLeadingColumns(), FPTR(C.getPtr()), C.getLeadingColumns());
			}
			else IF_DOUBLE
			{
				DCST(alpha)
				DCST(beta)
				err = cublasDgeam(handle, getCuBLASOperation(opa), getCuBLASOperation(opb), C.getNumRows(), C.getNumColumns(), DPTR(&alpha), DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(&beta), DPTR(B.getPtr()), B.getLeadingColumns(), DPTR(C.getPtr()), C.getLeadingColumns());
			}
			else IF_CX_FLOAT
			{
				CCST(alpha)
				CCST(beta)
				err = cublasCgeam(handle, getCuBLASOperation(opa), getCuBLASOperation(opb), C.getNumRows(), C.getNumColumns(), CPTR(&alpha), CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(&beta), CPTR(B.getPtr()), B.getLeadingColumns(), CPTR(C.getPtr()), C.getLeadingColumns());
			}
			else IF_CX_DOUBLE
			{
				ZCST(alpha)
				ZCST(beta)
				err = cublasZgeam(handle, getCuBLASOperation(opa), getCuBLASOperation(opb), C.getNumRows(), C.getNumColumns(), ZPTR(&alpha), ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(&beta), ZPTR(B.getPtr()), B.getLeadingColumns(), ZPTR(C.getPtr()), C.getLeadingColumns());
			}
			TEST_EXCEPTION(err)
		#else
			throw NotSupported;
		#endif
	}

		template<typename T, Location l>
		__host__ void BLASContext::geam(const Accessor<T,l>& A, MatrixOperation opa, const Accessor<T,l>& B, MatrixOperation opb, const Accessor<T,l>& C)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			geam(alpha, A, opa, beta, B, opb, C);
		}

		template<typename T, Location l>
		__host__ void BLASContext::geam(const Accessor<T,l>& A, const Accessor<T,l>& B, const Accessor<T,l>& C)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			geam(alpha, A, OpNo, beta, B, OpNo, C);
		}

	template<typename T, Location l>
	__host__ void BLASContext::dgmm(MatrixSideMode side, const Accessor<T,l>& A, const Accessor<T,l>& x, const Accessor<T,l>& C)
	{
		StaticAssert<l==DeviceSide>();
		ALLOWED_TYPES_VERIFICATION
		TEST_MONOLITHIC(x)
		TEST_SINGLE_SLICE(A)
	
		#ifdef __CUDACC__
			cublasStatus_t err;
			IF_FLOAT
				err = cublasSdgmm(handle, getCuBLASSideMode(side), C.getNumRows(), C.getNumColumns(), FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(x.getPtr()), 1, FPTR(C.getPtr()), C.getLeadingColumns());
			else IF_DOUBLE
				err = cublasDdgmm(handle, getCuBLASSideMode(side), C.getNumRows(), C.getNumColumns(), DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(x.getPtr()), 1, DPTR(C.getPtr()), C.getLeadingColumns()); 
			else IF_CX_FLOAT
				err = cublasCdgmm(handle, getCuBLASSideMode(side), C.getNumRows(), C.getNumColumns(), CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(x.getPtr()), 1, CPTR(C.getPtr()), C.getLeadingColumns());
			else IF_CX_DOUBLE
				err = cublasZdgmm(handle, getCuBLASSideMode(side), C.getNumRows(), C.getNumColumns(), ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(x.getPtr()), 1, ZPTR(C.getPtr()), C.getLeadingColumns());
			TEST_EXCEPTION(err)
		#else
			throw NotSupported;
		#endif
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

