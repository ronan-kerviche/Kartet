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
// Lists :
	typedef TypeList< float,
		TypeList< double,
		TypeList< cuFloatComplex,
		TypeList< cuDoubleComplex,
		Void
		> > > > CuBLASKnownTypes;

// Type tools :
	#define ALLOWED_TYPES_VERIFICATION	STATIC_ASSERT( Belongs<CuBLASKnownTypes, T>::Value )
	#define TYPE_MUST_BE_COMPLEX		STATIC_ASSERT( TypeInfo<T>::isComplex )
	#define TEST_CONTEXT			if(BLAS::singleton==NULL) throw InvalidBLASContext;
	#define TEST_MONOLITHIC(x)		if(!(x).isMonolithic()) throw IncompatibleLayout;
	#define TEST_SINGLE_SLICE(x)		if((x).getNumSlices()>1) throw IncompatibleLayout; 
	#define IF_FLOAT			if(SameTypes<T, float>::test)
	#define IF_DOUBLE			if(SameTypes<T, double>::test)
	#define IF_CX_FLOAT			if(SameTypes<T, cuFloatComplex>::test)
	#define IF_CX_DOUBLE			if(SameTypes<T, cuDoubleComplex>::test)
	#define HDL				(BLAS::singleton->handle)
	#define FCST(x)				const float _##x = static_cast<float>(x);
	#define DCST(x)				const double _##x = static_cast<double>(x);
	#define CCST(x)				const cuFloatComplex _##x = toFloatComplex(x);
	#define ZCST(x)				const cuDoubleComplex _##x = toDoubleComplex(x);
	#define FPTR(x)				reinterpret_cast<float*>(x)
	#define DPTR(x)				reinterpret_cast<double*>(x)
	#define CPTR(x)				reinterpret_cast<cuFloatComplex*>(x)
	#define ZPTR(x)				reinterpret_cast<cuDoubleComplex*>(x)
	#define TEST_EXCEPTION(x)		if(x!=CUBLAS_STATUS_SUCCESS) throw static_cast<Exception>(CuBLASExceptionOffset + x);

// BLAS :
	__host__ inline BLAS::BLAS(void)
	 : 	handle(NULL)
	{
		if(singleton==NULL)
		{
			cublasStatus_t err = cublasCreate(&handle);	
			if(err!=CUBLAS_STATUS_SUCCESS)
				throw static_cast<Exception>(CuBLASExceptionOffset + err);
			singleton = this;
		}
	}

	__host__ inline BLAS::~BLAS(void)
	{
		if(singleton==this)
		{
			singleton = NULL;
			cublasStatus_t err = cublasDestroy(handle);
			if(err!=CUBLAS_STATUS_SUCCESS)
				throw static_cast<Exception>(CuBLASExceptionOffset + err);
		}
	}

	template<typename T>
	__host__ int amax(const Accessor<T>& x)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_CONTEXT
		TEST_MONOLITHIC(x)
		int res;
		cublasStatus_t err;
		IF_FLOAT
			err = cublasIsamax(HDL, x.getNumElements(), FPTR(x.getPtr()), 1, &res);
		else IF_DOUBLE
			err = cublasIdamax(HDL, x.getNumElements(), DPTR(x.getPtr()), 1, &res);
		else IF_CX_FLOAT
			err = cublasIcamax(HDL, x.getNumElements(), CPTR(x.getPtr()), 1, &res);
		else IF_CX_DOUBLE
			err = cublasIzamax(HDL, x.getNumElements(), ZPTR(x.getPtr()), 1, &res);
		TEST_EXCEPTION(err)
		return res;
	}

	template<typename T>
	__host__ int amin(const Accessor<T>& x)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_CONTEXT
		TEST_MONOLITHIC(x)
		int res;
		cublasStatus_t err;
		IF_FLOAT
			err = cublasIsamin(HDL, x.getNumElements(), FPTR(x.getPtr()), 1, &res);
		else IF_DOUBLE
			err = cublasIdamin(HDL, x.getNumElements(), DPTR(x.getPtr()), 1, &res);
		else IF_CX_FLOAT
			err = cublasIcamin(HDL, x.getNumElements(), CPTR(x.getPtr()), 1, &res);
		else IF_CX_DOUBLE
			err = cublasIzamin(HDL, x.getNumElements(), ZPTR(x.getPtr()), 1, &res);
		TEST_EXCEPTION(err)
		return res;
	}

	template<typename T>
	__host__ typename TypeInfo<T>::BaseType asum(const Accessor<T>& x)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_CONTEXT
		TEST_MONOLITHIC(x)
		typename TypeInfo<T>::BaseType res;
		cublasStatus_t err;
		IF_FLOAT
			err = cublasSasum(HDL, x.getNumElements(), FPTR(x.getPtr()), 1, FPTR(&res));
		else IF_DOUBLE
			err = cublasDasum(HDL, x.getNumElements(), DPTR(x.getPtr()), 1, DPTR(&res));
		else IF_CX_FLOAT
			err = cublasScasum(HDL, x.getNumElements(), CPTR(x.getPtr()), 1, FPTR(&res));
		else IF_CX_DOUBLE
			err = cublasDzasum(HDL, x.getNumElements(), ZPTR(x.getPtr()), 1, DPTR(&res));
		TEST_EXCEPTION(err)
		return res;
	}

	template<typename T>
	__host__ T dot(const Accessor<T>& x, const Accessor<T>& y, bool conjugate)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_CONTEXT
		TEST_MONOLITHIC(x)
		TEST_MONOLITHIC(y)
		T res;
		cublasStatus_t err;
		IF_FLOAT
			err = cublasSdot(HDL, x.getNumElements(), FPTR(x.getPtr()), 1, FPTR(y.getPtr()), 1, FPTR(&res));
		else IF_DOUBLE
			err = cublasDdot(HDL, x.getNumElements(), DPTR(x.getPtr()), 1, DPTR(y.getPtr()), 1, DPTR(&res));
		else IF_CX_FLOAT
		{
			if(!conjugate)
				err = cublasCdotu(HDL, x.getNumElements(), CPTR(x.getPtr()), 1, CPTR(y.getPtr()), 1, CPTR(&res));
			else
				err = cublasCdotc(HDL, x.getNumElements(), CPTR(x.getPtr()), 1, CPTR(y.getPtr()), 1, CPTR(&res));
		}
		else IF_CX_DOUBLE
		{
			if(!conjugate)
				err = cublasZdotu(HDL, x.getNumElements(), ZPTR(x.getPtr()), 1, ZPTR(y.getPtr()), 1, ZPTR(&res));
			else
				err = cublasZdotc(HDL, x.getNumElements(), ZPTR(x.getPtr()), 1, ZPTR(y.getPtr()), 1, ZPTR(&res));
		}
		TEST_EXCEPTION(err)
		return res;
	}

	template<typename T>
	__host__ typename TypeInfo<T>::BaseType nrm2(const Accessor<T>& x)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_CONTEXT
		TEST_MONOLITHIC(x)
		typename TypeInfo<T>::BaseType res;
		cublasStatus_t err;
		IF_FLOAT
			err = cublasSnrm2(HDL, x.getNumElements(), FPTR(x.getPtr()), 1, FPTR(&res));
		else IF_DOUBLE
			err = cublasDnrm2(HDL, x.getNumElements(), DPTR(x.getPtr()), 1, DPTR(&res));
		else IF_CX_FLOAT
			err = cublasScnrm2(HDL, x.getNumElements(), CPTR(x.getPtr()), 1, FPTR(&res));
		else IF_CX_DOUBLE
			err = cublasDznrm2(HDL, x.getNumElements(), ZPTR(x.getPtr()), 1, DPTR(&res));
		TEST_EXCEPTION(err)
		return res;
	}

	template<typename T, typename TAlpha, typename TBeta>
	__host__ void gemv(const TAlpha& alpha, const Accessor<T>& A, cublasOperation_t trans, const Accessor<T>& x, const TBeta& beta, const Accessor<T>& y)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_CONTEXT
		TEST_MONOLITHIC(x)
		TEST_MONOLITHIC(y)
		TEST_SINGLE_SLICE(A)
		cublasStatus_t err;
		IF_FLOAT
		{
			FCST(alpha)
			FCST(beta)
			err = cublasSgemv(HDL, trans, A.getNumRows(), A.getNumColumns(), &_alpha, FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(x.getPtr()), 1, &_beta, FPTR(y.getPtr()), 1);
		}
		else IF_DOUBLE
		{
			DCST(alpha)
			DCST(beta)
			err = cublasDgemv(HDL, trans, A.getNumRows(), A.getNumColumns(), &_alpha, DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(x.getPtr()), 1, &_beta, DPTR(y.getPtr()), 1);
		}		
		else IF_CX_FLOAT
		{
			CCST(alpha)
			CCST(beta)
			err = cublasCgemv(HDL, trans, A.getNumRows(), A.getNumColumns(), &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(x.getPtr()), 1, &_beta, CPTR(y.getPtr()), 1);
		}
		else IF_CX_DOUBLE
		{
			ZCST(alpha)
			ZCST(beta)
			err = cublasZgemv(HDL, trans, A.getNumRows(), A.getNumColumns(), &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(x.getPtr()), 1, &_beta, ZPTR(y.getPtr()), 1);
		}
		TEST_EXCEPTION(err)
	}

	template<typename T, typename TAlpha>
	void ger(const TAlpha& alpha, const Accessor<T>& x, const Accessor<T>& y, const Accessor<T>& A, bool conjugate)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_CONTEXT
		TEST_MONOLITHIC(x)
		TEST_MONOLITHIC(y)
		TEST_SINGLE_SLICE(A)
		cublasStatus_t err;
		IF_FLOAT
		{
			FCST(alpha)
			err = cublasSger(HDL, A.getNumRows(), A.getNumColumns(), &_alpha, FPTR(x.getPtr()), 1, FPTR(y.getPtr()), 1, FPTR(A.getPtr()), A.LeadingColumns);
		}
		else IF_DOUBLE
		{
			DCST(alpha)
			err = cublasDger(HDL, A.getNumRows(), A.getNumColumns(), &_alpha, DPTR(x.getPtr()), 1, DPTR(y.getPtr()), 1, DPTR(A.getPtr()), A.LeadingColumns);
		}
		else IF_CX_FLOAT
		{
			CCST(alpha)
			if(!conjugate)
				err = cublasCgeru(HDL, A.getNumRows(), A.getNumColumns(), &_alpha, CPTR(x.getPtr()), 1, CPTR(y.getPtr()), 1, CPTR(A.getPtr()), A.getLeadingColumns());
			else
				err = cublasCgerc(HDL, A.getNumRows(), A.getNumColumns(), &_alpha, CPTR(x.getPtr()), 1, CPTR(y.getPtr()), 1, CPTR(A.getPtr()), A.getLeadingColumns());
		}
		else IF_CX_DOUBLE
		{
			ZCST(alpha)
			if(!conjugate)
				err = cublasZgeru(HDL, A.getNumRows(), A.getNumColumns(), &_alpha, ZPTR(x.getPtr()), 1, ZPTR(y.getPtr()), 1, ZPTR(A.getPtr()), A.getLeadingColumns());
			else
				err = cublasZgerc(HDL, A.getNumRows(), A.getNumColumns(), &_alpha, ZPTR(x.getPtr()), 1, ZPTR(y.getPtr()), 1, ZPTR(A.getPtr()), A.getLeadingColumns());
		}
		TEST_EXCEPTION(err)
	}

	template<typename T, typename TAlpha, typename TBeta>
	__host__ void symv(const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x, const TBeta& beta, const Accessor<T>& y)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_CONTEXT
		TEST_MONOLITHIC(x)
		TEST_MONOLITHIC(y)
		TEST_SINGLE_SLICE(A)
		cublasStatus_t err;
		IF_FLOAT
		{
			FCST(alpha)
			FCST(beta)
			err = cublasSsymv(HDL, uplo, A.getNumRows(), &_alpha, FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(x.getPtr()), 1, &_beta, FPTR(y.getPtr()), 1);
		}
		else IF_DOUBLE
		{
			DCST(alpha)
			DCST(beta)
			err = cublasDsymv(HDL, uplo, A.getNumRows(), &_alpha, DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(x.getPtr()), 1, &_beta, DPTR(y.getPtr()), 1);
		}
		else IF_CX_FLOAT
		{
			CCST(alpha)
			CCST(beta)
			err = cublasCsymv(HDL, uplo, A.getNumRows(), &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(x.getPtr()), 1, &_beta, CPTR(y.getPtr()), 1);
		}
		else IF_CX_DOUBLE
		{
			ZCST(alpha)
			ZCST(beta)
			err = cublasZsymv(HDL, uplo, A.getNumRows(), &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(x.getPtr()), 1, &_beta, ZPTR(y.getPtr()), 1);
		}
		TEST_EXCEPTION(err)
	}
	
	template<typename T, typename TAlpha>
	__host__ void syr(const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_CONTEXT
		TEST_MONOLITHIC(x)
		TEST_SINGLE_SLICE(A)
		cublasStatus_t err;
		IF_FLOAT
		{
			FCST(alpha)
			err = cublasSsyr(HDL, uplo, A.getNumRows(), &_alpha, FPTR(x.getPtr()), 1, FPTR(A.gePtr()), A.getLeadingColumns());
		}
		else IF_DOUBLE
		{
			DCST(alpha)
			err = cublasDsyr(HDL, uplo, A.getNumRows(), &_alpha, DPTR(x.getPtr()), 1, DPTR(A.gePtr()), A.getLeadingColumns());
		}
		else IF_CX_FLOAT
		{
			CCST(alpha)
			err = cublasCsyr(HDL, uplo, A.getNumRows(), &_alpha, CPTR(x.getPtr()), 1, CPTR(A.gePtr()), A.getLeadingColumns());
		}
		else IF_CX_DOUBLE
		{
			ZCST(alpha)
			err = cublasZsyr(HDL, uplo, A.getNumRows(), &_alpha, ZPTR(x.getPtr()), 1, ZPTR(A.gePtr()), A.getLeadingColumns());
		}
		TEST_EXCEPTION(err)
	}

	template<typename T, typename TAlpha>
	__host__ void syr2(const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x, const Accessor<T>& y)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_CONTEXT
		TEST_MONOLITHIC(x)
		TEST_MONOLITHIC(y)
		TEST_SINGLE_SLICE(A)
		cublasStatus_t err;
		IF_FLOAT
		{
			FCST(alpha)
			err = cublasSsyr2(HDL, uplo, A.getNumRows(), &_alpha, FPTR(x.getPtr()), 1, FPTR(y.getPtr()), 1, FPTR(A.gePtr()), A.getLeadingColumns());
		}
		else IF_DOUBLE
		{
			FCST(alpha)
			err = cublasSsyr2(HDL, uplo, A.getNumRows(), &_alpha, DPTR(x.getPtr()), 1, DPTR(y.getPtr()), 1, DPTR(A.gePtr()), A.getLeadingColumns());
		}
		else IF_CX_FLOAT
		{
			FCST(alpha)
			err = cublasSsyr2(HDL, uplo, A.getNumRows(), &_alpha, CPTR(x.getPtr()), 1, CPTR(y.getPtr()), 1, CPTR(A.gePtr()), A.getLeadingColumns());
		}
		else IF_CX_DOUBLE
		{
			FCST(alpha)
			err = cublasSsyr2(HDL, uplo, A.getNumRows(), &_alpha, ZPTR(x.getPtr()), 1, ZPTR(y.getPtr()), 1, ZPTR(A.gePtr()), A.getLeadingColumns());
		}
		TEST_EXCEPTION(err)
	}

	template<typename T>
	__host__ void trmv(cublasFillMode_t uplo, cublasDiagType_t diag, const Accessor<T>& A, cublasOperation_t trans, const Accessor<T>& x)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_CONTEXT
		TEST_MONOLITHIC(x)
		TEST_SINGLE_SLICE(A)
		cublasStatus_t err;
		IF_FLOAT
			err = cublasStrmv(HDL, uplo, trans, diag, A.getNumRows(), FPTR(A.gePtr()), A.getLeadingColumns(), FPTR(x.getPtr()), 1);
		else IF_DOUBLE
			err = cublasStrmv(HDL, uplo, trans, diag, A.getNumRows(), DPTR(A.gePtr()), A.getLeadingColumns(), DPTR(x.getPtr()), 1);
		else IF_CX_FLOAT
			err = cublasStrmv(HDL, uplo, trans, diag, A.getNumRows(), CPTR(A.gePtr()), A.getLeadingColumns(), CPTR(x.getPtr()), 1);
		else IF_CX_DOUBLE
			err = cublasStrmv(HDL, uplo, trans, diag, A.getNumRows(), ZPTR(A.gePtr()), A.getLeadingColumns(), ZPTR(x.getPtr()), 1);
		TEST_EXCEPTION(err)
	}

	template<typename T>
	__host__ void trsv(cublasFillMode_t uplo, cublasDiagType_t diag, const Accessor<T>& A, cublasOperation_t trans, const Accessor<T>& x)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_CONTEXT
		TEST_MONOLITHIC(x)
		TEST_SINGLE_SLICE(A)
		cublasStatus_t err;
		IF_FLOAT
			err = cublasStrsv(HDL, uplo, trans, diag, A.getNumRows(), FPTR(A.gePtr()), A.getLeadingColumns(), FPTR(x.getPtr()), 1);
		else IF_DOUBLE
			err = cublasStrsv(HDL, uplo, trans, diag, A.getNumRows(), DPTR(A.gePtr()), A.getLeadingColumns(), DPTR(x.getPtr()), 1);
		else IF_CX_FLOAT
			err = cublasStrsv(HDL, uplo, trans, diag, A.getNumRows(), CPTR(A.gePtr()), A.getLeadingColumns(), CPTR(x.getPtr()), 1);
		else IF_CX_DOUBLE
			err = cublasStrsv(HDL, uplo, trans, diag, A.getNumRows(), ZPTR(A.gePtr()), A.getLeadingColumns(), ZPTR(x.getPtr()), 1);
		TEST_EXCEPTION(err)
	}

	template<typename T, typename TAlpha, typename TBeta>
	__host__ void hemv(const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x, const TBeta& beta, const Accessor<T>& y)
	{
		ALLOWED_TYPES_VERIFICATION
		TYPE_MUST_BE_COMPLEX
		TEST_CONTEXT
		TEST_MONOLITHIC(x)
		TEST_SINGLE_SLICE(A)
		cublasStatus_t err;
		IF_CX_FLOAT
		{
			CCST(alpha)
			CCST(beta)
			err = cublasChemv(HDL, uplo, A.getNumRows(), &_alpha, CPTR(A.gePtr()), A.getLeadingColumns(), CPTR(x.getPtr()), 1, &_beta, CPTR(y.getPtr()), 1);
		}
		else IF_CX_DOUBLE
		{
			ZCST(alpha)
			ZCST(beta)
			err = cublasZhemv(HDL, uplo, A.getNumRows(), &_alpha, ZPTR(A.gePtr()), A.getLeadingColumns(), ZPTR(x.getPtr()), 1, &_beta, ZPTR(y.getPtr()), 1);
		}
		TEST_EXCEPTION(err)
	}

	template<typename T, typename TAlpha>
	__host__ void her(const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x)
	{
		ALLOWED_TYPES_VERIFICATION
		TYPE_MUST_BE_COMPLEX
		TEST_CONTEXT
		TEST_MONOLITHIC(x)
		TEST_SINGLE_SLICE(A)
		cublasStatus_t err;
		IF_CX_FLOAT
		{
			CCST(alpha)
			err = cublasCher(HDL, uplo, &_alpha, CPTR(x.getPtr()), 1, CPTR(A.gePtr()), A.getLeadingColumns());
		}                
		else IF_CX_DOUBLE
		{
			ZCST(alpha)
			err = cublasZher(HDL, uplo, &_alpha, ZPTR(x.getPtr()), 1, ZPTR(A.gePtr()), A.getLeadingColumns());
		}
		TEST_EXCEPTION(err)
	}

	template<typename T, typename TAlpha>
	__host__ void her2(const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x, const Accessor<T>& y)
	{
		ALLOWED_TYPES_VERIFICATION
		TYPE_MUST_BE_COMPLEX
		TEST_CONTEXT
		TEST_MONOLITHIC(x)
		TEST_SINGLE_SLICE(A)
		cublasStatus_t err;
		IF_CX_FLOAT
		{
			CCST(alpha)
			err = cublasCher2(HDL, uplo, A.getNumRows(), &_alpha, CPTR(x.getPtr()), 1, CPTR(y.getPtr()), 1, CPTR(A.gePtr()), A.getLeadingColumns());
		}
		else IF_CX_DOUBLE
		{
			ZCST(alpha)
			err = cublasCher2(HDL, uplo, A.getNumRows(), &_alpha, ZPTR(x.getPtr()), 1, ZPTR(y.getPtr()), 1, CPTR(A.gePtr()), A.getLeadingColumns());
		}
		TEST_EXCEPTION(err)
	}

	template<typename T, typename TAlpha, typename TBeta>
	__host__ void gemm(const TAlpha& alpha, const Accessor<T>& A, cublasOperation_t transa, const Accessor<T>& B, cublasOperation_t transb, const TBeta& beta, const Accessor<T>& C)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_CONTEXT
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(B)
		TEST_SINGLE_SLICE(C)
		cublasStatus_t err;
		const int k = (transa==CUBLAS_OP_T || transa==CUBLAS_OP_C) ? A.getNumRows() : A.getNumColumns();
		IF_FLOAT
		{
			FCST(alpha)
			FCST(beta)
			err = cublasSgemm(HDL, transa, transb, C.getNumRows(), C.getNumColumns(), k, &_alpha, FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(B.getPtr()), B.getLeadingColumns(), &_beta, FPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_DOUBLE
		{
			DCST(alpha)
			DCST(beta)
			err = cublasDgemm(HDL, transa, transb, C.getNumRows(), C.getNumColumns(), k, &_alpha, DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(B.getPtr()), B.getLeadingColumns(), &_beta, DPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_CX_FLOAT
		{
			CCST(alpha)
			CCST(beta)
			err = cublasCgemm(HDL, transa, transb, C.getNumRows(), C.getNumColumns(), k, &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(B.getPtr()), B.getLeadingColumns(), &_beta, CPTR(C.getPtr()), C.getLeadingColumns());
		}		
		else IF_CX_DOUBLE
		{
			ZCST(alpha)
			ZCST(beta)
			err = cublasZgemm(HDL, transa, transb, C.getNumRows(), C.getNumColumns(), k, &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(B.getPtr()), B.getLeadingColumns(), &_beta, ZPTR(C.getPtr()), C.getLeadingColumns());
		}	
		TEST_EXCEPTION(err)
	}

	template<typename T, typename TAlpha, typename TBeta>
	__host__ void symm(cublasSideMode_t side, const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& B, const TBeta& beta, const Accessor<T>& C)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_CONTEXT
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(B)
		TEST_SINGLE_SLICE(C)
		cublasStatus_t err;
		IF_FLOAT
		{
			FCST(alpha)
			FCST(beta)
			err = cublasSsymm(HDL, side, uplo, B.getNumRows(), C.getNumColumns(), &_alpha, FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(B.getPtr()), B.getLeadingColumns(), &_beta, FPTR(C.getPtr()), C.getLeadingColumns());
		}
                else IF_DOUBLE
		{
			DCST(alpha)
			DCST(beta)
			err = cublasDsymm(HDL, side, uplo, B.getNumRows(), C.getNumColumns(), &_alpha, DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(B.getPtr()), B.getLeadingColumns(), &_beta, DPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_CX_FLOAT
		{
			CCST(alpha)
			CCST(beta)
			err = cublasCsymm(HDL, side, uplo, B.getNumRows(), C.getNumColumns(), &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(B.getPtr()), B.getLeadingColumns(), &_beta, CPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_CX_DOUBLE
		{
			ZCST(alpha)
			ZCST(beta)
			err = cublasZsymm(HDL, side, uplo, B.getNumRows(), C.getNumColumns(), &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(B.getPtr()), B.getLeadingColumns(), &_beta, ZPTR(C.getPtr()), C.getLeadingColumns());
		}
		TEST_EXCEPTION(err)
	}

	template<typename T, typename TAlpha, typename TBeta>
	__host__ void syrk(const TAlpha& alpha, const Accessor<T>& A, cublasOperation_t transa, const TBeta& beta, cublasFillMode_t uplo, const Accessor<T>& C)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_CONTEXT
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(C)
		cublasStatus_t err;
		const int k = (transa==CUBLAS_OP_T || transa==CUBLAS_OP_C) ? A.getNumRows() : A.getNumColumns();
		IF_FLOAT
		{
			FCST(alpha)
			FCST(beta)
			err = cublasSsyrk(HDL, uplo, transa, C.getNumRows(), k, &_alpha, FPTR(A.getPtr()), A.getLeadingColumns(), &_beta, FPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_DOUBLE
		{
			DCST(alpha)
			DCST(beta)
			err = cublasDsyrk(HDL, uplo, transa, C.getNumRows(), k, &_alpha, DPTR(A.getPtr()), A.getLeadingColumns(), &_beta, DPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_CX_FLOAT
		{
			CCST(alpha)
			CCST(beta)
			err = cublasCsyrk(HDL, uplo, transa, C.getNumRows(), k, &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), &_beta, CPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_CX_DOUBLE
		{
			ZCST(alpha)
			ZCST(beta)
			err = cublasZsyrk(HDL, uplo, transa, C.getNumRows(), k, &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), &_beta, ZPTR(C.getPtr()), C.getLeadingColumns());
		}
		TEST_EXCEPTION(err)
	}

	template<typename T, typename TAlpha>
	__host__ void syr2k(cublasOperation_t trans, const TAlpha& alpha, const Accessor<T>& A, const Accessor<T>& B, const T& beta, cublasFillMode_t uplo, const Accessor<T>& C)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_CONTEXT
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(C)
		cublasStatus_t err;
		const int k = (trans==CUBLAS_OP_T || trans==CUBLAS_OP_C) ? A.getNumRows() : A.getNumColumns();
		IF_FLOAT
		{
			FCST(alpha)
			FCST(beta)
			err = cublasSsyr2k(HDL, uplo, trans, C.getNumRows(), k, &_alpha, FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(B.getPtr()), B.getLeadingColumns(), &_beta, FPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_DOUBLE
		{
			DCST(alpha)
			DCST(beta)
			err = cublasDsyr2k(HDL, uplo, trans, C.getNumRows(), k, &_alpha, DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(B.getPtr()), B.getLeadingColumns(), &_beta, DPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_CX_FLOAT
		{
			CCST(alpha)
			CCST(beta)
			err = cublasCsyr2k(HDL, uplo, trans, C.getNumRows(), k, &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(B.getPtr()), B.getLeadingColumns(), &_beta, CPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_CX_DOUBLE
		{
			ZCST(alpha)
			ZCST(beta)
			err = cublasZsyr2k(HDL, uplo, trans, C.getNumRows(), k, &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(B.getPtr()), B.getLeadingColumns(), &_beta, ZPTR(C.getPtr()), C.getLeadingColumns());
		}
		TEST_EXCEPTION(err)
	}

	template<typename T, typename TAlpha>
	__host__ void trmm(cublasSideMode_t side, const TAlpha& alpha, cublasFillMode_t uplo, cublasDiagType_t diag, const Accessor<T>& A, cublasOperation_t transa, const Accessor<T>& B, const Accessor<T>& C)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_CONTEXT
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(B)
		TEST_SINGLE_SLICE(C)
		cublasStatus_t err;
		IF_FLOAT
		{
			FCST(alpha)
			err = cublasStrmm(HDL, side, uplo, transa, diag, B.getNumRows(), B.getNumColumns(), &_alpha, FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(B.getPtr()), B.getLeadingColumns(), FPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_DOUBLE
		{
			DCST(alpha)
			err = cublasDtrmm(HDL, side, uplo, transa, diag, B.getNumRows(), B.getNumColumns(), &_alpha, DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(B.getPtr()), B.getLeadingColumns(), DPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_CX_FLOAT
		{
			CCST(alpha)
			err = cublasCtrmm(HDL, side, uplo, transa, diag, B.getNumRows(), B.getNumColumns(), &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(B.getPtr()), B.getLeadingColumns(), CPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_CX_DOUBLE
		{
			ZCST(alpha)
			err = cublasZtrmm(HDL, side, uplo, transa, diag, B.getNumRows(), B.getNumColumns(), &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(B.getPtr()), B.getLeadingColumns(), ZPTR(C.getPtr()), C.getLeadingColumns());
		}
		TEST_EXCEPTION(err)
	}
		
	template<typename T, typename TAlpha>
	__host__ void trsm(cublasSideMode_t side, cublasFillMode_t uplo, cublasDiagType_t diag, const Accessor<T>& A, cublasOperation_t transa, const TAlpha& alpha, const Accessor<T>& B)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_CONTEXT
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(B)
		cublasStatus_t err;
		IF_FLOAT
		{
			FCST(alpha)
			err = cublasStrsm(HDL, side, uplo, transa, diag, B.getNumRows(), B.getNumColumns(), &_alpha, FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(B.getPtr()), B.getLeadingColumns());
		}
		else IF_DOUBLE
		{
			DCST(alpha)
			err = cublasDtrsm(HDL, side, uplo, transa, diag, B.getNumRows(), B.getNumColumns(), &_alpha, DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(B.getPtr()), B.getLeadingColumns());
		}
		else IF_CX_FLOAT
		{
			CCST(alpha)
			err = cublasCtrsm(HDL, side, uplo, transa, diag, B.getNumRows(), B.getNumColumns(), &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(B.getPtr()), B.getLeadingColumns());
		}
		else IF_CX_DOUBLE
		{
			ZCST(alpha)
			err = cublasZtrsm(HDL, side, uplo, transa, diag, B.getNumRows(), B.getNumColumns(), &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(B.getPtr()), B.getLeadingColumns());
		}
		TEST_EXCEPTION(err)
	}

	template<typename T, typename TAlpha, typename TBeta>
	__host__ void hemm(cublasSideMode_t side, const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& B, const TBeta& beta, const Accessor<T>& C)
	{
		ALLOWED_TYPES_VERIFICATION
		TYPE_MUST_BE_COMPLEX
		TEST_CONTEXT
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(B)
		TEST_SINGLE_SLICE(C)
		cublasStatus_t err;
		IF_CX_FLOAT
		{
			CCST(alpha)
			CCST(beta)
			err = cublasChemm(HDL, side, uplo, C.getNumRows(), C.getNumColumns(), CPTR(&alpha), CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(B.getPtr()), B.getLeadingColumns(), CPTR(&beta), CPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_CX_DOUBLE
		{
			ZCST(alpha)
			ZCST(beta)
			err = cublasZhemm(HDL, side, uplo, C.getNumRows(), C.getNumColumns(), ZPTR(&alpha), ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(B.getPtr()), B.getLeadingColumns(), ZPTR(&beta), ZPTR(C.getPtr()), C.getLeadingColumns());
		}
		TEST_EXCEPTION(err)
	}

	template<typename T, typename TAlpha, typename TBeta>
	__host__ void herk(const TAlpha& alpha, const Accessor<T>& A, cublasOperation_t transa, const TBeta& beta, cublasFillMode_t uplo, const Accessor<T>& C)
	{
		ALLOWED_TYPES_VERIFICATION
		TYPE_MUST_BE_COMPLEX
		TEST_CONTEXT
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(C)
		cublasStatus_t err;
		int k = (transa==CUBLAS_OP_T || transa==CUBLAS_OP_C) ? A.getNumRows() : A.getNumColumns();
		IF_CX_FLOAT
		{
			CCST(alpha)
			CCST(beta)
			err = cublasCherk(HDL, uplo, transa, C.getNumRows(), k, CPTR(&alpha), CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(&beta), CPTR(C.getPtr()), C.getLeadingColumns());
		}
                else IF_CX_DOUBLE
		{
			ZCST(alpha)
			ZCST(beta)
			err = cublasZherk(HDL, uplo, transa, C.getNumRows(), k, ZPTR(&alpha), ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(&beta), ZPTR(C.getPtr()), C.getLeadingColumns());
		}

		TEST_EXCEPTION(err)
	}
		
	template<typename T, typename TAlpha, typename TBeta>
	__host__ void her2k(cublasOperation_t trans, const T& alpha, const Accessor<T>& A, const Accessor<T>& B, const T& beta, cublasFillMode_t uplo, const Accessor<T>& C)
	{
		ALLOWED_TYPES_VERIFICATION
		TYPE_MUST_BE_COMPLEX
		TEST_CONTEXT
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(B)
		TEST_SINGLE_SLICE(C)
		cublasStatus_t err;
		const int k = (trans==CUBLAS_OP_T || trans==CUBLAS_OP_C) ? A.getNumRows() : A.getNumColumns();
		IF_CX_FLOAT
		{
			CCST(alpha)
			CCST(beta)
			err = cublasCher2k(HDL, uplo, trans, C.getNumRows(), k, CPTR(&alpha), CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(B.getPtr()), B.getLeadingColumns(), CPTR(&beta), CPTR(C.getPtr()), C.getLeadingColumns());
		}
		IF_CX_DOUBLE
		{
			ZCST(alpha)
			ZCST(beta)
			err = cublasZher2k(HDL, uplo, trans, C.getNumRows(), k, ZPTR(&alpha), ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(B.getPtr()), B.getLeadingColumns(), ZPTR(&beta), ZPTR(C.getPtr()), C.getLeadingColumns());
		}
		TEST_EXCEPTION(err)
	}

	template<typename T, typename TAlpha, typename TBeta>
	__host__ void geam(const TAlpha& alpha, const Accessor<T>& A, cublasOperation_t transa, const TBeta& beta, const Accessor<T>& B, cublasOperation_t transb, const Accessor<T>& C)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_CONTEXT
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(B)
		TEST_SINGLE_SLICE(C)
		cublasStatus_t err;
		IF_FLOAT
		{
			FCST(alpha)
			FCST(beta)
			err = cublasSgeam(HDL, transa, transb, C.getNumRows(), C.getNumColumns(), FPTR(&alpha), FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(&beta), FPTR(B.getPtr()), B.getLeadingColumns(), FPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_DOUBLE
		{
			DCST(alpha)
			DCST(beta)
			err = cublasDgeam(HDL, transa, transb, C.getNumRows(), C.getNumColumns(), DPTR(&alpha), DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(&beta), DPTR(B.getPtr()), B.getLeadingColumns(), DPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_CX_FLOAT
		{
			CCST(alpha)
			CCST(beta)
			err = cublasCgeam(HDL, transa, transb, C.getNumRows(), C.getNumColumns(), CPTR(&alpha), CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(&beta), CPTR(B.getPtr()), B.getLeadingColumns(), CPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_CX_DOUBLE
		{
			ZCST(alpha)
			ZCST(beta)
			err = cublasZgeam(HDL, transa, transb, C.getNumRows(), C.getNumColumns(), ZPTR(&alpha), ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(&beta), ZPTR(B.getPtr()), B.getLeadingColumns(), ZPTR(C.getPtr()), C.getLeadingColumns());
		}
		TEST_EXCEPTION(err)
	}

	template<typename T>
	__host__ void dgmm(cublasSideMode_t mode, const Accessor<T>& A, const Accessor<T>& x, const Accessor<T>& C)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_CONTEXT
		TEST_MONOLITHIC(x)
		TEST_SINGLE_SLICE(A)
		cublasStatus_t err;
		IF_FLOAT
			err = cublasSdgmm(HDL, mode, C.getNumRows(), C.getNumColumns(), FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(x.getPtr()), 1, FPTR(C.getPtr()), C.getLeadingColumns());
		else IF_DOUBLE
			err = cublasDdgmm(HDL, mode, C.getNumRows(), C.getNumColumns(), DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(x.getPtr()), 1, DPTR(C.getPtr()), C.getLeadingColumns()); 
		else IF_CX_FLOAT
			err = cublasCdgmm(HDL, mode, C.getNumRows(), C.getNumColumns(), CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(x.getPtr()), 1, CPTR(C.getPtr()), C.getLeadingColumns());
		else IF_CX_DOUBLE
			err = cublasZdgmm(HDL, mode, C.getNumRows(), C.getNumColumns(), ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(x.getPtr()), 1, ZPTR(C.getPtr()), C.getLeadingColumns());
		TEST_EXCEPTION(err)
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
	#undef HDL
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

