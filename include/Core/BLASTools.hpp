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
	#define ALLOWED_TYPES_VERIFICATION		STATIC_ASSERT( Belongs<CuBLASKnownTypes, T>::value )
	#define TYPE_MUST_BE_COMPLEX			STATIC_ASSERT( TypeInfo<T>::isComplex )
	#define TEST_MONOLITHIC(x)			{if(!(x).isMonolithic()) throw IncompatibleLayout;}
	#define TEST_SINGLE_SLICE(x)			{if((x).getNumSlices()>1) throw IncompatibleLayout;} 
	#define TEST_PRODUCT(A, transa, B, transb, C)	{if(!isProductValid(A, transa, B, transb, C)) throw InvalidOperation;}
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
	__host__ inline BLASContext::BLASContext(void)
	 : 	handle(NULL)
	{
		cublasStatus_t err = cublasCreate(&handle);	
		if(err!=CUBLAS_STATUS_SUCCESS)
			throw static_cast<Exception>(CuBLASExceptionOffset + err);
	}

	__host__ inline BLASContext::~BLASContext(void)
	{
		cublasStatus_t err = cublasDestroy(handle);
		if(err!=CUBLAS_STATUS_SUCCESS)
			throw static_cast<Exception>(CuBLASExceptionOffset + err);
	}

	__host__ inline bool BLASContext::isProductValid(const Layout& A, cublasOperation_t transa, const Layout& B, cublasOperation_t transb, const Layout& C)
	{
		index_t aR = 0,
			aC = 0,
			bR = 0,
			bC = 0,
			cR = C.getNumRows(),
			cC = C.getNumColumns();

		if(transa==CUBLAS_OP_T || transa==CUBLAS_OP_C)
		{
			aR = A.getNumColumns();
			aC = A.getNumRows();
		}
		else
		{
			aR = A.getNumRows();
			aC = A.getNumColumns();
		}

		if(transb==CUBLAS_OP_T || transb==CUBLAS_OP_C)
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
		return BLASContext::isProductValid(A, CUBLAS_OP_N, B, CUBLAS_OP_N, C);
	}

	__host__ inline Layout BLASContext::getProductLayout(const Layout& A, cublasOperation_t transa, const Layout& B, cublasOperation_t transb)
	{
		index_t aR = 0,
			aC = 0,
			bR = 0,
			bC = 0;

		if(A.getNumSlices()!=1 || B.getNumSlices()!=1)
			throw InvalidOperation;

		if(transa==CUBLAS_OP_T || transa==CUBLAS_OP_C)
		{
			aR = A.getNumColumns();
			aC = A.getNumRows();
		}
		else
		{
			aR = A.getNumRows();
			aC = A.getNumColumns();
		}

		if(transb==CUBLAS_OP_T || transb==CUBLAS_OP_C)
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
		return getProductLayout(A, CUBLAS_OP_N, B, CUBLAS_OP_N);
	}

	template<typename T>
	__host__ int BLASContext::amax(const Accessor<T>& x)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_MONOLITHIC(x)
		int res;
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
		return res;
	}

	template<typename T>
	__host__ int BLASContext::amin(const Accessor<T>& x)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_MONOLITHIC(x)
		int res;
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
		return res;
	}

	template<typename T>
	__host__ typename TypeInfo<T>::BaseType BLASContext::asum(const Accessor<T>& x)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_MONOLITHIC(x)
		typename TypeInfo<T>::BaseType res;
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
		return res;
	}

	template<typename T>
	__host__ T BLASContext::dot(const Accessor<T>& x, const Accessor<T>& y, bool conjugate)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_MONOLITHIC(x)
		TEST_MONOLITHIC(y)
		T res;
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
		return res;
	}

	template<typename T>
	__host__ typename TypeInfo<T>::BaseType BLASContext::nrm2(const Accessor<T>& x)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_MONOLITHIC(x)
		typename TypeInfo<T>::BaseType res;
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
		return res;
	}

	template<typename T, typename TAlpha, typename TBeta>
	__host__ void BLASContext::gemv(const TAlpha& alpha, const Accessor<T>& A, cublasOperation_t trans, const Accessor<T>& x, const TBeta& beta, const Accessor<T>& y)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_MONOLITHIC(x)
		TEST_MONOLITHIC(y)
		TEST_SINGLE_SLICE(A)
		TEST_PRODUCT(A, trans, x, CUBLAS_OP_N, y)
		cublasStatus_t err;
		IF_FLOAT
		{
			FCST(alpha)
			FCST(beta)
			err = cublasSgemv(handle, trans, A.getNumRows(), A.getNumColumns(), &_alpha, FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(x.getPtr()), 1, &_beta, FPTR(y.getPtr()), 1);
		}
		else IF_DOUBLE
		{
			DCST(alpha)
			DCST(beta)
			err = cublasDgemv(handle, trans, A.getNumRows(), A.getNumColumns(), &_alpha, DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(x.getPtr()), 1, &_beta, DPTR(y.getPtr()), 1);
		}		
		else IF_CX_FLOAT
		{
			CCST(alpha)
			CCST(beta)
			err = cublasCgemv(handle, trans, A.getNumRows(), A.getNumColumns(), &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(x.getPtr()), 1, &_beta, CPTR(y.getPtr()), 1);
		}
		else IF_CX_DOUBLE
		{
			ZCST(alpha)
			ZCST(beta)
			err = cublasZgemv(handle, trans, A.getNumRows(), A.getNumColumns(), &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(x.getPtr()), 1, &_beta, ZPTR(y.getPtr()), 1);
		}
		TEST_EXCEPTION(err)
	}

		template<typename T>
		__host__ void BLASContext::gemv(const Accessor<T>& A, cublasOperation_t trans, const Accessor<T>& x, const Accessor<T>& y)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			gemv(alpha, A, trans, x, beta, y);
		}

		template<typename T>
		__host__ void BLASContext::gemv(const Accessor<T>& A, const Accessor<T>& x, const Accessor<T>& y)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			gemv(alpha, A, CUBLAS_OP_N, x, beta, y);
		}

	template<typename T, typename TAlpha>
	void BLASContext::ger(const TAlpha& alpha, const Accessor<T>& x, const Accessor<T>& y, const Accessor<T>& A, bool conjugate)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_MONOLITHIC(x)
		TEST_MONOLITHIC(y)
		TEST_SINGLE_SLICE(A)
		TEST_PRODUCT(x, CUBLAS_OP_N, y, CUBLAS_OP_T, A)
		cublasStatus_t err;
		IF_FLOAT
		{
			FCST(alpha)
			err = cublasSger(handle, A.getNumRows(), A.getNumColumns(), &_alpha, FPTR(x.getPtr()), 1, FPTR(y.getPtr()), 1, FPTR(A.getPtr()), A.LeadingColumns);
		}
		else IF_DOUBLE
		{
			DCST(alpha)
			err = cublasDger(handle, A.getNumRows(), A.getNumColumns(), &_alpha, DPTR(x.getPtr()), 1, DPTR(y.getPtr()), 1, DPTR(A.getPtr()), A.LeadingColumns);
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
	}

		template<typename T>
		__host__ void ger(const Accessor<T>& x, const Accessor<T>& y, const Accessor<T>& A, bool conjugate)
		{
			const T alpha = complexCopy<T>(1);
			ger(alpha, x, y, A, conjugate);
		}

	template<typename T, typename TAlpha, typename TBeta>
	__host__ void BLASContext::symv(const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x, const TBeta& beta, const Accessor<T>& y)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_MONOLITHIC(x)
		TEST_MONOLITHIC(y)
		TEST_SINGLE_SLICE(A)
		TEST_PRODUCT(A, CUBLAS_OP_N, x, CUBLAS_OP_N, y)
		cublasStatus_t err;
		IF_FLOAT
		{
			FCST(alpha)
			FCST(beta)
			err = cublasSsymv(handle, uplo, A.getNumRows(), &_alpha, FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(x.getPtr()), 1, &_beta, FPTR(y.getPtr()), 1);
		}
		else IF_DOUBLE
		{
			DCST(alpha)
			DCST(beta)
			err = cublasDsymv(handle, uplo, A.getNumRows(), &_alpha, DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(x.getPtr()), 1, &_beta, DPTR(y.getPtr()), 1);
		}
		else IF_CX_FLOAT
		{
			CCST(alpha)
			CCST(beta)
			err = cublasCsymv(handle, uplo, A.getNumRows(), &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(x.getPtr()), 1, &_beta, CPTR(y.getPtr()), 1);
		}
		else IF_CX_DOUBLE
		{
			ZCST(alpha)
			ZCST(beta)
			err = cublasZsymv(handle, uplo, A.getNumRows(), &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(x.getPtr()), 1, &_beta, ZPTR(y.getPtr()), 1);
		}
		TEST_EXCEPTION(err)
	}

		template<typename T>
		__host__ void symv(cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x, const Accessor<T>& y)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			symv(alpha, uplo, A, x, beta, y);
		}
	
	template<typename T, typename TAlpha>
	__host__ void BLASContext::syr(const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_MONOLITHIC(x)
		TEST_SINGLE_SLICE(A)
		TEST_PRODUCT(x, CUBLAS_OP_N, x, CUBLAS_OP_T, A)
		cublasStatus_t err;
		IF_FLOAT
		{
			FCST(alpha)
			err = cublasSsyr(handle, uplo, A.getNumRows(), &_alpha, FPTR(x.getPtr()), 1, FPTR(A.gePtr()), A.getLeadingColumns());
		}
		else IF_DOUBLE
		{
			DCST(alpha)
			err = cublasDsyr(handle, uplo, A.getNumRows(), &_alpha, DPTR(x.getPtr()), 1, DPTR(A.gePtr()), A.getLeadingColumns());
		}
		else IF_CX_FLOAT
		{
			CCST(alpha)
			err = cublasCsyr(handle, uplo, A.getNumRows(), &_alpha, CPTR(x.getPtr()), 1, CPTR(A.gePtr()), A.getLeadingColumns());
		}
		else IF_CX_DOUBLE
		{
			ZCST(alpha)
			err = cublasZsyr(handle, uplo, A.getNumRows(), &_alpha, ZPTR(x.getPtr()), 1, ZPTR(A.gePtr()), A.getLeadingColumns());
		}
		TEST_EXCEPTION(err)
	}

		template<typename T>
		__host__ void syr(cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x)
		{
			const T alpha = complexCopy<T>(1);
			syr(alpha, uplo, A, x);
		}

	template<typename T, typename TAlpha>
	__host__ void BLASContext::syr2(const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x, const Accessor<T>& y)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_MONOLITHIC(x)
		TEST_MONOLITHIC(y)
		TEST_SINGLE_SLICE(A)
		TEST_PRODUCT(x, CUBLAS_OP_N, y, CUBLAS_OP_T, A)
		TEST_PRODUCT(y, CUBLAS_OP_N, x, CUBLAS_OP_T, A)
		cublasStatus_t err;
		IF_FLOAT
		{
			FCST(alpha)
			err = cublasSsyr2(handle, uplo, A.getNumRows(), &_alpha, FPTR(x.getPtr()), 1, FPTR(y.getPtr()), 1, FPTR(A.gePtr()), A.getLeadingColumns());
		}
		else IF_DOUBLE
		{
			FCST(alpha)
			err = cublasSsyr2(handle, uplo, A.getNumRows(), &_alpha, DPTR(x.getPtr()), 1, DPTR(y.getPtr()), 1, DPTR(A.gePtr()), A.getLeadingColumns());
		}
		else IF_CX_FLOAT
		{
			FCST(alpha)
			err = cublasSsyr2(handle, uplo, A.getNumRows(), &_alpha, CPTR(x.getPtr()), 1, CPTR(y.getPtr()), 1, CPTR(A.gePtr()), A.getLeadingColumns());
		}
		else IF_CX_DOUBLE
		{
			FCST(alpha)
			err = cublasSsyr2(handle, uplo, A.getNumRows(), &_alpha, ZPTR(x.getPtr()), 1, ZPTR(y.getPtr()), 1, ZPTR(A.gePtr()), A.getLeadingColumns());
		}
		TEST_EXCEPTION(err)
	}

		template<typename T>
		__host__ void syr2(cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x, const Accessor<T>& y)
		{
			const T alpha = complexCopy<T>(1);
			syr2(alpha, uplo, A, x, y);
		}

	template<typename T>
	__host__ void BLASContext::trmv(cublasFillMode_t uplo, cublasDiagType_t diag, const Accessor<T>& A, cublasOperation_t trans, const Accessor<T>& x)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_MONOLITHIC(x)
		TEST_SINGLE_SLICE(A)
		TEST_PRODUCT(A, trans, x, CUBLAS_OP_N, x)
		cublasStatus_t err;
		IF_FLOAT
			err = cublasStrmv(handle, uplo, trans, diag, A.getNumRows(), FPTR(A.gePtr()), A.getLeadingColumns(), FPTR(x.getPtr()), 1);
		else IF_DOUBLE
			err = cublasStrmv(handle, uplo, trans, diag, A.getNumRows(), DPTR(A.gePtr()), A.getLeadingColumns(), DPTR(x.getPtr()), 1);
		else IF_CX_FLOAT
			err = cublasStrmv(handle, uplo, trans, diag, A.getNumRows(), CPTR(A.gePtr()), A.getLeadingColumns(), CPTR(x.getPtr()), 1);
		else IF_CX_DOUBLE
			err = cublasStrmv(handle, uplo, trans, diag, A.getNumRows(), ZPTR(A.gePtr()), A.getLeadingColumns(), ZPTR(x.getPtr()), 1);
		TEST_EXCEPTION(err)
	}

		

	template<typename T>
	__host__ void BLASContext::trsv(cublasFillMode_t uplo, cublasDiagType_t diag, const Accessor<T>& A, cublasOperation_t trans, const Accessor<T>& x)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_MONOLITHIC(x)
		TEST_SINGLE_SLICE(A)
		TEST_PRODUCT(A, trans, x, CUBLAS_OP_N, x)
		cublasStatus_t err;
		IF_FLOAT
			err = cublasStrsv(handle, uplo, trans, diag, A.getNumRows(), FPTR(A.gePtr()), A.getLeadingColumns(), FPTR(x.getPtr()), 1);
		else IF_DOUBLE
			err = cublasStrsv(handle, uplo, trans, diag, A.getNumRows(), DPTR(A.gePtr()), A.getLeadingColumns(), DPTR(x.getPtr()), 1);
		else IF_CX_FLOAT
			err = cublasStrsv(handle, uplo, trans, diag, A.getNumRows(), CPTR(A.gePtr()), A.getLeadingColumns(), CPTR(x.getPtr()), 1);
		else IF_CX_DOUBLE
			err = cublasStrsv(handle, uplo, trans, diag, A.getNumRows(), ZPTR(A.gePtr()), A.getLeadingColumns(), ZPTR(x.getPtr()), 1);
		TEST_EXCEPTION(err)
	}

	template<typename T, typename TAlpha, typename TBeta>
	__host__ void BLASContext::hemv(const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x, const TBeta& beta, const Accessor<T>& y)
	{
		ALLOWED_TYPES_VERIFICATION
		TYPE_MUST_BE_COMPLEX
		TEST_MONOLITHIC(x)
		TEST_SINGLE_SLICE(A)
		TEST_PRODUCT(A, CUBLAS_OP_N, x, CUBLAS_OP_N, y)
		cublasStatus_t err;
		IF_CX_FLOAT
		{
			CCST(alpha)
			CCST(beta)
			err = cublasChemv(handle, uplo, A.getNumRows(), &_alpha, CPTR(A.gePtr()), A.getLeadingColumns(), CPTR(x.getPtr()), 1, &_beta, CPTR(y.getPtr()), 1);
		}
		else IF_CX_DOUBLE
		{
			ZCST(alpha)
			ZCST(beta)
			err = cublasZhemv(handle, uplo, A.getNumRows(), &_alpha, ZPTR(A.gePtr()), A.getLeadingColumns(), ZPTR(x.getPtr()), 1, &_beta, ZPTR(y.getPtr()), 1);
		}
		TEST_EXCEPTION(err)
	}

		template<typename T>
		__host__ void BLASContext::hemv(cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x, const Accessor<T>& y)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			hemv(alpha, uplo, A, x, beta, y);
		}

	template<typename T, typename TAlpha>
	__host__ void BLASContext::her(const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x)
	{
		ALLOWED_TYPES_VERIFICATION
		TYPE_MUST_BE_COMPLEX
		TEST_MONOLITHIC(x)
		TEST_SINGLE_SLICE(A)
		TEST_PRODUCT(x, CUBLAS_OP_N, x, CUBLAS_OP_C, A)
		cublasStatus_t err;
		IF_CX_FLOAT
		{
			CCST(alpha)
			err = cublasCher(handle, uplo, &_alpha, CPTR(x.getPtr()), 1, CPTR(A.gePtr()), A.getLeadingColumns());
		}                
		else IF_CX_DOUBLE
		{
			ZCST(alpha)
			err = cublasZher(handle, uplo, &_alpha, ZPTR(x.getPtr()), 1, ZPTR(A.gePtr()), A.getLeadingColumns());
		}
		TEST_EXCEPTION(err)
	}

		template<typename T>
		__host__ void her(cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x)
		{
			const T alpha = complexCopy<T>(1);
			her(alpha, uplo, A, x);
		}

	template<typename T, typename TAlpha>
	__host__ void BLASContext::her2(const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x, const Accessor<T>& y)
	{
		ALLOWED_TYPES_VERIFICATION
		TYPE_MUST_BE_COMPLEX
		TEST_MONOLITHIC(x)
		TEST_SINGLE_SLICE(A)
		TEST_PRODUCT(x, CUBLAS_OP_N, y, CUBLAS_OP_C, A)
		TEST_PRODUCT(y, CUBLAS_OP_N, x, CUBLAS_OP_C, A)
		cublasStatus_t err;
		IF_CX_FLOAT
		{
			CCST(alpha)
			err = cublasCher2(handle, uplo, A.getNumRows(), &_alpha, CPTR(x.getPtr()), 1, CPTR(y.getPtr()), 1, CPTR(A.gePtr()), A.getLeadingColumns());
		}
		else IF_CX_DOUBLE
		{
			ZCST(alpha)
			err = cublasCher2(handle, uplo, A.getNumRows(), &_alpha, ZPTR(x.getPtr()), 1, ZPTR(y.getPtr()), 1, CPTR(A.gePtr()), A.getLeadingColumns());
		}
		TEST_EXCEPTION(err)
	}

		template<typename T>
		__host__ void her2(cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x, const Accessor<T>& y)
		{
			const T alpha = complexCopy<T>(1);
			her2(alpha, uplo, A, x, y);
		}

	template<typename T, typename TAlpha, typename TBeta>
	__host__ void BLASContext::gemm(const TAlpha& alpha, const Accessor<T>& A, cublasOperation_t transa, const Accessor<T>& B, cublasOperation_t transb, const TBeta& beta, const Accessor<T>& C)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(B)
		TEST_SINGLE_SLICE(C)
		TEST_PRODUCT(A, transa, B, transb, C)
		cublasStatus_t err;
		const int k = (transa==CUBLAS_OP_T || transa==CUBLAS_OP_C) ? A.getNumRows() : A.getNumColumns();
		IF_FLOAT
		{
			FCST(alpha)
			FCST(beta)
			err = cublasSgemm(handle, transa, transb, C.getNumRows(), C.getNumColumns(), k, &_alpha, FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(B.getPtr()), B.getLeadingColumns(), &_beta, FPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_DOUBLE
		{
			DCST(alpha)
			DCST(beta)
			err = cublasDgemm(handle, transa, transb, C.getNumRows(), C.getNumColumns(), k, &_alpha, DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(B.getPtr()), B.getLeadingColumns(), &_beta, DPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_CX_FLOAT
		{
			CCST(alpha)
			CCST(beta)
			err = cublasCgemm(handle, transa, transb, C.getNumRows(), C.getNumColumns(), k, &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(B.getPtr()), B.getLeadingColumns(), &_beta, CPTR(C.getPtr()), C.getLeadingColumns());
		}		
		else IF_CX_DOUBLE
		{
			ZCST(alpha)
			ZCST(beta)
			err = cublasZgemm(handle, transa, transb, C.getNumRows(), C.getNumColumns(), k, &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(B.getPtr()), B.getLeadingColumns(), &_beta, ZPTR(C.getPtr()), C.getLeadingColumns());
		}	
		TEST_EXCEPTION(err)
	}

		template<typename T>
		__host__ void BLASContext::gemm(const Accessor<T>& A, cublasOperation_t transa, const Accessor<T>& B, cublasOperation_t transb, const Accessor<T>& C)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			gemm(alpha, A, transa, B, transb, beta, C);
		}

		template<typename T>
		__host__ void BLASContext::gemm(const Accessor<T>& A, const Accessor<T>& B, const Accessor<T>& C)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			gemm(alpha, A, CUBLAS_OP_N, B, CUBLAS_OP_N, beta, C);
		}

	template<typename T, typename TAlpha, typename TBeta>
	__host__ void BLASContext::symm(cublasSideMode_t side, const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& B, const TBeta& beta, const Accessor<T>& C)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(B)
		TEST_SINGLE_SLICE(C)
		if(side==CUBLAS_SIDE_LEFT)
			TEST_PRODUCT(A, CUBLAS_OP_N, B, CUBLAS_OP_N, C)
		else // CUBLAS_SIDE_RIGHT
			TEST_PRODUCT(B, CUBLAS_OP_N, A, CUBLAS_OP_N, C)
		cublasStatus_t err;
		IF_FLOAT
		{
			FCST(alpha)
			FCST(beta)
			err = cublasSsymm(handle, side, uplo, B.getNumRows(), C.getNumColumns(), &_alpha, FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(B.getPtr()), B.getLeadingColumns(), &_beta, FPTR(C.getPtr()), C.getLeadingColumns());
		}
                else IF_DOUBLE
		{
			DCST(alpha)
			DCST(beta)
			err = cublasDsymm(handle, side, uplo, B.getNumRows(), C.getNumColumns(), &_alpha, DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(B.getPtr()), B.getLeadingColumns(), &_beta, DPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_CX_FLOAT
		{
			CCST(alpha)
			CCST(beta)
			err = cublasCsymm(handle, side, uplo, B.getNumRows(), C.getNumColumns(), &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(B.getPtr()), B.getLeadingColumns(), &_beta, CPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_CX_DOUBLE
		{
			ZCST(alpha)
			ZCST(beta)
			err = cublasZsymm(handle, side, uplo, B.getNumRows(), C.getNumColumns(), &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(B.getPtr()), B.getLeadingColumns(), &_beta, ZPTR(C.getPtr()), C.getLeadingColumns());
		}
		TEST_EXCEPTION(err)
	}

		template<typename T>
		__host__ void symm(cublasSideMode_t side, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& B, const Accessor<T>& C)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			symm(side, alpha, uplo, A, B, beta, C);
		}

	template<typename T, typename TAlpha, typename TBeta>
	__host__ void BLASContext::syrk(const TAlpha& alpha, const Accessor<T>& A, cublasOperation_t transa, const TBeta& beta, cublasFillMode_t uplo, const Accessor<T>& C)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(C)
		if(transa==CUBLAS_OP_N)
			TEST_PRODUCT(A, CUBLAS_OP_N, A, CUBLAS_OP_T, C)
		else
			TEST_PRODUCT(A, CUBLAS_OP_T, A, CUBLAS_OP_N, C)
		cublasStatus_t err;
		const int k = (transa==CUBLAS_OP_T || transa==CUBLAS_OP_C) ? A.getNumRows() : A.getNumColumns();
		IF_FLOAT
		{
			FCST(alpha)
			FCST(beta)
			err = cublasSsyrk(handle, uplo, transa, C.getNumRows(), k, &_alpha, FPTR(A.getPtr()), A.getLeadingColumns(), &_beta, FPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_DOUBLE
		{
			DCST(alpha)
			DCST(beta)
			err = cublasDsyrk(handle, uplo, transa, C.getNumRows(), k, &_alpha, DPTR(A.getPtr()), A.getLeadingColumns(), &_beta, DPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_CX_FLOAT
		{
			CCST(alpha)
			CCST(beta)
			err = cublasCsyrk(handle, uplo, transa, C.getNumRows(), k, &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), &_beta, CPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_CX_DOUBLE
		{
			ZCST(alpha)
			ZCST(beta)
			err = cublasZsyrk(handle, uplo, transa, C.getNumRows(), k, &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), &_beta, ZPTR(C.getPtr()), C.getLeadingColumns());
		}
		TEST_EXCEPTION(err)
	}

		template<typename T>
		__host__ void BLASContext::syrk(const Accessor<T>& A, cublasOperation_t transa, cublasFillMode_t uplo, const Accessor<T>& C)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			syrk(alpha, A, transa, beta, uplo, C);
		}

		template<typename T>
		__host__ void BLASContext::syrk(const Accessor<T>& A, cublasFillMode_t uplo, const Accessor<T>& C)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			syrk(alpha, A, CUBLAS_OP_N, beta, uplo, C);
		}

	template<typename T, typename TAlpha>
	__host__ void BLASContext::syr2k(cublasOperation_t trans, const TAlpha& alpha, const Accessor<T>& A, const Accessor<T>& B, const T& beta, cublasFillMode_t uplo, const Accessor<T>& C)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(C)
		if(trans==CUBLAS_OP_N)
		{
			TEST_PRODUCT(A, CUBLAS_OP_N, B, CUBLAS_OP_T, C)
			TEST_PRODUCT(B, CUBLAS_OP_N, A, CUBLAS_OP_T, C)
		}		
		else
		{
			TEST_PRODUCT(A, CUBLAS_OP_T, B, CUBLAS_OP_N, C)
			TEST_PRODUCT(B, CUBLAS_OP_T, A, CUBLAS_OP_N, C)
		}	
		cublasStatus_t err;
		const int k = (trans==CUBLAS_OP_T || trans==CUBLAS_OP_C) ? A.getNumRows() : A.getNumColumns();
		IF_FLOAT
		{
			FCST(alpha)
			FCST(beta)
			err = cublasSsyr2k(handle, uplo, trans, C.getNumRows(), k, &_alpha, FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(B.getPtr()), B.getLeadingColumns(), &_beta, FPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_DOUBLE
		{
			DCST(alpha)
			DCST(beta)
			err = cublasDsyr2k(handle, uplo, trans, C.getNumRows(), k, &_alpha, DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(B.getPtr()), B.getLeadingColumns(), &_beta, DPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_CX_FLOAT
		{
			CCST(alpha)
			CCST(beta)
			err = cublasCsyr2k(handle, uplo, trans, C.getNumRows(), k, &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(B.getPtr()), B.getLeadingColumns(), &_beta, CPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_CX_DOUBLE
		{
			ZCST(alpha)
			ZCST(beta)
			err = cublasZsyr2k(handle, uplo, trans, C.getNumRows(), k, &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(B.getPtr()), B.getLeadingColumns(), &_beta, ZPTR(C.getPtr()), C.getLeadingColumns());
		}
		TEST_EXCEPTION(err)
	}

		template<typename T>
		__host__ void BLASContext::syr2k(cublasOperation_t trans, const Accessor<T>& A, const Accessor<T>& B, cublasFillMode_t uplo, const Accessor<T>& C)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			syr2k(trans, alpha, A, B, beta, uplo, C);
		}

		template<typename T>
		__host__ void BLASContext::syr2k(const Accessor<T>& A, const Accessor<T>& B, cublasFillMode_t uplo, const Accessor<T>& C)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			syr2k(CUBLAS_OP_N, alpha, A, B, beta, uplo, C);
		}

	template<typename T, typename TAlpha>
	__host__ void BLASContext::trmm(cublasSideMode_t side, const TAlpha& alpha, cublasFillMode_t uplo, cublasDiagType_t diag, const Accessor<T>& A, cublasOperation_t transa, const Accessor<T>& B, const Accessor<T>& C)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(B)
		TEST_SINGLE_SLICE(C)
		if(side==CUBLAS_SIDE_LEFT)
			TEST_PRODUCT(A, transa, B, CUBLAS_OP_N, C)
		else // CUBLAS_SIDE_RIGHT
			TEST_PRODUCT(B, CUBLAS_OP_N, A, transa, C)
		TEST_PRODUCT(A, CUBLAS_OP_N, B, CUBLAS_OP_T, C)
		cublasStatus_t err;
		IF_FLOAT
		{
			FCST(alpha)
			err = cublasStrmm(handle, side, uplo, transa, diag, B.getNumRows(), B.getNumColumns(), &_alpha, FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(B.getPtr()), B.getLeadingColumns(), FPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_DOUBLE
		{
			DCST(alpha)
			err = cublasDtrmm(handle, side, uplo, transa, diag, B.getNumRows(), B.getNumColumns(), &_alpha, DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(B.getPtr()), B.getLeadingColumns(), DPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_CX_FLOAT
		{
			CCST(alpha)
			err = cublasCtrmm(handle, side, uplo, transa, diag, B.getNumRows(), B.getNumColumns(), &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(B.getPtr()), B.getLeadingColumns(), CPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_CX_DOUBLE
		{
			ZCST(alpha)
			err = cublasZtrmm(handle, side, uplo, transa, diag, B.getNumRows(), B.getNumColumns(), &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(B.getPtr()), B.getLeadingColumns(), ZPTR(C.getPtr()), C.getLeadingColumns());
		}
		TEST_EXCEPTION(err)
	}

		template<typename T>
		__host__ void BLASContext::trmm(cublasSideMode_t side, cublasFillMode_t uplo, cublasDiagType_t diag, const Accessor<T>& A, cublasOperation_t transa, const Accessor<T>& B, const Accessor<T>& C)
		{
			const T alpha = complexCopy<T>(1);
			trmm(side, alpha, uplo, diag, A, transa, B, C);
		}
		
		template<typename T>
		__host__ void BLASContext::trmm(cublasSideMode_t side, cublasFillMode_t uplo, cublasDiagType_t diag, const Accessor<T>& A, const Accessor<T>& B, const Accessor<T>& C)
		{
			const T alpha = complexCopy<T>(1);
			trmm(side, alpha, uplo, diag, A, CUBLAS_OP_N, B, C);
		}
		
	template<typename T, typename TAlpha>
	__host__ void BLASContext::trsm(cublasSideMode_t side, cublasFillMode_t uplo, cublasDiagType_t diag, const Accessor<T>& A, cublasOperation_t transa, const TAlpha& alpha, const Accessor<T>& B)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(B)
		if(side==CUBLAS_SIDE_LEFT)
			TEST_PRODUCT(A, transa, B, CUBLAS_OP_N, B)
		else // CUBLAS_SIDE_RIGHT
			TEST_PRODUCT(B, CUBLAS_OP_N, A, transa, B)
		cublasStatus_t err;
		IF_FLOAT
		{
			FCST(alpha)
			err = cublasStrsm(handle, side, uplo, transa, diag, B.getNumRows(), B.getNumColumns(), &_alpha, FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(B.getPtr()), B.getLeadingColumns());
		}
		else IF_DOUBLE
		{
			DCST(alpha)
			err = cublasDtrsm(handle, side, uplo, transa, diag, B.getNumRows(), B.getNumColumns(), &_alpha, DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(B.getPtr()), B.getLeadingColumns());
		}
		else IF_CX_FLOAT
		{
			CCST(alpha)
			err = cublasCtrsm(handle, side, uplo, transa, diag, B.getNumRows(), B.getNumColumns(), &_alpha, CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(B.getPtr()), B.getLeadingColumns());
		}
		else IF_CX_DOUBLE
		{
			ZCST(alpha)
			err = cublasZtrsm(handle, side, uplo, transa, diag, B.getNumRows(), B.getNumColumns(), &_alpha, ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(B.getPtr()), B.getLeadingColumns());
		}
		TEST_EXCEPTION(err)
	}

		template<typename T>
		__host__ void BLASContext::trsm(cublasSideMode_t side, cublasFillMode_t uplo, cublasDiagType_t diag, const Accessor<T>& A, cublasOperation_t transa, const Accessor<T>& B)
		{
			const T alpha = complexCopy<T>(1);
			trsm(side, uplo, diag, A, transa, alpha, B);
		}

		template<typename T>
		__host__ void BLASContext::trsm(cublasSideMode_t side, cublasFillMode_t uplo, cublasDiagType_t diag, const Accessor<T>& A, const Accessor<T>& B)
		{
			const T alpha = complexCopy<T>(1);
			trsm(side, uplo, diag, A, CUBLAS_OP_N, alpha, B);
		}

	template<typename T, typename TAlpha, typename TBeta>
	__host__ void BLASContext::hemm(cublasSideMode_t side, const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& B, const TBeta& beta, const Accessor<T>& C)
	{
		ALLOWED_TYPES_VERIFICATION
		TYPE_MUST_BE_COMPLEX
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(B)
		TEST_SINGLE_SLICE(C)
		if(side==CUBLAS_SIDE_LEFT)
			TEST_PRODUCT(A, CUBLAS_OP_N, B, CUBLAS_OP_N, C)
		else // CUBLAS_SIDE_RIGHT
			TEST_PRODUCT(B, CUBLAS_OP_N, A, CUBLAS_OP_N, C)
		cublasStatus_t err;
		IF_CX_FLOAT
		{
			CCST(alpha)
			CCST(beta)
			err = cublasChemm(handle, side, uplo, C.getNumRows(), C.getNumColumns(), CPTR(&alpha), CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(B.getPtr()), B.getLeadingColumns(), CPTR(&beta), CPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_CX_DOUBLE
		{
			ZCST(alpha)
			ZCST(beta)
			err = cublasZhemm(handle, side, uplo, C.getNumRows(), C.getNumColumns(), ZPTR(&alpha), ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(B.getPtr()), B.getLeadingColumns(), ZPTR(&beta), ZPTR(C.getPtr()), C.getLeadingColumns());
		}
		TEST_EXCEPTION(err)
	}

		template<typename T>
		__host__ void BLASContext::hemm(cublasSideMode_t side, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& B, const Accessor<T>& C)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			hemm(side, alpha, uplo, A, B, beta, C);
		}

	template<typename T, typename TAlpha, typename TBeta>
	__host__ void BLASContext::herk(const TAlpha& alpha, const Accessor<T>& A, cublasOperation_t transa, const TBeta& beta, cublasFillMode_t uplo, const Accessor<T>& C)
	{
		ALLOWED_TYPES_VERIFICATION
		TYPE_MUST_BE_COMPLEX
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(C)
		if(transa==CUBLAS_OP_N)
			TEST_PRODUCT(A, CUBLAS_OP_N, A, CUBLAS_OP_C, C)
		else
			TEST_PRODUCT(A, CUBLAS_OP_C, A, CUBLAS_OP_N, C)
		cublasStatus_t err;
		int k = (transa==CUBLAS_OP_T || transa==CUBLAS_OP_C) ? A.getNumRows() : A.getNumColumns();
		IF_CX_FLOAT
		{
			CCST(alpha)
			CCST(beta)
			err = cublasCherk(handle, uplo, transa, C.getNumRows(), k, CPTR(&alpha), CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(&beta), CPTR(C.getPtr()), C.getLeadingColumns());
		}
                else IF_CX_DOUBLE
		{
			ZCST(alpha)
			ZCST(beta)
			err = cublasZherk(handle, uplo, transa, C.getNumRows(), k, ZPTR(&alpha), ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(&beta), ZPTR(C.getPtr()), C.getLeadingColumns());
		}

		TEST_EXCEPTION(err)
	}

		template<typename T>
		__host__ void BLASContext::herk(const Accessor<T>& A, cublasOperation_t transa, cublasFillMode_t uplo, const Accessor<T>& C)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			herk(alpha, A, transa, beta, uplo, C);
		}

		template<typename T>
		__host__ void BLASContext::herk(const Accessor<T>& A, cublasFillMode_t uplo, const Accessor<T>& C)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			herk(alpha, A, CUBLAS_OP_N, beta, uplo, C);
		}		

	template<typename T, typename TAlpha, typename TBeta>
	__host__ void BLASContext::her2k(cublasOperation_t trans, const TAlpha& alpha, const Accessor<T>& A, const Accessor<T>& B, const TBeta& beta, cublasFillMode_t uplo, const Accessor<T>& C)
	{
		ALLOWED_TYPES_VERIFICATION
		TYPE_MUST_BE_COMPLEX
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(B)
		TEST_SINGLE_SLICE(C)
		if(trans==CUBLAS_OP_N)
		{
			TEST_PRODUCT(A, CUBLAS_OP_N, B, CUBLAS_OP_C, C)
			TEST_PRODUCT(B, CUBLAS_OP_N, A, CUBLAS_OP_C, C)
		}		
		else
		{
			TEST_PRODUCT(A, CUBLAS_OP_C, B, CUBLAS_OP_N, C)
			TEST_PRODUCT(B, CUBLAS_OP_C, A, CUBLAS_OP_N, C)
		}	
		cublasStatus_t err;
		const int k = (trans==CUBLAS_OP_T || trans==CUBLAS_OP_C) ? A.getNumRows() : A.getNumColumns();
		IF_CX_FLOAT
		{
			CCST(alpha)
			CCST(beta)
			err = cublasCher2k(handle, uplo, trans, C.getNumRows(), k, CPTR(&alpha), CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(B.getPtr()), B.getLeadingColumns(), CPTR(&beta), CPTR(C.getPtr()), C.getLeadingColumns());
		}
		IF_CX_DOUBLE
		{
			ZCST(alpha)
			ZCST(beta)
			err = cublasZher2k(handle, uplo, trans, C.getNumRows(), k, ZPTR(&alpha), ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(B.getPtr()), B.getLeadingColumns(), ZPTR(&beta), ZPTR(C.getPtr()), C.getLeadingColumns());
		}
		TEST_EXCEPTION(err)
	}

		template<typename T>
		__host__ void BLASContext::her2k(cublasOperation_t trans, const Accessor<T>& A, const Accessor<T>& B, cublasFillMode_t uplo, const Accessor<T>& C)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			her2k(trans, alpha, A, B, beta, uplo, C);
		}

		template<typename T>
		__host__ void BLASContext::her2k(const Accessor<T>& A, const Accessor<T>& B, cublasFillMode_t uplo, const Accessor<T>& C)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			her2k(CUBLAS_OP_N, alpha, A, B, beta, uplo, C);
		}

	template<typename T, typename TAlpha, typename TBeta>
	__host__ void BLASContext::geam(const TAlpha& alpha, const Accessor<T>& A, cublasOperation_t transa, const TBeta& beta, const Accessor<T>& B, cublasOperation_t transb, const Accessor<T>& C)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_SINGLE_SLICE(A)
		TEST_SINGLE_SLICE(B)
		TEST_SINGLE_SLICE(C)
		cublasStatus_t err;
		IF_FLOAT
		{
			FCST(alpha)
			FCST(beta)
			err = cublasSgeam(handle, transa, transb, C.getNumRows(), C.getNumColumns(), FPTR(&alpha), FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(&beta), FPTR(B.getPtr()), B.getLeadingColumns(), FPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_DOUBLE
		{
			DCST(alpha)
			DCST(beta)
			err = cublasDgeam(handle, transa, transb, C.getNumRows(), C.getNumColumns(), DPTR(&alpha), DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(&beta), DPTR(B.getPtr()), B.getLeadingColumns(), DPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_CX_FLOAT
		{
			CCST(alpha)
			CCST(beta)
			err = cublasCgeam(handle, transa, transb, C.getNumRows(), C.getNumColumns(), CPTR(&alpha), CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(&beta), CPTR(B.getPtr()), B.getLeadingColumns(), CPTR(C.getPtr()), C.getLeadingColumns());
		}
		else IF_CX_DOUBLE
		{
			ZCST(alpha)
			ZCST(beta)
			err = cublasZgeam(handle, transa, transb, C.getNumRows(), C.getNumColumns(), ZPTR(&alpha), ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(&beta), ZPTR(B.getPtr()), B.getLeadingColumns(), ZPTR(C.getPtr()), C.getLeadingColumns());
		}
		TEST_EXCEPTION(err)
	}

		template<typename T>
		__host__ void BLASContext::geam(const Accessor<T>& A, cublasOperation_t transa, const Accessor<T>& B, cublasOperation_t transb, const Accessor<T>& C)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			geam(alpha, A, transa, beta, B, transb, C);
		}

		template<typename T>
		__host__ void BLASContext::geam(const Accessor<T>& A, const Accessor<T>& B, const Accessor<T>& C)
		{
			const T alpha = complexCopy<T>(1),
				beta = complexCopy<T>(0);
			geam(alpha, A, CUBLAS_OP_N, beta, B, CUBLAS_OP_N, C);
		}

	template<typename T>
	__host__ void BLASContext::dgmm(cublasSideMode_t mode, const Accessor<T>& A, const Accessor<T>& x, const Accessor<T>& C)
	{
		ALLOWED_TYPES_VERIFICATION
		TEST_MONOLITHIC(x)
		TEST_SINGLE_SLICE(A)
		cublasStatus_t err;
		IF_FLOAT
			err = cublasSdgmm(handle, mode, C.getNumRows(), C.getNumColumns(), FPTR(A.getPtr()), A.getLeadingColumns(), FPTR(x.getPtr()), 1, FPTR(C.getPtr()), C.getLeadingColumns());
		else IF_DOUBLE
			err = cublasDdgmm(handle, mode, C.getNumRows(), C.getNumColumns(), DPTR(A.getPtr()), A.getLeadingColumns(), DPTR(x.getPtr()), 1, DPTR(C.getPtr()), C.getLeadingColumns()); 
		else IF_CX_FLOAT
			err = cublasCdgmm(handle, mode, C.getNumRows(), C.getNumColumns(), CPTR(A.getPtr()), A.getLeadingColumns(), CPTR(x.getPtr()), 1, CPTR(C.getPtr()), C.getLeadingColumns());
		else IF_CX_DOUBLE
			err = cublasZdgmm(handle, mode, C.getNumRows(), C.getNumColumns(), ZPTR(A.getPtr()), A.getLeadingColumns(), ZPTR(x.getPtr()), 1, ZPTR(C.getPtr()), C.getLeadingColumns());
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

