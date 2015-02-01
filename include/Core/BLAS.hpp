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

#ifndef __KARTET_BLAS__
#define __KARTET_BLAS__

	#include <cublas_v2.h>
	#include "Core/Array.hpp"

namespace Kartet
{
// BLAS Handle :
	class BLAS
	{
		private :
			static BLAS* singleton;
			cublasHandle_t 	handle;	

		// Friends :
			template<typename T>
			__host__ friend int amax(const Accessor<T>& x);

			template<typename T>
			__host__ friend int amin(const Accessor<T>& x);

			template<typename T>
			__host__ friend typename TypeInfo<T>::BaseType asum(const Accessor<T>& x);

			template<typename T>
			__host__ friend T dot(const Accessor<T>& x, const Accessor<T>& y, bool conjugate = true);

			template<typename T>
			__host__ friend typename TypeInfo<T>::BaseType nrm2(const Accessor<T>& a);

			template<typename T, typename TAlpha, typename TBeta>
			__host__ friend void gemv(const TAlpha& alpha, const Accessor<T>& A, cublasOperation_t trans, const Accessor<T>& x, const TBeta& beta, const Accessor<T>& y);

			template<typename T, typename TAlpha>
			__host__ friend void ger(const TAlpha& alpha, const Accessor<T>& x, const Accessor<T>& y, const Accessor<T>& A, bool conjugate = true);

			template<typename T, typename TAlpha, typename TBeta>
			__host__ friend void symv(const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x, const TBeta& beta, const Accessor<T>& y);
			
			template<typename T, typename TAlpha>
			__host__ friend void syr(const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x);

			template<typename T, typename TAlpha>
			__host__ friend void syr2(const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x, const Accessor<T>& y);

			template<typename T>
			__host__ friend void trmv(cublasFillMode_t uplo, cublasDiagType_t diag, const Accessor<T>& A, cublasOperation_t trans, const Accessor<T>& x);

			template<typename T>
			__host__ friend void trsv(cublasFillMode_t uplo, cublasDiagType_t diag, const Accessor<T>& A, cublasOperation_t trans, const Accessor<T>& x);

			template<typename T, typename TAlpha, typename TBeta>
			__host__ friend void hemv(const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x, const TBeta& beta, const Accessor<T>& y);

			template<typename T, typename TAlpha>
			__host__ friend void her(const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x);

			template<typename T, typename TAlpha>
			__host__ friend void her2(const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x, const Accessor<T>& y);

			template<typename T, typename TAlpha, typename TBeta>
			__host__ friend void gemm(const TAlpha& alpha, const Accessor<T>& A, cublasOperation_t transa, const Accessor<T>& B, cublasOperation_t transb, const TBeta& beta, const Accessor<T>& C);

			template<typename T, typename TAlpha, typename TBeta>
			__host__ friend void symm(cublasSideMode_t side, const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& B, const TBeta& beta, const Accessor<T>& C);

			template<typename T, typename TAlpha, typename TBeta>
			__host__ friend void syrk(const TAlpha& alpha, const Accessor<T>& A, cublasOperation_t transa, const TBeta& beta, cublasFillMode_t uplo, const Accessor<T>& C);

			template<typename T, typename TAlpha>
			__host__ friend void syr2k(cublasOperation_t trans, const TAlpha& alpha, const Accessor<T>& A, const Accessor<T>& B, const T& beta, cublasFillMode_t uplo, const Accessor<T>& C);

			template<typename T, typename TAlpha>
			__host__ friend void trmm(cublasSideMode_t side, const TAlpha& alpha, cublasFillMode_t uplo, cublasDiagType_t diag, const Accessor<T>& A, cublasOperation_t transa, const Accessor<T>& B, const Accessor<T>& C);
		
			template<typename T, typename TAlpha>
			__host__ friend void trsm(cublasSideMode_t side, cublasFillMode_t uplo, cublasDiagType_t diag, const Accessor<T>& A, cublasOperation_t transa, const TAlpha& alpha, const Accessor<T>& B);

			template<typename T, typename TAlpha, typename TBeta>
			__host__ friend void hemm(cublasSideMode_t side, const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& B, const TBeta& beta, const Accessor<T>& C);

			template<typename T, typename TAlpha, typename TBeta>
			__host__ friend void herk(const TAlpha& alpha, const Accessor<T>& A, cublasOperation_t transa, const TBeta& beta, cublasFillMode_t uplo, const Accessor<T>& C);
		
			template<typename T, typename TAlpha, typename TBeta>
			__host__ friend void her2k(cublasOperation_t trans, const TAlpha& alpha, const Accessor<T>& A, const Accessor<T>& B, const TBeta& beta, cublasFillMode_t uplo, const Accessor<T>& C);

			template<typename T, typename TAlpha, typename TBeta>
			__host__ friend void geam(const TAlpha& alpha, const Accessor<T>& A, cublasOperation_t transa, const TBeta& beta, const Accessor<T>& B, cublasOperation_t transb, const Accessor<T>& C);

			template<typename T>
			__host__ friend void dgmm(cublasSideMode_t mode, const Accessor<T>& A, const Accessor<T>& v, const Accessor<T>& C);

		public :
			__host__ inline BLAS(void);
			__host__ inline ~BLAS(void);
	};

	BLAS* BLAS::singleton = NULL;
} // namespace Kartet

	#include "BLASTools.hpp"

#endif

