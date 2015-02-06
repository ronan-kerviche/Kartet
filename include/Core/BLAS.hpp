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
// BLASContext :
	class BLASContext
	{
		private :
			cublasHandle_t 	handle;	

		public :
			__host__ inline BLASContext(void);
			__host__ inline ~BLASContext(void);

			__host__ static inline bool isProductValid(const Layout& A, cublasOperation_t transa, const Layout& B, cublasOperation_t transb, const Layout& C);

			template<typename T>
			__host__ int amax(const Accessor<T>& x);

			template<typename T>
			__host__ int amin(const Accessor<T>& x);

			template<typename T>
			__host__ typename TypeInfo<T>::BaseType asum(const Accessor<T>& x);

			template<typename T>
			__host__ T dot(const Accessor<T>& x, const Accessor<T>& y, bool conjugate = true);

			template<typename T>
			__host__ typename TypeInfo<T>::BaseType nrm2(const Accessor<T>& a);

			template<typename T, typename TAlpha, typename TBeta>
			__host__ void gemv(const TAlpha& alpha, const Accessor<T>& A, cublasOperation_t trans, const Accessor<T>& x, const TBeta& beta, const Accessor<T>& y);

				template<typename T>
				__host__ void gemv(const Accessor<T>& A, cublasOperation_t trans, const Accessor<T>& x, const Accessor<T>& y);

				template<typename T>
				__host__ void gemv(const Accessor<T>& A, const Accessor<T>& x, const Accessor<T>& y);

			template<typename T, typename TAlpha>
			__host__ void ger(const TAlpha& alpha, const Accessor<T>& x, const Accessor<T>& y, const Accessor<T>& A, bool conjugate = true);

				template<typename T>
				__host__ void ger(const Accessor<T>& x, const Accessor<T>& y, const Accessor<T>& A, bool conjugate = true);

			template<typename T, typename TAlpha, typename TBeta>
			__host__ void symv(const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x, const TBeta& beta, const Accessor<T>& y);

				template<typename T>
				__host__ void symv(cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x, const Accessor<T>& y);
			
			template<typename T, typename TAlpha>
			__host__ void syr(const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x);

				template<typename T>
				__host__ void syr(cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x);

			template<typename T, typename TAlpha>
			__host__ void syr2(const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x, const Accessor<T>& y);
		
				template<typename T>
				__host__ void syr2(cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x, const Accessor<T>& y);

			template<typename T>
			__host__ void trmv(cublasFillMode_t uplo, cublasDiagType_t diag, const Accessor<T>& A, cublasOperation_t trans, const Accessor<T>& x);

			template<typename T>
			__host__ void trsv(cublasFillMode_t uplo, cublasDiagType_t diag, const Accessor<T>& A, cublasOperation_t trans, const Accessor<T>& x);

			template<typename T, typename TAlpha, typename TBeta>
			__host__ void hemv(const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x, const TBeta& beta, const Accessor<T>& y);

				template<typename T>
				__host__ void hemv(cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x, const Accessor<T>& y);

			template<typename T, typename TAlpha>
			__host__ void her(const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x);

				template<typename T>
				__host__ void her(cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x);

			template<typename T, typename TAlpha>
			__host__ void her2(const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x, const Accessor<T>& y);

				template<typename T>
				__host__ void her2(cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& x, const Accessor<T>& y);

			template<typename T, typename TAlpha, typename TBeta>
			__host__ void gemm(const TAlpha& alpha, const Accessor<T>& A, cublasOperation_t transa, const Accessor<T>& B, cublasOperation_t transb, const TBeta& beta, const Accessor<T>& C);

				template<typename T>
				__host__ void gemm(const Accessor<T>& A, cublasOperation_t transa, const Accessor<T>& B, cublasOperation_t transb, const Accessor<T>& C);

				template<typename T>
				__host__ void gemm(const Accessor<T>& A, const Accessor<T>& B, const Accessor<T>& C);

			template<typename T, typename TAlpha, typename TBeta>
			__host__ void symm(cublasSideMode_t side, const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& B, const TBeta& beta, const Accessor<T>& C);

				template<typename T>
				__host__ void symm(cublasSideMode_t side, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& B, const Accessor<T>& C);

			template<typename T, typename TAlpha, typename TBeta>
			__host__ void syrk(const TAlpha& alpha, const Accessor<T>& A, cublasOperation_t transa, const TBeta& beta, cublasFillMode_t uplo, const Accessor<T>& C);

				template<typename T>
				__host__ void syrk(const Accessor<T>& A, cublasOperation_t transa, cublasFillMode_t uplo, const Accessor<T>& C);

				template<typename T>
				__host__ void syrk(const Accessor<T>& A, cublasFillMode_t uplo, const Accessor<T>& C);

			template<typename T, typename TAlpha>
			__host__ void syr2k(cublasOperation_t trans, const TAlpha& alpha, const Accessor<T>& A, const Accessor<T>& B, const T& beta, cublasFillMode_t uplo, const Accessor<T>& C);

				template<typename T>
				__host__ void syr2k(cublasOperation_t trans, const Accessor<T>& A, const Accessor<T>& B, cublasFillMode_t uplo, const Accessor<T>& C);

				template<typename T>
				__host__ void syr2k(const Accessor<T>& A, const Accessor<T>& B, cublasFillMode_t uplo, const Accessor<T>& C);

			template<typename T, typename TAlpha>
			__host__ void trmm(cublasSideMode_t side, const TAlpha& alpha, cublasFillMode_t uplo, cublasDiagType_t diag, const Accessor<T>& A, cublasOperation_t transa, const Accessor<T>& B, const Accessor<T>& C);

				template<typename T>
				__host__ void trmm(cublasSideMode_t side, cublasFillMode_t uplo, cublasDiagType_t diag, const Accessor<T>& A, cublasOperation_t transa, const Accessor<T>& B, const Accessor<T>& C);
		
				template<typename T>
				__host__ void trmm(cublasSideMode_t side, cublasFillMode_t uplo, cublasDiagType_t diag, const Accessor<T>& A, const Accessor<T>& B, const Accessor<T>& C);

			template<typename T, typename TAlpha>
			__host__ void trsm(cublasSideMode_t side, cublasFillMode_t uplo, cublasDiagType_t diag, const Accessor<T>& A, cublasOperation_t transa, const TAlpha& alpha, const Accessor<T>& B);

				template<typename T>
				__host__ void trsm(cublasSideMode_t side, cublasFillMode_t uplo, cublasDiagType_t diag, const Accessor<T>& A, cublasOperation_t transa, const Accessor<T>& B);

				template<typename T>
				__host__ void trsm(cublasSideMode_t side, cublasFillMode_t uplo, cublasDiagType_t diag, const Accessor<T>& A, const Accessor<T>& B);

			template<typename T, typename TAlpha, typename TBeta>
			__host__ void hemm(cublasSideMode_t side, const TAlpha& alpha, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& B, const TBeta& beta, const Accessor<T>& C);

				template<typename T>
				__host__ void hemm(cublasSideMode_t side, cublasFillMode_t uplo, const Accessor<T>& A, const Accessor<T>& B, const Accessor<T>& C);

			template<typename T, typename TAlpha, typename TBeta>
			__host__ void herk(const TAlpha& alpha, const Accessor<T>& A, cublasOperation_t transa, const TBeta& beta, cublasFillMode_t uplo, const Accessor<T>& C);
		
				template<typename T>
				__host__ void herk(const Accessor<T>& A, cublasOperation_t transa, cublasFillMode_t uplo, const Accessor<T>& C);

				template<typename T>
				__host__ void herk(const Accessor<T>& A, cublasFillMode_t uplo, const Accessor<T>& C);

			template<typename T, typename TAlpha, typename TBeta>
			__host__ void her2k(cublasOperation_t trans, const TAlpha& alpha, const Accessor<T>& A, const Accessor<T>& B, const TBeta& beta, cublasFillMode_t uplo, const Accessor<T>& C);

				template<typename T>
				__host__ void her2k(cublasOperation_t trans, const Accessor<T>& A, const Accessor<T>& B, cublasFillMode_t uplo, const Accessor<T>& C);

				template<typename T>
				__host__ void her2k(const Accessor<T>& A, const Accessor<T>& B, cublasFillMode_t uplo, const Accessor<T>& C);

			template<typename T, typename TAlpha, typename TBeta>
			__host__ void geam(const TAlpha& alpha, const Accessor<T>& A, cublasOperation_t transa, const TBeta& beta, const Accessor<T>& B, cublasOperation_t transb, const Accessor<T>& C);

				template<typename T>
				__host__ void geam(const Accessor<T>& A, cublasOperation_t transa, const Accessor<T>& B, cublasOperation_t transb, const Accessor<T>& C);

				template<typename T>
				__host__ void geam(const Accessor<T>& A, const Accessor<T>& B, const Accessor<T>& C);

			template<typename T>
			__host__ void dgmm(cublasSideMode_t mode, const Accessor<T>& A, const Accessor<T>& v, const Accessor<T>& C);

	};

} // namespace Kartet

	#include "BLASTools.hpp"

#endif

