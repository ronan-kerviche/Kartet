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

	#ifdef __CUDACC__
		#include <cublas_v2.h>
	#endif

	#ifdef KARTET_USE_ATLAS
		#ifdef __cplusplus
		extern "C"
		{
		#endif
			#include <atlas/cblas.h>
		#ifdef __cplusplus
		}
		#endif
	#endif

	#include "Core/LibTools.hpp"
	#include "Core/Array.hpp"

namespace Kartet
{
	enum MatrixOperation
	{
		OpNo = 'N',
		OpTr = 'T',
		OpHr = 'C'
	};

	enum MatrixFillMode
	{
		MatrixFillUp = 'U',
		MatrixFillLow = 'L'
	};

	enum MatrixDiagType
	{
		MatrixDiagNonUnit = 'N',
		MatrixDiagUnit = 'U'
	};

	enum MatrixSideMode
	{
		MatrixLeftSide = 'L',
		MatrixRightSide = 'R'
	};

// BLASContext :
	class BLASContext
	{
		private :
			#ifdef __CUDACC__
				cublasHandle_t 	handle;
			#endif

			__host__ inline BLASContext(const BLASContext&);
		public :
			__host__ inline BLASContext(bool initDevice=true);
			__host__ inline ~BLASContext(void);

			// Converters :
			#ifdef __CUDACC__
			__host__ static inline cublasOperation_t getCuBLASOperation(const MatrixOperation& op);
			__host__ static inline cublasFillMode_t getCuBLASFillMode(const MatrixFillMode& m);
			__host__ static inline cublasDiagType_t getCuBLASDiagType(const MatrixDiagType& t);
			__host__ static inline cublasSideMode_t getCuBLASSideMode(const MatrixSideMode& s);
			#endif

			#ifdef KARTET_USE_ATLAS
			__host__ static inline CBLAS_TRANSPOSE getCBLASOperation(const MatrixOperation& op);
			__host__ static inline CBLAS_UPLO getCBLASFillMode(const MatrixFillMode& m);
			__host__ static inline CBLAS_DIAG getCBLASDiagType(const MatrixDiagType& t);
			__host__ static inline CBLAS_SIDE getCBLASSideMode(const MatrixSideMode& s);
			#endif

			// Layout tools :
			__host__ static inline bool isProductValid(const Layout& A, MatrixOperation opa, const Layout& B, MatrixOperation opb, const Layout& C);
			__host__ static inline bool isProductValid(const Layout& A, const Layout& B, const Layout& C);
			__host__ static inline Layout getProductLayout(const Layout& A, MatrixOperation opa, const Layout& B, MatrixOperation opb);
			__host__ static inline Layout getProductLayout(const Layout& A, const Layout& B);
			

			template<typename T, Location l>
			__host__ int Iamax(const Accessor<T,l>& x);

			template<typename T, Location l>
			__host__ int Iamin(const Accessor<T,l>& x);

			template<typename T, Location l>
			__host__ typename TypeInfo<T>::BaseType asum(const Accessor<T,l>& x);

			template<typename T, Location l>
			__host__ T dot(const Accessor<T,l>& x, const Accessor<T,l>& y, bool conjugate = true);

			template<typename T, Location l>
			__host__ typename TypeInfo<T>::BaseType nrm2(const Accessor<T,l>& a);

			template<typename T, Location l, typename TAlpha, typename TBeta>
			__host__ void gemv(const TAlpha& alpha, const Accessor<T,l>& A, MatrixOperation op, const Accessor<T,l>& x, const TBeta& beta, const Accessor<T,l>& y);

				template<typename T, Location l>
				__host__ void gemv(const Accessor<T,l>& A, MatrixOperation op, const Accessor<T,l>& x, const Accessor<T,l>& y);

				template<typename T, Location l>
				__host__ void gemv(const Accessor<T,l>& A, const Accessor<T,l>& x, const Accessor<T,l>& y);

			template<typename T, Location l, typename TAlpha>
			__host__ void ger(const TAlpha& alpha, const Accessor<T,l>& x, const Accessor<T,l>& y, const Accessor<T,l>& A, bool conjugate = true);

				template<typename T, Location l>
				__host__ void ger(const Accessor<T,l>& x, const Accessor<T,l>& y, const Accessor<T,l>& A, bool conjugate = true);

			template<typename T, Location l, typename TAlpha, typename TBeta>
			__host__ void symv(const TAlpha& alpha, MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x, const TBeta& beta, const Accessor<T,l>& y);

				template<typename T, Location l>
				__host__ void symv(MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x, const Accessor<T,l>& y);
			
			template<typename T, Location l, typename TAlpha>
			__host__ void syr(const TAlpha& alpha, MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x);

				template<typename T, Location l>
				__host__ void syr(MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x);

			template<typename T, Location l, typename TAlpha>
			__host__ void syr2(const TAlpha& alpha, MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x, const Accessor<T,l>& y);
		
				template<typename T, Location l>
				__host__ void syr2(MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x, const Accessor<T,l>& y);

			template<typename T, Location l>
			__host__ void trmv(MatrixFillMode uplo, MatrixDiagType diag, const Accessor<T,l>& A, MatrixOperation op, const Accessor<T,l>& x);

			template<typename T, Location l>
			__host__ void trsv(MatrixFillMode uplo, MatrixDiagType diag, const Accessor<T,l>& A, MatrixOperation op, const Accessor<T,l>& x);

			template<typename T, Location l, typename TAlpha, typename TBeta>
			__host__ void hemv(const TAlpha& alpha, MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x, const TBeta& beta, const Accessor<T,l>& y);

				template<typename T, Location l>
				__host__ void hemv(MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x, const Accessor<T,l>& y);

			template<typename T, Location l, typename TAlpha>
			__host__ void her(const TAlpha& alpha, MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x);

				template<typename T, Location l>
				__host__ void her(MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x);

			template<typename T, Location l, typename TAlpha>
			__host__ void her2(const TAlpha& alpha, MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x, const Accessor<T,l>& y);

				template<typename T, Location l>
				__host__ void her2(MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& x, const Accessor<T,l>& y);

			template<typename T, Location l, typename TAlpha, typename TBeta>
			__host__ void gemm(const TAlpha& alpha, const Accessor<T,l>& A, MatrixOperation opa, const Accessor<T,l>& B, MatrixOperation opb, const TBeta& beta, const Accessor<T,l>& C);

				template<typename T, Location l>
				__host__ void gemm(const Accessor<T,l>& A, MatrixOperation opa, const Accessor<T,l>& B, MatrixOperation opb, const Accessor<T,l>& C);

				template<typename T, Location l>
				__host__ void gemm(const Accessor<T,l>& A, const Accessor<T,l>& B, const Accessor<T,l>& C);

			template<typename T, Location l, typename TAlpha, typename TBeta>
			__host__ void symm(MatrixSideMode side, const TAlpha& alpha, MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& B, const TBeta& beta, const Accessor<T,l>& C);

				template<typename T, Location l>
				__host__ void symm(MatrixSideMode side, MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& B, const Accessor<T,l>& C);

			template<typename T, Location l, typename TAlpha, typename TBeta>
			__host__ void syrk(const TAlpha& alpha, const Accessor<T,l>& A, MatrixOperation opa, const TBeta& beta, MatrixFillMode uplo, const Accessor<T,l>& C);

				template<typename T, Location l>
				__host__ void syrk(const Accessor<T,l>& A, MatrixOperation opa, MatrixFillMode uplo, const Accessor<T,l>& C);

				template<typename T, Location l>
				__host__ void syrk(const Accessor<T,l>& A, MatrixFillMode uplo, const Accessor<T,l>& C);

			template<typename T, Location l, typename TAlpha>
			__host__ void syr2k(MatrixOperation op, const TAlpha& alpha, const Accessor<T,l>& A, const Accessor<T,l>& B, const T& beta, MatrixFillMode uplo, const Accessor<T,l>& C);

				template<typename T, Location l>
				__host__ void syr2k(MatrixOperation op, const Accessor<T,l>& A, const Accessor<T,l>& B, MatrixFillMode uplo, const Accessor<T,l>& C);

				template<typename T, Location l>
				__host__ void syr2k(const Accessor<T,l>& A, const Accessor<T,l>& B, MatrixFillMode uplo, const Accessor<T,l>& C);

			template<typename T, Location l, typename TAlpha>
			__host__ void trmm(MatrixSideMode side, const TAlpha& alpha, MatrixFillMode uplo, MatrixDiagType diag, const Accessor<T,l>& A, MatrixOperation opa, const Accessor<T,l>& B, const Accessor<T,l>& C);

				template<typename T, Location l>
				__host__ void trmm(MatrixSideMode side, MatrixFillMode uplo, MatrixDiagType diag, const Accessor<T,l>& A, MatrixOperation opa, const Accessor<T,l>& B, const Accessor<T,l>& C);
		
				template<typename T, Location l>
				__host__ void trmm(MatrixSideMode side, MatrixFillMode uplo, MatrixDiagType diag, const Accessor<T,l>& A, const Accessor<T,l>& B, const Accessor<T,l>& C);

			template<typename T, Location l, typename TAlpha>
			__host__ void trsm(MatrixSideMode side, MatrixFillMode uplo, MatrixDiagType diag, const Accessor<T,l>& A, MatrixOperation opa, const TAlpha& alpha, const Accessor<T,l>& B);

				template<typename T, Location l>
				__host__ void trsm(MatrixSideMode side, MatrixFillMode uplo, MatrixDiagType diag, const Accessor<T,l>& A, MatrixOperation opa, const Accessor<T,l>& B);

				template<typename T, Location l>
				__host__ void trsm(MatrixSideMode side, MatrixFillMode uplo, MatrixDiagType diag, const Accessor<T,l>& A, const Accessor<T,l>& B);

			template<typename T, Location l, typename TAlpha, typename TBeta>
			__host__ void hemm(MatrixSideMode side, const TAlpha& alpha, MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& B, const TBeta& beta, const Accessor<T,l>& C);

				template<typename T, Location l>
				__host__ void hemm(MatrixSideMode side, MatrixFillMode uplo, const Accessor<T,l>& A, const Accessor<T,l>& B, const Accessor<T,l>& C);

			template<typename T, Location l, typename TAlpha, typename TBeta>
			__host__ void herk(const TAlpha& alpha, const Accessor<T,l>& A, MatrixOperation opa, const TBeta& beta, MatrixFillMode uplo, const Accessor<T,l>& C);
		
				template<typename T, Location l>
				__host__ void herk(const Accessor<T,l>& A, MatrixOperation opa, MatrixFillMode uplo, const Accessor<T,l>& C);

				template<typename T, Location l>
				__host__ void herk(const Accessor<T,l>& A, MatrixFillMode uplo, const Accessor<T,l>& C);

			template<typename T, Location l, typename TAlpha, typename TBeta>
			__host__ void her2k(MatrixOperation op, const TAlpha& alpha, const Accessor<T,l>& A, const Accessor<T,l>& B, const TBeta& beta, MatrixFillMode uplo, const Accessor<T,l>& C);

				template<typename T, Location l>
				__host__ void her2k(MatrixOperation op, const Accessor<T,l>& A, const Accessor<T,l>& B, MatrixFillMode uplo, const Accessor<T,l>& C);

				template<typename T, Location l>
				__host__ void her2k(const Accessor<T,l>& A, const Accessor<T,l>& B, MatrixFillMode uplo, const Accessor<T,l>& C);

			template<typename T, Location l, typename TAlpha, typename TBeta>
			__host__ void geam(const TAlpha& alpha, const Accessor<T,l>& A, MatrixOperation opa, const TBeta& beta, const Accessor<T,l>& B, MatrixOperation opb, const Accessor<T,l>& C);

				template<typename T, Location l>
				__host__ void geam(const Accessor<T,l>& A, MatrixOperation opa, const Accessor<T,l>& B, MatrixOperation opb, const Accessor<T,l>& C);

				template<typename T, Location l>
				__host__ void geam(const Accessor<T,l>& A, const Accessor<T,l>& B, const Accessor<T,l>& C);

			template<typename T, Location l>
			__host__ void dgmm(MatrixSideMode side, const Accessor<T,l>& A, const Accessor<T,l>& v, const Accessor<T,l>& C);

	};

} // namespace Kartet

	#include "BLASTools.hpp"

#endif

