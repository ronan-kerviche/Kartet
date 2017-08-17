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

#ifndef __KARTET_RANDOM_SOURCE__
#define __KARTET_RANDOM_SOURCE__

// Includes :
	#include <ctime>
	#include <cstdlib>
	
	#ifdef __CUDACC__
		#include <curand.h>
	#endif

	#include "Core/LibTools.hpp"
	#include "Core/Exceptions.hpp"
	#include "Core/Array.hpp"

namespace Kartet
{
// Classes :
	template<Location l=KARTET_DEFAULT_LOCATION>
	class RandomSourceContext
	{
		protected :
			#ifdef __CUDACC__
				curandGenerator_t gen;

				__host__ RandomSourceContext(const curandRngType_t& rngType = CURAND_RNG_PSEUDO_DEFAULT);
			#else
				__host__ RandomSourceContext(void);
			#endif
		public :
			__host__ ~RandomSourceContext(void);

			__host__ void setSeed(const unsigned long long& seed);
			__host__ void setSeed(void);
	};
	
	template<Location l=KARTET_DEFAULT_LOCATION>
	class UniformSource : public RandomSourceContext<l>
	{
		private :
			__host__ void apply(const Layout& mainLayout, const Layout& currentAccessLayout, float* ptr, size_t offset, int i, int j, int k) const;
			__host__ void apply(const Layout& mainLayout, const Layout& currentAccessLayout, double* ptr, size_t offset, int i, int j, int k) const;

			friend class Layout;
		public :
			#ifdef __CUDACC__
				__host__ UniformSource(const curandRngType_t& rngType = CURAND_RNG_PSEUDO_DEFAULT);
			#else
				__host__ UniformSource(void);
			#endif
			__host__ ~UniformSource(void);

			__host__ const Accessor<float,l>& operator>>(const Accessor<float,l>& a) const;
			__host__ const Accessor<double,l>& operator>>(const Accessor<double,l>& a) const;
	};

	template<Location l=KARTET_DEFAULT_LOCATION>
	class NormalSource : public RandomSourceContext<l>
	{
		private :
			__host__ void apply(const Layout& mainLayout, const Layout& currentAccessLayout, float* ptr, size_t offset, int i, int j, int k) const;
			__host__ void apply(const Layout& mainLayout, const Layout& currentAccessLayout, double* ptr, size_t offset, int i, int j, int k) const;

			friend class Layout;
		public :
			double mean, std;

			#ifdef __CUDACC__
				__host__ NormalSource(const curandRngType_t& rngType = CURAND_RNG_PSEUDO_DEFAULT);
				__host__ NormalSource(double _mean, double _std, const curandRngType_t& rngType = CURAND_RNG_PSEUDO_DEFAULT);
			#else
				__host__ NormalSource(void);
				__host__ NormalSource(double _mean, double _std);
			#endif
			__host__ ~NormalSource(void);

			__host__ const Accessor<float,l>& operator>>(const Accessor<float,l>& a) const;
			__host__ const Accessor<double,l>& operator>>(const Accessor<double,l>& a) const;
	};
	
	template<Location l=KARTET_DEFAULT_LOCATION>
	class LogNormalSource : public RandomSourceContext<l>
	{
		private :
			__host__ inline void apply(const Layout& mainLayout, const Layout& currentAccessLayout, float* ptr, size_t offset, int i, int j, int k) const;
			__host__ inline void apply(const Layout& mainLayout, const Layout& currentAccessLayout, double* ptr, size_t offset, int i, int j, int k) const;

			friend class Layout;
		public :
			double mean, std;

			#ifdef __CUDACC__
				__host__ LogNormalSource(const curandRngType_t& rngType = CURAND_RNG_PSEUDO_DEFAULT);
				__host__ LogNormalSource(double _mean, double _std, const curandRngType_t& rngType = CURAND_RNG_PSEUDO_DEFAULT);
			#else
				__host__ LogNormalSource(void);
				__host__ LogNormalSource(double _mean, double _std);
			#endif
			__host__ ~LogNormalSource(void);

			__host__ const Accessor<float,l>& operator>>(const Accessor<float,l>& a) const;
			__host__ const Accessor<double,l>& operator>>(const Accessor<double,l>& a) const;
	};

} // namespace Kartet

	#include "RandomSourceTools.hpp"

#endif

