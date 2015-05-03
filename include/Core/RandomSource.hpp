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

// Implementation :
	#define TEST_EXCEPTION(x)	if(x!=CURAND_STATUS_SUCCESS) throw static_cast<Exception>(CuRandExceptionOffset + x);

	template<Location l>
	#ifdef __CUDACC__	
	__host__ RandomSourceContext<l>::RandomSourceContext(const curandRngType_t& rngType)
	#else
	__host__ RandomSourceContext<l>::RandomSourceContext(void)
	#endif
	{
		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				curandStatus_t err = curandCreateGenerator(&gen, rngType);
				TEST_EXCEPTION(err)
			#else
				throw NotSupported;
			#endif
		}
		else
			std::cerr << "RandomSourceContext<HostSide>::RandomSourceContext - WARNING : Random number generator setting discarded on host side." << std::endl;
	}

	template<Location l>
	__host__ RandomSourceContext<l>::~RandomSourceContext(void)
	{
		#ifdef __CUDACC__
			if(l==DeviceSide)
			{
				curandStatus_t err = curandDestroyGenerator(gen);
				TEST_EXCEPTION(err)
			}
		#endif
	}

	template<Location l>
	__host__ void RandomSourceContext<l>::setSeed(const unsigned long long& seed)
	{
		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				curandStatus_t err = curandSetPseudoRandomGeneratorSeed(gen, seed);
				TEST_EXCEPTION(err)
			#else
				throw NotSupported;
			#endif
		}
		else
			std::cerr << "RandomSourceContext<HostSide>::setSeed(" << seed << ") - WARNING : seed discarded on host side." << std::endl;
	}

	template<Location l>
	__host__ void RandomSourceContext<l>::setSeed(void)
	{
		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				const unsigned long long seed = static_cast<unsigned long long>(rand()) << 32 +  static_cast<unsigned long long>(rand()); // use rand to fill the seed.
				curandStatus_t err = curandSetPseudoRandomGeneratorSeed(gen, seed);
				TEST_EXCEPTION(err)
			#else
				throw NotSupported;
			#endif
		}
		else
			std::cerr << "RandomSourceContext<HostSide>::setSeed(void) - WARNING : seed discarded on host side." << std::endl;
	}

// Uniform :
	template<Location l>
	#ifdef __CUDACC__
	__host__ UniformSource<l>::UniformSource(const curandRngType_t& rngType)
	 :	RandomSourceContext<l>(rngType)
	#else
	__host__ UniformSource<l>::UniformSource(void)
	#endif
	{ }

	template<Location l>
	__host__ UniformSource<l>::~UniformSource(void)
	{ }

	template<Location l>
	__host__ void UniformSource<l>::apply(const Layout& mainLayout, const Layout& currentAccessLayout, float* ptr, size_t offset, int i, int j, int k) const
	{
		UNUSED_PARAMETER(mainLayout)
		UNUSED_PARAMETER(i)
		UNUSED_PARAMETER(j)
		UNUSED_PARAMETER(k)
		
		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				curandStatus_t err = curandGenerateUniform(RandomSourceContext<l>::gen, ptr+offset, currentAccessLayout.getNumElements());
				TEST_EXCEPTION(err)
			#else 
				throw NotSupported;
			#endif
		}
		else
		{
			for(index_t p=0; p<currentAccessLayout.getNumElements(); p++)
				*(ptr+offset+p) = static_cast<float>(rand())/static_cast<float>(RAND_MAX);
		}
	}

	template<Location l>
	__host__ void UniformSource<l>::apply(const Layout& mainLayout, const Layout& currentAccessLayout, double* ptr, size_t offset, int i, int j, int k) const
	{
		UNUSED_PARAMETER(mainLayout)
		UNUSED_PARAMETER(i)
		UNUSED_PARAMETER(j)
		UNUSED_PARAMETER(k)

		if(l==DeviceSide)
		{
			#ifdef __CUDACC__
				curandStatus_t err = curandGenerateUniformDouble(RandomSourceContext<l>::gen, ptr+offset, currentAccessLayout.getNumElements());
				TEST_EXCEPTION(err)
			#else 
				throw NotSupported;
			#endif
		}
		else
		{
			for(index_t p=0; p<currentAccessLayout.getNumElements(); p++)
				*(ptr+offset+p) = static_cast<double>(rand())/static_cast<double>(RAND_MAX);
		}
	}

	template<Location l>
	__host__ const Accessor<float,l>& UniformSource<l>::operator>>(const Accessor<float,l>& a) const
	{
		a.hostScan(*this);
		return a;
	}

	template<Location l>
	__host__ const Accessor<double,l>& UniformSource<l>::operator>>(const Accessor<double,l>& a) const
	{
		a.hostScan(*this);
		return a;
	}

// Normal :
	template<Location l>
	#ifdef __CUDACC__
	__host__ NormalSource<l>::NormalSource(const curandRngType_t& rngType)
	 : 	RandomSourceContext<l>(rngType),
		mean(0.0), 
		std(1.0)
	#else
	__host__ NormalSource<l>::NormalSource(void)
	 : 	mean(0.0), 
		std(1.0)
	#endif
	{ }

	template<Location l>
	#ifdef __CUDACC__
	__host__ NormalSource<l>::NormalSource(double _mean, double _std, const curandRngType_t& rngType)
	 : 	RandomSourceContext<l>(rngType),
		mean(_mean), 
		std(_std)
	#else
	__host__ NormalSource<l>::NormalSource(double _mean, double _std)
	 : 	mean(_mean), 
		std(_std)
	#endif
	{ }

	template<Location l>
	__host__ NormalSource<l>::~NormalSource(void)
	{ }

	template<Location l>
	__host__ void NormalSource<l>::apply(const Layout& mainLayout, const Layout& currentAccessLayout, float* ptr, size_t offset, int i, int j, int k) const
	{
		UNUSED_PARAMETER(mainLayout)
		UNUSED_PARAMETER(i)
		UNUSED_PARAMETER(j)
		UNUSED_PARAMETER(k)

		#ifdef __CUDACC__
			if(l==DeviceSide)
			{
				curandStatus_t err = curandGenerateNormal(RandomSourceContext<l>::gen, ptr+offset, currentAccessLayout.getNumElements(), mean, std);
				TEST_EXCEPTION(err)
			}
			else
		#endif
				throw NotSupported;
	}
	
	template<Location l>
	__host__ void NormalSource<l>::apply(const Layout& mainLayout, const Layout& currentAccessLayout, double* ptr, size_t offset, int i, int j, int k) const
	{
		UNUSED_PARAMETER(mainLayout)
		UNUSED_PARAMETER(i)
		UNUSED_PARAMETER(j)
		UNUSED_PARAMETER(k)

		#ifdef __CUDACC__
			if(l==DeviceSide)
			{
				curandStatus_t err = curandGenerateNormalDouble(RandomSourceContext<l>::gen, ptr+offset, currentAccessLayout.getNumElements(), mean, std);
				TEST_EXCEPTION(err)
			}
			else
		#endif
				throw NotSupported;
	}

	template<Location l>
	__host__ const Accessor<float,l>& NormalSource<l>::operator>>(const Accessor<float,l>& a) const
	{
		if(l!=DeviceSide)
			throw NotSupported;

		a.hostScan(*this);
		return a;
	}

	template<Location l>
	__host__ const Accessor<double,l>& NormalSource<l>::operator>>(const Accessor<double,l>& a) const
	{
		if(l!=DeviceSide)
			throw NotSupported;

		a.hostScan(*this);
		return a;
	}

// LogNormal :
	template<Location l>
	#ifdef __CUDACC__
	__host__ inline LogNormalSource<l>::LogNormalSource(const curandRngType_t& rngType)
	 : 	RandomSourceContext<l>(rngType),
		mean(0.0), 
		std(1.0)
	#else
	__host__ inline LogNormalSource<l>::LogNormalSource(void)
	 : 	mean(0.0), 
		std(1.0)
	#endif
	{ }

	template<Location l>
	#ifdef __CUDACC__
	__host__ LogNormalSource<l>::LogNormalSource(double _mean, double _std, const curandRngType_t& rngType)
	 : 	RandomSourceContext<l>(rngType),
		mean(_mean), 
		std(_std)
	#else
	__host__ LogNormalSource<l>::LogNormalSource(double _mean, double _std)
	 : 	mean(_mean), 
		std(_std)
	#endif
	{ }	

	template<Location l>
	__host__ LogNormalSource<l>::~LogNormalSource(void)
	{ }

	template<Location l>
	__host__ void LogNormalSource<l>::apply(const Layout& mainLayout, const Layout& currentAccessLayout, float* ptr, size_t offset, int i, int j, int k) const
	{
		UNUSED_PARAMETER(mainLayout)
		UNUSED_PARAMETER(i)
		UNUSED_PARAMETER(j)
		UNUSED_PARAMETER(k)

		#ifdef __CUDACC__
			if(l==DeviceSide)
			{
				curandStatus_t err = curandGenerateLogNormal(RandomSourceContext<l>::gen, ptr+offset, currentAccessLayout.getNumElements(), mean, std);
				TEST_EXCEPTION(err)
			}
			else
		#endif
				throw NotSupported;
	}
	
	template<Location l>
	__host__ void LogNormalSource<l>::apply(const Layout& mainLayout, const Layout& currentAccessLayout, double* ptr, size_t offset, int i, int j, int k) const
	{
		UNUSED_PARAMETER(mainLayout)
		UNUSED_PARAMETER(i)
		UNUSED_PARAMETER(j)
		UNUSED_PARAMETER(k)

		#ifdef __CUDACC__
			if(l==DeviceSide)
			{
				curandStatus_t err = curandGenerateLogNormalDouble(RandomSourceContext<l>::gen, ptr+offset, currentAccessLayout.getNumElements(), mean, std);
				TEST_EXCEPTION(err)
			}
			else
		#endif
				throw NotSupported;
	}

	template<Location l>
	__host__ const Accessor<float,l>& LogNormalSource<l>::operator>>(const Accessor<float,l>& a) const
	{
		if(l!=DeviceSide)
			throw NotSupported;

		a.hostScan(*this);
		return a;
	}

	template<Location l>
	__host__ const Accessor<double,l>& LogNormalSource<l>::operator>>(const Accessor<double,l>& a) const
	{
		if(l!=DeviceSide)
			throw NotSupported;

		a.hostScan(*this);
		return a;
	}

	#undef TEST_EXCEPTION

} // namespace Kartet

#endif

