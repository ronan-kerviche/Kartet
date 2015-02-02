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
	#include <curand.h>
	#include "Core/Exceptions.hpp"
	#include "Core/Array.hpp"

namespace Kartet
{
// Classes :
	class UniformSource
	{
		private :
			__host__ inline void apply(const Layout& mainLayout, const Layout& currentAccessLayout, float* ptr, size_t offset, int i, int j, int k) const;
			__host__ inline void apply(const Layout& mainLayout, const Layout& currentAccessLayout, double* ptr, size_t offset, int i, int j, int k) const;
			friend Layout;
			friend Accessor<float>;
			friend Accessor<double>;
		public :
			__host__ inline const Accessor<float>& operator>>(const Accessor<float>& a) const;
			__host__ inline const Accessor<double>& operator>>(const Accessor<double>& a) const;
	};

	class NormalSource
	{
		private :
			__host__ inline void apply(const Layout& mainLayout, const Layout& currentAccessLayout, float* ptr, size_t offset, int i, int j, int k) const;
			__host__ inline void apply(const Layout& mainLayout, const Layout& currentAccessLayout, double* ptr, size_t offset, int i, int j, int k) const;
			friend Layout;
			friend Accessor<float>;
			friend Accessor<double>;
		public :
			double mean, std;
			__host__ inline NormalSource(void);
			__host__ inline NormalSource(float _m, float _s);
			__host__ inline const Accessor<float>& operator>>(const Accessor<float>& a) const;
			__host__ inline const Accessor<double>& operator>>(const Accessor<double>& a) const;
	};

	class LogNormalSource
	{
		private :
			__host__ inline void apply(const Layout& mainLayout, const Layout& currentAccessLayout, float* ptr, size_t offset, int i, int j, int k) const;
			__host__ inline void apply(const Layout& mainLayout, const Layout& currentAccessLayout, double* ptr, size_t offset, int i, int j, int k) const;
			friend Layout;
			friend Accessor<float>;
			friend Accessor<double>;
		public :
			double mean, std;
			__host__ inline LogNormalSource(void);
			__host__ inline LogNormalSource(float _m, float _s);
			__host__ inline const Accessor<float>& operator>>(const Accessor<float>& a) const;
			__host__ inline const Accessor<double>& operator>>(const Accessor<double>& a) const;
	};

	class RandomSourceContext
	{
		private :
			template<typename T>
			struct StaticContainer
			{
				typedef StaticAssert< SameTypes<void,T>::test > TestAssertion; // Must use the void type to access the container.
				static RandomSourceContext* singleton;
			};

			curandGenerator_t gen;

			friend class UniformSource;
			friend class NormalSource;
			friend class LogNormalSource;

		public :
			__host__ inline RandomSourceContext(const curandRngType_t& rng_type = CURAND_RNG_PSEUDO_DEFAULT);
			__host__ inline ~RandomSourceContext(void);
	};

	template<typename T>
	RandomSourceContext* RandomSourceContext::StaticContainer<T>::singleton = NULL;

// Implementation :
	#define TEST_CONTEXT		if(RandomSourceContext::StaticContainer<void>::singleton==NULL) throw InvalidCuRandContext;
	#define GEN			(RandomSourceContext::StaticContainer<void>::singleton->gen)
	#define TEST_EXCEPTION(x)	if(x!=CURAND_STATUS_SUCCESS) throw static_cast<Exception>(CuRandExceptionOffset + x);

	__host__ inline RandomSourceContext::RandomSourceContext(const curandRngType_t& rng_type)
	{
		if(StaticContainer<void>::singleton==NULL)
		{
			curandStatus_t err = curandCreateGenerator(&gen, rng_type);
			TEST_EXCEPTION(err)
			StaticContainer<void>::singleton = this;
		}
	}

	__host__ inline RandomSourceContext::~RandomSourceContext(void)
	{
		if(StaticContainer<void>::singleton==this)
		{
			StaticContainer<void>::singleton = NULL;
			curandStatus_t err = curandDestroyGenerator(gen);
			TEST_EXCEPTION(err)
		}
	}

// Uniform :
	__host__ inline void UniformSource::apply(const Layout& mainLayout, const Layout& currentAccessLayout, float* ptr, size_t offset, int i, int j, int k) const
	{
		curandStatus_t err = curandGenerateUniform(GEN, ptr+offset, currentAccessLayout.getNumElements());
		TEST_EXCEPTION(err)
	}

	__host__ inline void UniformSource::apply(const Layout& mainLayout, const Layout& currentAccessLayout, double* ptr, size_t offset, int i, int j, int k) const
	{
		curandStatus_t err = curandGenerateUniformDouble(GEN, ptr+offset, currentAccessLayout.getNumElements());
		TEST_EXCEPTION(err)
	}

	__host__ inline const Accessor<float>& UniformSource::operator>>(const Accessor<float>& a) const
	{
		TEST_CONTEXT
		a.hostScan(*this);
		return a;
	}

	__host__ inline const Accessor<double>& UniformSource::operator>>(const Accessor<double>& a) const
	{
		TEST_CONTEXT
		a.hostScan(*this);
		return a;
	}

// Normal :
	__host__ inline NormalSource::NormalSource(void)
	 : 	mean(0.0), 
		std(1.0)
	{ }
	__host__ inline NormalSource::NormalSource(float _m, float _s)
	 : 	mean(_m), 
		std(_s)
	{ }

	__host__ inline void NormalSource::apply(const Layout& mainLayout, const Layout& currentAccessLayout, float* ptr, size_t offset, int i, int j, int k) const
	{
		curandStatus_t err = curandGenerateNormal(GEN, ptr+offset, currentAccessLayout.getNumElements(), mean, std);
		TEST_EXCEPTION(err)
	}
	
	__host__ inline void NormalSource::apply(const Layout& mainLayout, const Layout& currentAccessLayout, double* ptr, size_t offset, int i, int j, int k) const
	{
		curandStatus_t err = curandGenerateNormalDouble(GEN, ptr+offset, currentAccessLayout.getNumElements(), mean, std);
		TEST_EXCEPTION(err)
	}

	__host__ inline const Accessor<float>& NormalSource::operator>>(const Accessor<float>& a) const
	{
		TEST_CONTEXT
		a.hostScan(*this);
		return a;
	}

	__host__ inline const Accessor<double>& NormalSource::operator>>(const Accessor<double>& a) const
	{
		TEST_CONTEXT
		a.hostScan(*this);
		return a;
	}

// LogNormal :
	__host__ inline LogNormalSource::LogNormalSource(void)
	 : 	mean(0.0), 
		std(1.0)
	{ }
	__host__ inline LogNormalSource::LogNormalSource(float _m, float _s)
	 : 	mean(_m), 
		std(_s)
	{ }

	__host__ inline void LogNormalSource::apply(const Layout& mainLayout, const Layout& currentAccessLayout, float* ptr, size_t offset, int i, int j, int k) const
	{
		curandStatus_t err = curandGenerateLogNormal(GEN, ptr+offset, currentAccessLayout.getNumElements(), mean, std);
		TEST_EXCEPTION(err)
	}
	
	__host__ inline void LogNormalSource::apply(const Layout& mainLayout, const Layout& currentAccessLayout, double* ptr, size_t offset, int i, int j, int k) const
	{
		curandStatus_t err = curandGenerateLogNormalDouble(GEN, ptr+offset, currentAccessLayout.getNumElements(), mean, std);
		TEST_EXCEPTION(err)
	}

	__host__ inline const Accessor<float>& LogNormalSource::operator>>(const Accessor<float>& a) const
	{
		TEST_CONTEXT
		a.hostScan(*this);
		return a;
	}

	__host__ inline const Accessor<double>& LogNormalSource::operator>>(const Accessor<double>& a) const
	{
		TEST_CONTEXT
		a.hostScan(*this);
		return a;
	}

	#undef TEST_CONTEXT
	#undef TEST_MONOLITHIC
	#undef GEN
	#undef TEST_EXCEPTION

} // namespace Kartet

#endif

