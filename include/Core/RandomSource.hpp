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
	#include "Core/LibTools.hpp"
	#include "Core/Exceptions.hpp"
	#include "Core/Array.hpp"

namespace Kartet
{
// Classes :
	class RandomSourceContext
	{
		protected :
			curandGenerator_t gen;

			//__host__ RandomSourceContext(const RandomSourceContext&);
			__host__ inline RandomSourceContext(const curandRngType_t& rngType = CURAND_RNG_PSEUDO_DEFAULT);
		public :
			__host__ inline ~RandomSourceContext(void);

			__host__ inline void setSeed(const unsigned long long& seed);
			__host__ inline void setSeed(void);
	};
	
	class UniformSource : public RandomSourceContext
	{
		private :
			__host__ inline void apply(const Layout& mainLayout, const Layout& currentAccessLayout, float* ptr, size_t offset, int i, int j, int k) const;
			__host__ inline void apply(const Layout& mainLayout, const Layout& currentAccessLayout, double* ptr, size_t offset, int i, int j, int k) const;

			friend Layout;
			friend Accessor<float>;
			friend Accessor<double>;

			//__host__ UniformSource(const UniformSource&);
		public :
			__host__ inline UniformSource(const curandRngType_t& rngType = CURAND_RNG_PSEUDO_DEFAULT);
			__host__ inline ~UniformSource(void);

			__host__ inline const Accessor<float>& operator>>(const Accessor<float>& a) const;
			__host__ inline const Accessor<double>& operator>>(const Accessor<double>& a) const;
	};

	class NormalSource : public RandomSourceContext
	{
		private :
			__host__ inline void apply(const Layout& mainLayout, const Layout& currentAccessLayout, float* ptr, size_t offset, int i, int j, int k) const;
			__host__ inline void apply(const Layout& mainLayout, const Layout& currentAccessLayout, double* ptr, size_t offset, int i, int j, int k) const;

			friend Layout;
			friend Accessor<float>;
			friend Accessor<double>;

			//__host__ NormalSource(const NormalSource&);
		public :
			double mean, std;

			__host__ inline NormalSource(const curandRngType_t& rngType = CURAND_RNG_PSEUDO_DEFAULT);
			__host__ inline NormalSource(double _mean, double _std, const curandRngType_t& rngType = CURAND_RNG_PSEUDO_DEFAULT);
			__host__ inline ~NormalSource(void);

			__host__ inline const Accessor<float>& operator>>(const Accessor<float>& a) const;
			__host__ inline const Accessor<double>& operator>>(const Accessor<double>& a) const;
	};

	class LogNormalSource : public RandomSourceContext
	{
		private :
			__host__ inline void apply(const Layout& mainLayout, const Layout& currentAccessLayout, float* ptr, size_t offset, int i, int j, int k) const;
			__host__ inline void apply(const Layout& mainLayout, const Layout& currentAccessLayout, double* ptr, size_t offset, int i, int j, int k) const;

			friend Layout;
			friend Accessor<float>;
			friend Accessor<double>;

			//__host__ LogNormalSource(const LogNormalSource&);
		public :
			double mean, std;

			__host__ inline LogNormalSource(const curandRngType_t& rngType = CURAND_RNG_PSEUDO_DEFAULT);
			__host__ inline LogNormalSource(double _mean, double _std, const curandRngType_t& rngType = CURAND_RNG_PSEUDO_DEFAULT);
			__host__ inline ~LogNormalSource(void);

			__host__ inline const Accessor<float>& operator>>(const Accessor<float>& a) const;
			__host__ inline const Accessor<double>& operator>>(const Accessor<double>& a) const;
	};

// Implementation :
	#define TEST_EXCEPTION(x)	if(x!=CURAND_STATUS_SUCCESS) throw static_cast<Exception>(CuRandExceptionOffset + x);

	__host__ inline RandomSourceContext::RandomSourceContext(const curandRngType_t& rngType)
	{
		curandStatus_t err = curandCreateGenerator(&gen, rngType);
		TEST_EXCEPTION(err)
	}

	__host__ inline RandomSourceContext::~RandomSourceContext(void)
	{
		curandStatus_t err = curandDestroyGenerator(gen);
		TEST_EXCEPTION(err)
	}

	__host__ inline void RandomSourceContext::setSeed(const unsigned long long& seed)
	{
		curandStatus_t err = curandSetPseudoRandomGeneratorSeed(gen, seed);
		TEST_EXCEPTION(err)
	}

	__host__ inline void RandomSourceContext::setSeed(void)
	{
		const unsigned long long seed = static_cast<unsigned long long>(rand()) << 32 +  static_cast<unsigned long long>(rand()); // use rand to fill the seed.
		curandStatus_t err = curandSetPseudoRandomGeneratorSeed(gen, seed);
		TEST_EXCEPTION(err)
	}

// Uniform :
	__host__ inline UniformSource::UniformSource(const curandRngType_t& rngType)
	 :	RandomSourceContext(rngType)	
	{ }

	__host__ inline UniformSource::~UniformSource(void)
	{ }

	__host__ inline void UniformSource::apply(const Layout& mainLayout, const Layout& currentAccessLayout, float* ptr, size_t offset, int i, int j, int k) const
	{
		UNUSED_PARAMETER(mainLayout)
		UNUSED_PARAMETER(i)
		UNUSED_PARAMETER(j)
		UNUSED_PARAMETER(k)
		curandStatus_t err = curandGenerateUniform(gen, ptr+offset, currentAccessLayout.getNumElements());
		TEST_EXCEPTION(err)
	}

	__host__ inline void UniformSource::apply(const Layout& mainLayout, const Layout& currentAccessLayout, double* ptr, size_t offset, int i, int j, int k) const
	{
		UNUSED_PARAMETER(mainLayout)
		UNUSED_PARAMETER(i)
		UNUSED_PARAMETER(j)
		UNUSED_PARAMETER(k)
		curandStatus_t err = curandGenerateUniformDouble(gen, ptr+offset, currentAccessLayout.getNumElements());
		TEST_EXCEPTION(err)
	}

	__host__ inline const Accessor<float>& UniformSource::operator>>(const Accessor<float>& a) const
	{
		a.hostScan(*this);
		return a;
	}

	__host__ inline const Accessor<double>& UniformSource::operator>>(const Accessor<double>& a) const
	{
		a.hostScan(*this);
		return a;
	}

// Normal :
	__host__ inline NormalSource::NormalSource(const curandRngType_t& rngType)
	 : 	RandomSourceContext(rngType),
		mean(0.0), 
		std(1.0)
	{ }
	__host__ inline NormalSource::NormalSource(double _mean, double _std, const curandRngType_t& rngType)
	 : 	RandomSourceContext(rngType),
		mean(_mean), 
		std(_std)
	{ }

	__host__ inline NormalSource::~NormalSource(void)
	{ }

	__host__ inline void NormalSource::apply(const Layout& mainLayout, const Layout& currentAccessLayout, float* ptr, size_t offset, int i, int j, int k) const
	{
		UNUSED_PARAMETER(mainLayout)
		UNUSED_PARAMETER(i)
		UNUSED_PARAMETER(j)
		UNUSED_PARAMETER(k)
		curandStatus_t err = curandGenerateNormal(gen, ptr+offset, currentAccessLayout.getNumElements(), mean, std);
		TEST_EXCEPTION(err)
	}
	
	__host__ inline void NormalSource::apply(const Layout& mainLayout, const Layout& currentAccessLayout, double* ptr, size_t offset, int i, int j, int k) const
	{
		UNUSED_PARAMETER(mainLayout)
		UNUSED_PARAMETER(i)
		UNUSED_PARAMETER(j)
		UNUSED_PARAMETER(k)
		curandStatus_t err = curandGenerateNormalDouble(gen, ptr+offset, currentAccessLayout.getNumElements(), mean, std);
		TEST_EXCEPTION(err)
	}

	__host__ inline const Accessor<float>& NormalSource::operator>>(const Accessor<float>& a) const
	{
		a.hostScan(*this);
		return a;
	}

	__host__ inline const Accessor<double>& NormalSource::operator>>(const Accessor<double>& a) const
	{
		a.hostScan(*this);
		return a;
	}

// LogNormal :
	__host__ inline LogNormalSource::LogNormalSource(const curandRngType_t& rngType)
	 : 	RandomSourceContext(rngType),
		mean(0.0), 
		std(1.0)
	{ }
	__host__ inline LogNormalSource::LogNormalSource(double _mean, double _std, const curandRngType_t& rngType)
	 : 	RandomSourceContext(rngType),
		mean(_mean), 
		std(_std)
	{ }

	__host__ inline LogNormalSource::~LogNormalSource(void)
	{ }

	__host__ inline void LogNormalSource::apply(const Layout& mainLayout, const Layout& currentAccessLayout, float* ptr, size_t offset, int i, int j, int k) const
	{
		UNUSED_PARAMETER(mainLayout)
		UNUSED_PARAMETER(i)
		UNUSED_PARAMETER(j)
		UNUSED_PARAMETER(k)
		curandStatus_t err = curandGenerateLogNormal(gen, ptr+offset, currentAccessLayout.getNumElements(), mean, std);
		TEST_EXCEPTION(err)
	}
	
	__host__ inline void LogNormalSource::apply(const Layout& mainLayout, const Layout& currentAccessLayout, double* ptr, size_t offset, int i, int j, int k) const
	{
		UNUSED_PARAMETER(mainLayout)
		UNUSED_PARAMETER(i)
		UNUSED_PARAMETER(j)
		UNUSED_PARAMETER(k)
		curandStatus_t err = curandGenerateLogNormalDouble(gen, ptr+offset, currentAccessLayout.getNumElements(), mean, std);
		TEST_EXCEPTION(err)
	}

	__host__ inline const Accessor<float>& LogNormalSource::operator>>(const Accessor<float>& a) const
	{
		a.hostScan(*this);
		return a;
	}

	__host__ inline const Accessor<double>& LogNormalSource::operator>>(const Accessor<double>& a) const
	{
		a.hostScan(*this);
		return a;
	}

	#undef TEST_EXCEPTION

} // namespace Kartet

#endif

