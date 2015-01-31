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

#ifndef __KARTET_ARRAY_TOOLS__
#define __KARTET_ARRAY_TOOLS__

namespace Kartet
{
// Layout :
	__host__ __device__ inline Layout::Layout(index_t r, index_t c, index_t s, index_t lc, index_t ls)
	 : 	numRows(r),
		numColumns(c),
		numSlices(s),
		leadingColumns(lc),
		leadingSlices(ls)
	{
		if(leadingColumns==0)
			leadingColumns = numRows;
		if(leadingSlices==0)
			leadingSlices = numRows*numColumns;
	}

	__host__ __device__ inline Layout::Layout(const Layout& l)
	 : 	numRows(l.numRows),
		numColumns(l.numColumns),
		numSlices(l.numSlices),
		leadingColumns(l.leadingColumns),
		leadingSlices(l.leadingSlices)
	{ }

	__host__ __device__ inline index_t Layout::getNumElements(void) const
	{
		return (numRows * numColumns * numSlices);
	}
	
	__host__ __device__ inline index_t Layout::getNumElementsPerSlice(void) const
	{
		return (numRows * numColumns);
	}

	__host__ __device__ inline index_t Layout::getNumRows(void) const
	{
		return numRows;
	}

	__host__ __device__ inline index_t Layout::getNumColumns(void) const
	{
		return numColumns;
	}

	__host__ __device__ inline index_t Layout::getNumSlices(void) const
	{
		return numSlices;
	}

	__host__ __device__ inline index_t Layout::getWidth(void) const
	{
		return numColumns;
	}

	__host__ __device__ inline index_t Layout::getHeight(void) const
	{
		return numRows;
	}

	__host__ __device__ inline index_t Layout::getDepth(void) const
	{
		return numSlices;
	}

	__host__ __device__ inline index_t Layout::getLeadingColumns(void) const
	{
		return leadingColumns;
	}

	__host__ __device__ inline index_t Layout::getLeadingSlices(void) const
	{
		return leadingSlices;
	}

	__host__ __device__ inline dim3 Layout::getDimensions(void) const
	{
		return dim3(numRows, numColumns, numSlices);
	}

	__host__ inline void Layout::reinterpretLayout(index_t r, index_t c, index_t s)
	{
		if(r!=numRows && numRows!=leadingColumns) // Modification of the number of rows while interlaced into a larger memory area.
			throw InvalidLayoutChange;
		else if((r*c)!=(numRows*numColumns) && (numRows*numColumns)!=leadingSlices)
			throw InvalidLayoutChange;
		else if(r*c*s!=getNumElements())
			throw InvalidLayoutChange;
		else
		{
			// Preset :
			if(r!=numRows)
				leadingColumns	= r;
			if((r*c)!=(numRows*numColumns))		
				leadingSlices	= r*c;

			numRows			= r;
			numColumns		= c;
			numSlices 		= s;

			// Simplify :
			if(numColumns==1)
				leadingColumns	= numRows;
			if(numSlices==1)
				leadingSlices	= numRows * numColumns;
		}
	}

	__host__ inline void Layout::reinterpretLayout(const Layout& other)
	{
		reinterpretLayout(other.getNumRows(), other.getNumColumns(), other.getNumSlices());
	}

	__host__ inline void Layout::flatten(void)
	{
		reinterpretLayout(numRows, numColumns*numSlices, 1);
	}

	__host__ inline void Layout::vectorize(void)
	{
		reinterpretLayout(getNumElements(), 1, 1);
	}

	__host__ __device__ inline bool Layout::sameLayoutAs(const Layout& other) const
	{
		return (numRows==other.numRows && numColumns==other.numColumns && numSlices==other.numSlices); 
	}

	__host__ __device__ inline bool Layout::sameSliceLayoutAs(const Layout& other) const
	{
		return (numRows==other.numRows && numColumns==other.numColumns); 
	}

	__device__ inline index_t Layout::getI(void)
	{
		return blockIdx.y*blockDim.y+threadIdx.y;
	}

	__device__ inline index_t Layout::getJ(void)
	{
		return blockIdx.x*blockDim.x+threadIdx.x;
	}

	__device__ inline index_t Layout::getK(void)
	{
		return blockIdx.z*blockDim.z+threadIdx.z;
	}

	template<typename TOut>
	__device__ inline TOut Layout::getINorm(index_t i) const
	{
		return static_cast<TOut>(i)/static_cast<index_t>(numRows);
	}

	template<typename TOut>
	__device__ inline TOut Layout::getJNorm(index_t j) const
	{
		return static_cast<TOut>(j)/static_cast<index_t>(numColumns);
	}

	template<typename TOut>
	__device__ inline TOut Layout::getKNorm(index_t k) const
	{
		return static_cast<TOut>(k)/static_cast<TOut>(numSlices);
	}

	template<typename TOut>
	__device__ inline TOut Layout::getINorm(void) const
	{
		return getINorm<TOut>(getI());
	}

	template<typename TOut>
	__device__ inline TOut Layout::getJNorm(void) const
	{
		return getJNorm<TOut>(getJ());
	}

	template<typename TOut>
	__device__ inline TOut Layout::getKNorm(void) const
	{
		return getKNorm<TOut>(getK());
	}

	template<typename TOut>
	__device__ inline TOut Layout::getINormIncl(index_t i) const
	{
		return static_cast<TOut>(i)/static_cast<TOut>(numRows-1);
	}

	template<typename TOut>
	__device__ inline TOut Layout::getJNormIncl(index_t j) const
	{
		return static_cast<TOut>(j)/static_cast<TOut>(numColumns-1);
	}

	template<typename TOut>
	__device__ inline TOut Layout::getKNormIncl(index_t k) const
	{
		return static_cast<TOut>(k)/static_cast<TOut>(numSlices-1);
	}

	template<typename TOut>
	__device__ inline TOut Layout::getINormIncl(void) const
	{
		return getINormIncl<TOut>(getI());
	}

	template<typename TOut>
	__device__ inline TOut Layout::getJNormIncl(void) const
	{
		return getJNormIncl<TOut>(getJ());
	}

	template<typename TOut>
	__device__ inline TOut Layout::getKNormIncl(void) const
	{
		return getKNormIncl<TOut>(getK());
	}

	__device__ inline index_t Layout::getIClamped(index_t i) const
	{
		return min( max(static_cast<index_t>(0), i), numRows-1);
	}

	__device__ inline index_t Layout::getJClamped(index_t j) const
	{
		return min( max(static_cast<index_t>(0), j), numColumns-1);
	}

	__device__ inline index_t Layout::getKClamped(index_t k) const
	{
		return min( max(static_cast<index_t>(0), k), numSlices-1);
	}

	__device__ inline index_t Layout::getIWrapped(index_t i) const
	{
		return (i % numRows);
	}

	__device__ inline index_t Layout::getJWrapped(index_t j) const
	{
		return (j % numColumns);
	}

	__device__ inline index_t Layout::getKWrapped(index_t k) const
	{
		return (k % numSlices);	
	}

	__host__ __device__ inline index_t Layout::getIndex(index_t i, index_t j, index_t k) const
	{
		return k*leadingSlices + j*leadingColumns + i;
	}

	__device__ inline index_t Layout::getIndex(void) const
	{
		return getIndex(getI(), getJ(), getK());
	}
	
	__host__ __device__ inline index_t Layout::getIndicesFFTShift(index_t& i, index_t& j, index_t& k) const
	{
		const index_t	hi = numRows % 2,
				hj = numColumns % 2;

		if(i<(numRows-hi)/2) 	j = j + (numRows+hi)/2;
		else			j = j - (numRows-hi)/2;

		if(j<(numColumns-hj)/2) i = i + (numColumns+hj)/2;
		else 			i = i - (numColumns-hj)/2;

		return getIndex(i, j, k);
	}

	__host__ __device__ inline index_t Layout::getIndexFFTShift(index_t i, index_t j, index_t k) const
	{
		return getIndicesFFTShift(i, j, k);
	}

	__host__ __device__ inline index_t Layout::getIndicesFFTInverseShift(index_t& i, index_t& j, index_t& k) const
	{
		const index_t	hi = numRows % 2,
				hj = numColumns % 2;

		if(i<(numRows+hi)/2) 	i = i + (numRows-hi)/2;
		else			i = i - (numRows+hi)/2;

		if(j<(numColumns+hj)/2) j = j + (numColumns-hj)/2;
		else 			j = j - (numColumns+hj)/2;

		return getIndex(i, j, k);
	}

	__host__ __device__ inline index_t Layout::getIndexFFTInverseShift(index_t i, index_t j, index_t k) const
	{
		return getIndicesFFTInverseShift(i, j, k);
	}

	__host__ __device__ inline index_t Layout::getIndexClampedToEdge(index_t i, index_t j, index_t k) const
	{
		return getIndex(getIClamped(i), getJClamped(j), getKClamped(k));
	}

	__host__ __device__ inline index_t Layout::getIndexWarped(index_t i, index_t j, index_t k) const
	{
		return getIndex(getIWrapped(i), getJWrapped(j), getKWrapped(k));
	}

	__device__ inline index_t Layout::getIndexFFTShift(void) const
	{
		return getIndexFFTShift(getI(), getJ(), getK());
	}

	__device__ inline index_t Layout::getIndexFFTInverseShift(void) const
	{
		return getIndexFFTInverseShift(getI(), getJ(), getK());
	}

	__host__ __device__ inline bool Layout::inside(index_t i, index_t j, index_t k) const
	{
		return (i>=0 && i<numRows && j>=0 && j<numColumns && k>=0 && k<numSlices);
	}

	__device__ inline bool Layout::inside(void) const
	{
		return  inside(getI(), getJ(), getK());
	}

	__host__ __device__ inline bool Layout::validRowIndex(index_t i) const	
	{
		return (i>=0 && i<numRows);
	}

	__host__ __device__ inline bool Layout::validColumnIndex(index_t j) const
	{
		return (j>=0 && j<numColumns);
	}

	__host__ __device__ inline bool Layout::validSliceIndex(index_t k) const
	{
		return (k>=0 && k<numSlices);
	}

	__host__ __device__ inline void Layout::unpackIndex(index_t index, index_t& i, index_t& j, index_t& k)
	{
		index_t tmp = index/numRows;
		j = index - tmp*numRows;
		k = tmp/numColumns;
		i = tmp - k*numColumns;
	}

	__host__ inline dim3 Layout::getBlockSize(void) const
	{
		dim3 d;
		d.x = 1;
		d.y = min(static_cast<index_t>(numThreads), getNumRows());
		d.z = 1;
		return d;
	}

	__host__ inline dim3 Layout::getNumBlock(void) const
	{
		dim3 d;
		d.x = getNumColumns(); 
		d.y = ceil(static_cast<double>(getNumRows())/static_cast<double>(numThreads));
		d.z = getNumSlices();
		return d;
	}

// Accessor :
	template<typename T>
	__host__ __device__ Accessor<T>::Accessor(index_t r, index_t c, index_t s, index_t lc, index_t ls)
	 : 	Layout(r, c, s, lc, ls),
		gpu_ptr(NULL)	
	{ }

	template<typename T>
	__host__ __device__ Accessor<T>::Accessor(const Layout& layout)
	 : 	Layout(layout), 
		gpu_ptr(NULL)
	{ }

	template<typename T>
	__host__ __device__ Accessor<T>::Accessor(T* ptr, index_t r, index_t c, index_t s, index_t lc, index_t ls)
	 :	Layout(r, c, s, lc, ls),
		gpu_ptr(ptr)	
	{ }
	
	template<typename T>
	__host__ Accessor<T>::Accessor(const Array<T>& a)
	 :	Layout(a), 
		gpu_ptr(a.getPtr())
	{ }

	template<typename T>
	__host__ __device__ Accessor<T>::Accessor(const Accessor<T>& a)
	 : 	Layout(a),
		gpu_ptr(a.gpu_ptr)
	{ }

	template<typename T>
	__host__ __device__ inline T* Accessor<T>::getPtr(void) const
	{
		return gpu_ptr;
	}

	template<typename T>
	__host__ __device__ inline size_t Accessor<T>::getSize(void) const
	{
		return static_cast<size_t>(getNumElements())*sizeof(T);
	}	

	template<typename T>
	__device__ inline T& Accessor<T>::data(index_t i, index_t j, index_t k) const
	{
		return gpu_ptr[getIndex(i, j, k)];
	}

	template<typename T>
	__device__ inline T& Accessor<T>::data(index_t p) const
	{
		return gpu_ptr[p];
	}

	template<typename T>
	__device__ inline T& Accessor<T>::data(void) const
	{
		return gpu_ptr[getIndex()];
	}

	template<typename T>
	__device__ inline T& Accessor<T>::dataInSlice(int k) const
	{
		return gpu_ptr[getIndex(getI(),getJ(),k)];
	}

	template<typename T>
	__device__ inline T& Accessor<T>::dataFFTShift(void) const
	{
		return gpu_ptr[getIndexFFTShift()];
	}

	template<typename T>
	__device__ inline T& Accessor<T>::dataFFTInverseShift(void) const
	{
		return gpu_ptr[getIndexFFTInverseShift()];
	}

	template<typename T>
	T* Accessor<T>::getData(void) const
	{	
		T* ptr = new T[getNumElements()];
		getData(ptr);
		return ptr;
	}

	template<typename T>
	void Accessor<T>::getData(T* ptr) const
	{
		if(ptr==NULL)
			throw NullPointer;

		cudaDeviceSynchronize();

		if(getNumRows()==getLeadingColumns() && getNumElementsPerSlice()==getLeadingSlices())
		{
			cudaError_t err = cudaMemcpy(ptr, gpu_ptr, getNumElements()*sizeof(T), cudaMemcpyDeviceToHost);
			if(err!=cudaSuccess)
				throw static_cast<Exception>(err);
		}		
		else if(getNumRows()==getLeadingColumns())
		{
			for(index_t k=0; k<getNumSlices(); k++)
			{
				cudaError_t err = cudaMemcpy((ptr + k*getNumElementsPerSlice()), (gpu_ptr + k*getLeadingSlices()), getNumElementsPerSlice()*sizeof(T), cudaMemcpyDeviceToHost);
				if(err!=cudaSuccess)
					throw static_cast<Exception>(err);
			}
		}
		else
		{
			for(index_t k=0; k<getNumSlices(); k++)
			{
				for(index_t j=0; j<getNumColumns(); j++)
				{
					cudaError_t err = cudaMemcpy((ptr + k*getNumElementsPerSlice() + j*getNumRows()), (gpu_ptr + k*getLeadingSlices() + j*getLeadingColumns()), getNumRows()*sizeof(T), cudaMemcpyDeviceToHost);
					if(err!=cudaSuccess)
						throw static_cast<Exception>(err);
				}
			}
		}
	}

	template<typename T>
	void Accessor<T>::setData(const T* ptr)
	{
		if(ptr==NULL)
			throw NullPointer;

		cudaDeviceSynchronize();

		if(getNumRows()==getLeadingColumns() && getNumElementsPerSlice()==getLeadingSlices())
		{
			cudaError_t err = cudaMemcpy(gpu_ptr, ptr, getNumElements()*sizeof(T), cudaMemcpyHostToDevice);
			if(err!=cudaSuccess)
				throw static_cast<Exception>(err);
		}
		else if(getNumRows()==getLeadingColumns())
		{
			for(index_t k=0; k<getNumSlices(); k++)
			{
				cudaError_t err = cudaMemcpy((gpu_ptr + k*getLeadingSlices()), (ptr + k*getNumElementsPerSlice()), getNumElementsPerSlice()*sizeof(T), cudaMemcpyHostToDevice);
				if(err!=cudaSuccess)
					throw static_cast<Exception>(err);
			}
		}
		else
		{
			for(index_t k=0; k<getNumSlices(); k++)
			{
				for(index_t j=0; j<getNumColumns(); j++)
				{
					cudaError_t err = cudaMemcpy((gpu_ptr + k*getLeadingSlices() + j*getLeadingColumns()), (ptr + k*getNumElementsPerSlice() + j*getNumRows()), getNumRows()*sizeof(T), cudaMemcpyHostToDevice);
					if(err!=cudaSuccess)
						throw static_cast<Exception>(err);
				}
			}
		}
	}

	template<typename T>
	__host__ inline Accessor<T> Accessor<T>::value(index_t i, index_t j, index_t k) const
	{
		if(!inside(i, j, k))
			throw OutOfRange;		

		return Accessor<T>(gpu_ptr + getIndex(i, j, k), 1, 1, 1);
	}

	template<typename T>
	__host__ inline Accessor<T> Accessor<T>::vector(index_t j, index_t k) const
	{
		if(!validColumnIndex(j) || !validSliceIndex(k))
			throw OutOfRange;

		return Accessor<T>(gpu_ptr + getIndex(0,j,k), getNumRows());
	}

	template<typename T>
	__host__ inline Accessor<T> Accessor<T>::endVector(index_t k) const
	{
		if(!validSliceIndex(k))
			throw OutOfRange;
	
		return Accessor<T>(gpu_ptr + getIndex(0,getNumRows()-1,k), getNumRows());
	}

	template<typename T>
	__host__ inline Accessor<T> Accessor<T>::vectors(index_t jBegin, index_t jEnd, index_t k, index_t jStep) const
	{
		if(jStep<0 || jBegin>=jEnd)
			throw InvalidNegativeStep;
		if(!validColumnIndex(jBegin) || !validColumnIndex(jEnd) || !validSliceIndex(k))
			throw OutOfRange;

		return Accessor<T>(gpu_ptr + getIndex(0,jBegin,k), getNumRows(), (jBegin - jEnd + 1)/jStep, 1, jStep*getLeadingColumns());
	}

	template<typename T>
	__host__ inline Accessor<T> Accessor<T>::slice(index_t k) const
	{
		if(!validSliceIndex(k))
			throw OutOfRange;

		return Accessor<T>(gpu_ptr + getIndex(0,0,k), getNumRows(), getNumColumns(), 1, getLeadingColumns());
	}

	template<typename T>
	__host__ inline Accessor<T> Accessor<T>::endSlice(void) const
	{
		return Accessor<T>(gpu_ptr + getIndex(0,0,getNumSlices()-1), getNumColumns(), 1, getLeadingColumns());
	}

	template<typename T>
	__host__ inline Accessor<T> Accessor<T>::slices(index_t kBegin, index_t kEnd, index_t kStep) const
	{
		if(kStep<0 || kBegin>=kEnd)
			throw InvalidNegativeStep;
		if(!validSliceIndex(kBegin) || !validSliceIndex(kEnd))
			throw OutOfRange;

		return Accessor<T>(gpu_ptr + getIndex(0,0,kBegin), getNumRows(), getNumColumns(), (kBegin - kEnd + 1)/kStep, getLeadingColumns(), kStep*getLeadingSlices());
	}

// Array :
	template<typename T>
	__host__ Array<T>::Array(index_t r, index_t c, index_t s)
	 :	Accessor<T>(r, c, s)
	{
		cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&this->gpu_ptr), getNumElements()*sizeof(T));
		if(err!=cudaSuccess)
			throw static_cast<Exception>(err);
	}
	
	template<typename T>
	__host__ Array<T>::Array(const Layout& layout)
	 :	Accessor<T>(layout)
	{
		cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&this->gpu_ptr), getNumElements()*sizeof(T));
		if(err!=cudaSuccess)
			throw static_cast<Exception>(err);
	}

	template<typename T>
	__host__ Array<T>::Array(const T* ptr, index_t r, index_t c, index_t s)
	 :	Accessor<T>(r, c, s)
	{
		cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&this->gpu_ptr), getNumElements()*sizeof(T));
		if(err!=cudaSuccess)
			throw static_cast<Exception>(err);
		setData(ptr);
	}

	template<typename T>
	__host__ Array<T>::~Array(void)
	{
		cudaDeviceSynchronize();	
		cudaError_t err = cudaFree(this->gpu_ptr);
		if(err!=cudaSuccess)
			throw static_cast<Exception>(err);
		this->gpu_ptr = NULL;
	}

	template<typename T>
	Accessor<T>& Array<T>::accessor(void)
	{
		return (*this);
	}

	/*template<typename T>
	Array<T>::Array(const Array<T>& a)
	 : Accessor<T>(NULL, a.getWidth(), a.getHeight(), a.getNumSlices())
	{
		#ifdef __DEBUG__
			std::cout << "Array<T>::Array - Allocation of array : " << getWidth() << 'x' << getHeight() << 'x' << getNumSlices() << " (copy constructor)" << std::endl;
		#endif 

		cudaMalloc((void**)&this->gpu_ptr, getNumElements()*sizeof(T));
		*this = a;
	}

	template<typename T>
	template<typename Tin>
	Array<T>::Array(const Accessor<Tin>& a)
	 : Accessor<T>(NULL, a.getWidth(), a.getHeight(), a.getNumSlices())
	{
		#ifdef __DEBUG__
			std::cout << "Array<T>::Array - Allocation of array : " << getWidth() << 'x' << getHeight() << 'x' << getNumSlices() << " (copy constructor)" << std::endl;
		#endif 

		cudaMalloc((void**)&this->gpu_ptr, getNumElements()*sizeof(T));
		*this = a;
	}*/


/*	template<typename T>
	Array<T>::Array(const std::string& filename)
	 : Accessor<T>()
	{
		load(filename);
		checkForCudaErrors();
	}
*/

} // Namespace Kartet

#endif 

