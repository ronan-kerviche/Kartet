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
	__host__ __device__ inline Layout::Layout(index_t r, index_t c, index_t s, index_t lc, index_t ls, index_t o)
	 : 	numRows(r),
		numColumns(c),
		numSlices(s),
		leadingColumns(lc),
		leadingSlices(ls),
		offset(o)
	{
		if(leadingColumns<r || c==1)
			leadingColumns = numRows;
		if(leadingSlices<(numColumns*leadingColumns) || numSlices==1)
			leadingSlices = numColumns*leadingColumns;
	}

	__host__ __device__ inline Layout::Layout(const Layout& l)
	 : 	numRows(l.numRows),
		numColumns(l.numColumns),
		numSlices(l.numSlices),
		leadingColumns(l.leadingColumns),
		leadingSlices(l.leadingSlices),
		offset(l.offset)
	{ }

	__host__ __device__ inline index_t Layout::getNumElements(void) const
	{
		return (numRows * numColumns * numSlices);
	}
	
	__host__ __device__ inline index_t Layout::getNumElementsPerSlice(void) const
	{
		return (numRows * numColumns);
	}

	__host__ __device__ inline index_t Layout::getNumElementsPerFragment(void) const
	{
		if(leadingColumns==numRows)
		{
			if(leadingSlices==(numRows * numColumns))
				return (numRows * numColumns * numSlices);
			else
				return (numRows * numColumns);
		}
		else
			return numRows;
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

	__host__ __device__ inline index_t Layout::getNumFragments(void) const
	{
		if(leadingColumns==numRows)
		{
			if(leadingSlices==(numRows * numColumns))
				return 1;
			else
				return numSlices;
		}
		else
			return (numSlices * numColumns);
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

	__host__ __device__ inline index_t Layout::getOffset(void) const
	{
		return offset;
	}

	__host__ __device__ inline index_t Layout::setOffset(index_t newOffset)
	{
		index_t oldOffset = offset;
		offset = newOffset;
		return oldOffset;
	}

	__host__ __device__ inline dim3 Layout::getDimensions(void) const
	{
		return dim3(numRows, numColumns, numSlices);
	}
	
	__host__ __device__ inline dim3 Layout::getStride(void) const
	{
		return dim3(1, leadingColumns, leadingSlices);
	}

	__host__ __device__ inline bool Layout::isMonolithic(void) const
	{
		return (leadingColumns==numRows || numColumns==1) && (leadingSlices==(numRows*numColumns) || numSlices==1);
	}

	__host__ __device__ inline bool Layout::isSliceMonolithic(void) const
	{
		return (leadingColumns==numRows || numColumns==1);
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

	__host__ inline std::vector<Layout> Layout::splitLayoutPages(index_t jBegin, index_t numVectors) const
	{
		std::vector<Layout> pages;

		if(!validColumnIndex(jBegin) || numVectors<1)
			throw OutOfRange;
	
		for(index_t j=jBegin; j<getNumColumns(); j+=numVectors)
			pages.push_back(Layout(getNumRows(), min(numVectors, getNumColumns()-j), getNumSlices(), getLeadingColumns(), getLeadingSlices(), getOffset()+getIndex(0,j,0)));

		return pages;
	}

	__host__ __device__ inline bool Layout::sameLayoutAs(const Layout& other) const
	{
		return (numRows==other.numRows && numColumns==other.numColumns && numSlices==other.numSlices && leadingColumns==other.leadingColumns && leadingSlices==other.leadingSlices); 
	}

	__host__ __device__ inline bool Layout::sameSliceLayoutAs(const Layout& other) const
	{
		return (numRows==other.numRows && numColumns==other.numColumns && leadingColumns==other.leadingColumns); 
	}

	__device__ inline index_t Layout::getI(void)
	{
		//return blockIdx.y*blockDim.y+threadIdx.y;
		return blockIdx.x*blockDim.x+threadIdx.x;
	}

	__device__ inline index_t Layout::getJ(void)
	{
		//return blockIdx.x*blockDim.x+threadIdx.x;
		return blockIdx.y*blockDim.y+threadIdx.y;
	}

	__device__ inline index_t Layout::getK(void)
	{
		return blockIdx.z*blockDim.z+threadIdx.z;
	}

	template<typename TOut>
	__host__ __device__ inline TOut Layout::getINorm(index_t i) const
	{
		return static_cast<TOut>(i)/static_cast<index_t>(numRows);
	}

	template<typename TOut>
	__host__ __device__ inline TOut Layout::getJNorm(index_t j) const
	{
		return static_cast<TOut>(j)/static_cast<index_t>(numColumns);
	}

	template<typename TOut>
	__host__ __device__ inline TOut Layout::getKNorm(index_t k) const
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
	__host__ __device__ inline TOut Layout::getINormIncl(index_t i) const
	{
		return static_cast<TOut>(i)/static_cast<TOut>(numRows-1);
	}

	template<typename TOut>
	__host__ __device__ inline TOut Layout::getJNormIncl(index_t j) const
	{
		return static_cast<TOut>(j)/static_cast<TOut>(numColumns-1);
	}

	template<typename TOut>
	__host__ __device__ inline TOut Layout::getKNormIncl(index_t k) const
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

	__host__ __device__ inline bool Layout::isInside(index_t i, index_t j, index_t k) const
	{
		return (i>=0 && i<numRows && j>=0 && j<numColumns && k>=0 && k<numSlices);
	}

	__device__ inline bool Layout::isInside(void) const
	{
		return  isInside(getI(), getJ(), getK());
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

	__host__ __device__ inline void Layout::unpackIndex(index_t index, index_t& i, index_t& j, index_t& k) const
	{
		k = index / leadingSlices;
		j = (index - k*leadingSlices) / leadingColumns;
		i = index - k*leadingSlices - j*leadingColumns;
	}

	__host__ inline dim3 Layout::getBlockSize(void) const
	{
		dim3 d;
		/*d.x = 1;
		d.y = min(StaticContainer<void>::numThreads, numRows);
		d.z = 1;*/
		// From inner most dimension (I <-> X, J <-> Y, K <-> Z) :
		d.x = min(StaticContainer<void>::numThreads, numRows);
		d.y = min(StaticContainer<void>::numThreads/d.x, numColumns);
		d.z = min(StaticContainer<void>::numThreads/(d.x*d.y), numSlices);
		//std::cout << "Layout::getBlockSize : " << d.x << ", " << d.y << ", " << d.z << std::endl;
		return d;
	}

	__host__ inline dim3 Layout::getNumBlock(void) const
	{
		dim3 d;
		/*d.x = numColumns; 
		d.y = (numRows + StaticContainer<void>::numThreads - 1)/StaticContainer<void>::numThreads;
		d.z = numSlices;*/
		// From inner most dimension (I <-> X, J <-> Y, K <-> Z) :
		const dim3 blockSize = getBlockSize();
		d.x = (numRows + blockSize.x - 1)/blockSize.x;
		d.y = (numColumns + blockSize.y - 1)/blockSize.y;
		d.z = (numSlices + blockSize.z - 1)/blockSize.z;
		//std::cout << "Layout::getNumBlock : " << d.x << ", " << d.y << ", " << d.z << std::endl;
		return d;
	}

	__host__ inline Layout Layout::getVectorLayout(void) const
	{
		return Layout(getNumRows());
	}

	__host__ inline Layout Layout::getSliceLayout(void) const
	{
		return Layout(getNumRows(), getNumColumns(), 1, getLeadingColumns());
	}

	__host__ inline Layout Layout::getMonolithicLayout(void) const
	{
		return Layout(getNumRows(), getNumColumns(), getNumSlices());
	}

	template<class Op, typename T>
	__host__ void Layout::hostScan(T* ptr, const Op& op) const
	{
		// Op should have a function :
		// void apply(const Layout& mainLayout, const Layout& currentAccessLayout, T* ptr, size_t offset, int i, int j, int k) const;
		if(getNumRows()==getLeadingColumns() && getNumElementsPerSlice()==getLeadingSlices())
			op.apply(*this, *this, ptr, 0, 0, 0, 0);		
		else if(getNumRows()==getLeadingColumns())
		{
			const Layout sliceLayout = getSliceLayout();
			for(index_t k=0; k<getNumSlices(); k++)
				op.apply(*this, sliceLayout, ptr, k*getLeadingSlices(), 0, 0, k);	
		}
		else
		{
			const Layout vectorLayout = getVectorLayout();
			for(index_t k=0; k<getNumSlices(); k++)
			{
				for(index_t j=0; j<getNumColumns(); j++)
					op.apply(*this, vectorLayout, ptr, (k*getLeadingSlices() + j*getLeadingColumns()), 0, j, k);
			}
		}
	}

	__host__ inline Layout Layout::readFromFile(std::fstream& file, int* typeIndex)
	{
		if(!file.is_open())
			throw InvalidFileStream;
		
		char headerBuffer[sizeof(StaticContainer<void>::fileHeader)];
		std::memset(headerBuffer, 0, sizeof(headerBuffer));
		file.read(headerBuffer, sizeof(headerBuffer)-1);
		if(strncmp(StaticContainer<void>::fileHeader, headerBuffer, sizeof(headerBuffer)-1)!=0)
			throw InvalidFileHeader;
	
		// Read the type :
		int dummyType;
		if(typeIndex==NULL)
			typeIndex = &dummyType;
		file.read(reinterpret_cast<char*>(typeIndex), sizeof(int)); if(!file.good()) throw InvalidFileStream;

		// Read the sizes :
		index_t	r = 0,
			c = 0,
			s = 0;
		file.read(reinterpret_cast<char*>(&r), sizeof(index_t)); if(!file.good()) throw InvalidFileStream;
		file.read(reinterpret_cast<char*>(&c), sizeof(index_t)); if(!file.good()) throw InvalidFileStream;
		file.read(reinterpret_cast<char*>(&s), sizeof(index_t)); if(!file.good()) throw InvalidFileStream;

		// Return :
		return Layout(r, c, s);
	}

	__host__ inline Layout Layout::readFromFile(const std::string& filename, int* typeIndex)
	{
		std::fstream file(filename.c_str(), std::fstream::in | std::fstream::binary);

		if(!file.is_open())
		{
			file.close();
			throw InvalidFileStream;
		}

		Layout layout = readFromFile(file, typeIndex);

		file.close();

		return layout;
	}

	__host__ inline void Layout::writeToFile(std::fstream& file, int typeIndex)
	{
		if(!file.is_open())
			throw InvalidFileStream;

		// Write the header :
		file.write(StaticContainer<void>::fileHeader, sizeof(StaticContainer<void>::fileHeader)-1);
		
		// Write the type :	
		file.write(reinterpret_cast<char*>(&typeIndex), sizeof(int));
	
		// Write the size :
		file.write(reinterpret_cast<char*>(&numRows), sizeof(index_t));
		file.write(reinterpret_cast<char*>(&numColumns), sizeof(index_t));
		file.write(reinterpret_cast<char*>(&numSlices), sizeof(index_t));	
	}

	__host__ inline void Layout::writeToFile(const std::string& filename, int typeIndex)
	{
		std::fstream file(filename.c_str(), std::fstream::out | std::fstream::binary);

		if(!file.is_open())
		{
			file.close();
			throw InvalidFileStream;
		}

		writeToFile(file, typeIndex);

		file.close();
	}

	template<typename T>
	__host__ inline void Layout::writeToFile(std::fstream& file)
	{
		writeToFile(file, GetIndex<TypesSortedByAccuracy, T>::value);
	}

	template<typename T>
	__host__ inline void Layout::writeToFile(const std::string& filename)
	{
		writeToFile(filename, GetIndex<TypesSortedByAccuracy, T>::value);
	}

	__host__ inline std::ostream& operator<<(std::ostream& os, const Layout& layout)
	{
		if(layout.getNumSlices()==1)
			os << '[' << layout.getNumRows() << ", " << layout.getNumColumns();
		else
			os << '[' << layout.getNumRows() << ", " << layout.getNumColumns() << ", " << layout.getNumSlices();

		if(layout.getLeadingColumns()>layout.getNumRows() || layout.getLeadingSlices()>layout.getNumElementsPerSlice())
		{
			os << "; ";
			if(layout.getLeadingColumns()==layout.getNumRows())
				os << '_';
			else
				os << '+' << layout.getLeadingColumns();
			os << ", ";
			if(layout.getLeadingSlices()==layout.getNumElementsPerSlice())
				os << '_';
			else
				os << '+' << layout.getLeadingSlices();
			os << ", ";
			if(layout.getOffset()==0)
				os << '_';
			else
				os << '+' << layout.getOffset();
		}
		os << ']';

		return os;
	}

// Accessor :
	template<typename T>
	__host__ __device__ Accessor<T>::Accessor(index_t r, index_t c, index_t s, index_t lc, index_t ls, index_t o)
	 : 	Layout(r, c, s, lc, ls, o),
		devicePtr(NULL)	
	{ }

	template<typename T>
	__host__ __device__ Accessor<T>::Accessor(const Layout& layout)
	 : 	Layout(layout), 
		devicePtr(NULL)
	{ }

	template<typename T>
	__host__ __device__ Accessor<T>::Accessor(T* ptr, index_t r, index_t c, index_t s, index_t lc, index_t ls, index_t o)
	 :	Layout(r, c, s, lc, ls, o),
		devicePtr(ptr)
	{ }

	template<typename T>
	__host__ __device__ Accessor<T>::Accessor(T* ptr, const Layout& layout)
	 :	Layout(layout),
		devicePtr(ptr)
	{ }
	
	template<typename T>
	__host__ Accessor<T>::Accessor(const Array<T>& a)
	 :	Layout(a), 
		devicePtr(a.devicePtr)
	{ }

	template<typename T>
	__host__ __device__ Accessor<T>::Accessor(const Accessor<T>& a)
	 : 	Layout(a),
		devicePtr(a.devicePtr)
	{ }

	template<typename T>
	__host__ __device__ inline T* Accessor<T>::getPtr(void) const
	{
		return devicePtr;
	}

	template<typename T>
	__host__ __device__ inline size_t Accessor<T>::getSize(void) const
	{
		return static_cast<size_t>(getNumElements())*sizeof(T);
	}	

	template<typename T>
	__device__ inline T& Accessor<T>::data(index_t i, index_t j, index_t k) const
	{
		return devicePtr[getIndex(i, j, k)];
	}

	template<typename T>
	__device__ inline T& Accessor<T>::data(index_t p) const
	{
		return devicePtr[p];
	}

	template<typename T>
	__device__ inline T& Accessor<T>::data(void) const
	{
		return devicePtr[getIndex()];
	}

	template<typename T>
	__device__ inline T& Accessor<T>::dataInSlice(int k) const
	{
		return devicePtr[getIndex(getI(),getJ(),k)];
	}

	template<typename T>
	__device__ inline T& Accessor<T>::dataFFTShift(void) const
	{
		return devicePtr[getIndexFFTShift()];
	}

	template<typename T>
	__device__ inline T& Accessor<T>::dataFFTInverseShift(void) const
	{
		return devicePtr[getIndexFFTInverseShift()];
	}

	template<typename T>
	T* Accessor<T>::getData(void) const
	{	
		T* ptr = new T[getNumElements()];
		getData(ptr);
		return ptr;
	}

	// Tools for the memcpy :
		template<typename T>
		struct MemCpyToolBox
		{			
			const cudaMemcpyKind 	kind;
			const T			*from;
			T			*to;
			const Layout		deviceLayout,
						solidLayout;

			__host__ MemCpyToolBox(const cudaMemcpyKind _k, T* _to, const T* _from, const Layout& _deviceLayout)
			 :	kind(_k), 
				from(_from),
				to(_to),
				deviceLayout(_deviceLayout),
				solidLayout(_deviceLayout.getMonolithicLayout())
			{ }
			
			__host__ void apply(const Layout& mainLayout, const Layout& currentAccessLayout, T* ptr, size_t offset, int i, int j, int k) const
			{
				size_t	toOffset = 0,
					fromOffset = 0;
				if(kind==cudaMemcpyDeviceToHost)
				{
					toOffset	= solidLayout.getIndex(i, j, k);
					fromOffset	= deviceLayout.getIndex(i, j, k);
				}
				else
				{
					toOffset	= deviceLayout.getIndex(i, j, k);
					fromOffset	= solidLayout.getIndex(i, j, k);
				}
				cudaError_t err = cudaMemcpy((to + toOffset), (from + fromOffset), currentAccessLayout.getNumElements()*sizeof(T), kind);
				if(err!=cudaSuccess)
					throw static_cast<Exception>(CudaExceptionsOffset + err);
			}
		};

	template<typename T>
	void Accessor<T>::getData(T* ptr) const
	{
		if(ptr==NULL)
			throw NullPointer;

		cudaDeviceSynchronize();
		MemCpyToolBox<T> toolbox(cudaMemcpyDeviceToHost, ptr, devicePtr, *this);
		hostScan(toolbox);
	}

	template<typename T>
	void Accessor<T>::setData(const T* ptr) const
	{
		if(ptr==NULL)
			throw NullPointer;

		cudaDeviceSynchronize();
		MemCpyToolBox<T> toolbox(cudaMemcpyHostToDevice, devicePtr, ptr, *this);
		hostScan(toolbox);
	}

	template<typename T>
	__host__ const Layout& Accessor<T>::getLayout(void) const
	{
		return (*this);
	}

	template<typename T>
	__host__ Accessor<T> Accessor<T>::element(index_t i, index_t j, index_t k) const
	{
		if(!isInside(i, j, k))
			throw OutOfRange;		

		return Accessor<T>(devicePtr + getIndex(i, j, k), 1, 1, 1, 1, 1, getOffset()+getIndex(i, j, k));
	}

	template<typename T>
	__host__ Accessor<T> Accessor<T>::elements(index_t p, index_t numElements) const
	{
		if(p<0 || numElements<0 || p>=getNumElements() || (p+numElements)>=getNumElements())
			throw OutOfRange;

		// Test if the block is fully in this accessor :
		if(!isMonolithic())
		{
			index_t i = 0,
				j = 0,
				k = 0;
			unpackIndex(p, i, j, k);
			if(getNumRows()!=getLeadingColumns())
			{
				if(numElements>getNumRows())
					throw OutOfRange;
			}
			else if(getNumElementsPerSlice()!=getLeadingColumns())
			{
				if(numElements>getNumElementsPerSlice())
					throw OutOfRange;
			}
		}
		return Accessor<T>(devicePtr + p, numElements, 1, 1, numElements, numElements, getOffset()+p);
	}

	template<typename T>
	__host__ Accessor<T> Accessor<T>::vector(index_t j) const
	{
		if(!validColumnIndex(j))
			throw OutOfRange;

		return Accessor<T>(devicePtr + getIndex(0,j,0), getNumRows(), 1, getNumSlices(), getNumRows(), getLeadingSlices(), getOffset()+getIndex(0,j,0));
	}

	template<typename T>
	__host__ Accessor<T> Accessor<T>::endVector(void) const
	{
		return vector(getNumColumns()-1);
	}

	template<typename T>
	__host__ Accessor<T> Accessor<T>::vectors(index_t jBegin, index_t numVectors, index_t jStep) const
	{
		if(jStep<=0 || numVectors<=0)
			throw InvalidNegativeStep;
		if(!validColumnIndex(jBegin) || !validColumnIndex(jBegin+(numVectors-1)*jStep))
			throw OutOfRange;

		return Accessor<T>(devicePtr + getIndex(0,jBegin,0), getNumRows(), numVectors, getNumSlices(), jStep*getLeadingColumns(), getLeadingSlices(), getOffset()+getIndex(0,jBegin,0));
	}

	template<typename T>
	__host__ Accessor<T> Accessor<T>::slice(index_t k) const
	{
		if(!validSliceIndex(k))
			throw OutOfRange;

		return Accessor<T>(devicePtr + getIndex(0,0,k), getNumRows(), getNumColumns(), 1, getLeadingColumns(), getOffset()+getIndex(0,0,k));
	}

	template<typename T>
	__host__ Accessor<T> Accessor<T>::endSlice(void) const
	{
		return slice(getNumSlices()-1);
	}

	template<typename T>
	__host__ Accessor<T> Accessor<T>::slices(index_t kBegin, index_t numSlices, index_t kStep) const
	{
		if(kStep<0 || numSlices<=0)
			throw InvalidNegativeStep;
		if(!validSliceIndex(kBegin) || !validSliceIndex(kBegin+(numSlices-1)*kStep-1))
			throw OutOfRange;

		return Accessor<T>(devicePtr + getIndex(0,0,kBegin), getNumRows(), getNumColumns(), numSlices, getLeadingColumns(), kStep*getLeadingSlices(), getOffset()+getIndex(0,0,kBegin));
	}

	template<typename T>
	__host__ Accessor<T> Accessor<T>::subArray(index_t iBegin, index_t jBegin, index_t numRows, index_t numColumns) const
	{
		if(numRows<=0 || numColumns<=0)
			throw InvalidNegativeStep;
		if(!validRowIndex(iBegin) || !validRowIndex(iBegin+numRows-1) || !validColumnIndex(jBegin) || !validColumnIndex(jBegin+numColumns-1))
			throw OutOfRange;
		
		return Accessor<T>(devicePtr + getIndex(iBegin,jBegin,0), numRows, numColumns, getNumSlices(), getLeadingColumns(), getLeadingSlices(), getOffset()+getIndex(iBegin,jBegin,0));
	}

	template<typename T>
	__host__  std::vector< Accessor<T> > Accessor<T>::splitPages(index_t jBegin, index_t numVectors) const
	{
		std::vector< Accessor<T> > pages;
		Layout tmp = this->getLayout();
		tmp.setOffset(0);
		const std::vector<Layout> pagesLayout = tmp.splitLayoutPages(jBegin, numVectors);

		for(std::vector<Layout>::const_iterator it=pagesLayout.begin(); it!=pagesLayout.end(); it++)	
		{
			pages.push_back( Accessor<T>(devicePtr + it->getOffset(), *it) );
			pages.back().setOffset(getOffset() + pages.back().getOffset());
		}

		return pages;
	}

	template<typename T>
	template<class Op>
	__host__ void Accessor<T>::hostScan(const Op& op) const
	{
		Layout::hostScan(devicePtr, op);
	}

	template<typename T>
	__host__ std::ostream& operator<<(std::ostream& os, const Accessor<T>& A)
	{
		#define FMT std::right << std::setfill(fillCharacter) << std::setw(maxWidth)
		const int precision = 4,
			  maxWidth = precision+3;
		const char fillCharacter = ' ';
		const char* spacing = "  ";
		const Kartet::Layout l = A.getMonolithicLayout();
		T* tmp = A.getData();
		
		// Get old parameter :
		const int oldPrecision = os.precision(precision);

		os << "(Array of size " << A.getLayout() << ')' << std::endl;
		for(int k=0; k<l.getNumSlices(); k++)
		{
			if(l.getNumSlices()>1)
				os << "Slice " << k << " : "<< std::endl;
	
			for(int i=0; i<l.getNumRows(); i++)
			{
				for(int j=0; j<(l.getNumColumns()-1); j++)
					os << FMT << tmp[l.getIndex(i,j,k)] << spacing;
				os << FMT << tmp[l.getIndex(i,l.getNumColumns()-1,k)] << std::endl;
			}
		}
		#undef FMT

		os.precision(oldPrecision);

		delete[] tmp;
		return os;
	}

// Array :
	template<typename T>
	__host__ Array<T>::Array(index_t r, index_t c, index_t s)
	 :	Accessor<T>(r, c, s)
	{
		cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&this->devicePtr), getNumElements()*sizeof(T));
		if(err!=cudaSuccess)
			throw static_cast<Exception>(CudaExceptionsOffset + err);
	}
	
	template<typename T>
	__host__ Array<T>::Array(const Layout& layout)
	 :	Accessor<T>(layout.getMonolithicLayout())
	{
		cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&this->devicePtr), getNumElements()*sizeof(T));
		if(err!=cudaSuccess)
			throw static_cast<Exception>(CudaExceptionsOffset + err);
	}

	template<typename T>
	__host__ Array<T>::Array(const T* ptr, index_t r, index_t c, index_t s)
	 :	Accessor<T>(r, c, s)
	{
		cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&this->devicePtr), getNumElements()*sizeof(T));
		if(err!=cudaSuccess)
			throw static_cast<Exception>(CudaExceptionsOffset + err);
		setData(ptr);
	}

	template<typename T>
	__host__ Array<T>::Array(const T* ptr, const Layout& layout)
	 :	Accessor<T>(layout.getMonolithicLayout())
	{
		cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&this->devicePtr), getNumElements()*sizeof(T));
		if(err!=cudaSuccess)
			throw static_cast<Exception>(CudaExceptionsOffset + err);
		setData(ptr);
	}

	template<typename T>
	__host__ Array<T>::Array(const Array<T>& A)
	 :	Accessor<T>(A.getMonolithicLayout())
	{
		cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&this->devicePtr), getNumElements()*sizeof(T));
		if(err!=cudaSuccess)
			throw static_cast<Exception>(CudaExceptionsOffset + err);
		(*this) = A;
	}

	template<typename T>
	template<typename TIn>
	__host__ Array<T>::Array(const Accessor<TIn>& A)
	 : 	Accessor<T>(A.getMonolithicLayout())
	{
		cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&this->devicePtr), getNumElements()*sizeof(T));
		if(err!=cudaSuccess)
			throw static_cast<Exception>(CudaExceptionsOffset + err);
		(*this) = A;
	}
	
	template<typename T>
	__host__ Array<T>::Array(const std::string& filename, bool convert, size_t maxBufferSize)
	 :	Accessor<T>(Layout::readFromFile(filename))
	{
		cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&this->devicePtr), getNumElements()*sizeof(T));
		if(err!=cudaSuccess)
			throw static_cast<Exception>(CudaExceptionsOffset + err);
		readFromFile(filename, convert, maxBufferSize);
	}

	template<typename T>
	__host__ Array<T>::~Array(void)
	{
		cudaDeviceSynchronize();	
		cudaError_t err = cudaFree(this->devicePtr);
		if(err!=cudaSuccess)
			throw static_cast<Exception>(CudaExceptionsOffset + err);
		this->devicePtr = NULL;
	}

	template<typename T>
	__host__ Accessor<T>& Array<T>::accessor(void)
	{
		return (*this);
	}

	template<typename T>
	__host__ void Array<T>::readFromFile(std::fstream& file, bool convert, size_t maxBufferSize)
	{
		if(!isMonolithic())
			throw InvalidOperation;

		if(!file.is_open())
			throw InvalidFileStream;

		if(maxBufferSize==0)
			maxBufferSize = static_cast<size_t>(getNumElements())*sizeof(T);

		int typeIndex = -1;
		const Layout layout = Layout::readFromFile(file, &typeIndex);

		if(!layout.sameLayoutAs(*this))
			throw InvalidOperation;

		if(!convert && (typeIndex!=GetIndex<TypesSortedByAccuracy, T>::value))
			throw InvalidOperation;

		const size_t 	size = sizeOfType(typeIndex),
				numBufferElements = std::min(static_cast<size_t>(layout.getNumElements())*size, maxBufferSize)/size,
				numReads = static_cast<size_t>(static_cast<float>(layout.getNumElements())/static_cast<float>(numBufferElements) + 0.5f);
		index_t offset = 0;
		char	*bufferRead = new char[numBufferElements*size],
			*bufferCast = NULL;

		if(typeIndex!=GetIndex<TypesSortedByAccuracy, T>::value)
			bufferCast = new char[numBufferElements*sizeof(T)];

		try
		{
			for(size_t k=0; k<numReads; k++)
			{
				const size_t currentNumElements = std::min(numBufferElements, static_cast<size_t>(layout.getNumElements())-k*numBufferElements);
				file.read(bufferRead, currentNumElements*size);

				if(!file.good())
					throw InvalidFileStream;
			
				if(bufferCast==NULL)
				{
					cudaError_t err = cudaMemcpy(getPtr() + offset, reinterpret_cast<void*>(bufferRead), currentNumElements*sizeof(T), cudaMemcpyHostToDevice);
					if(err!=cudaSuccess)
						throw static_cast<Exception>(CudaExceptionsOffset + err);
				}
				else
				{
					copy(reinterpret_cast<T*>(bufferCast), bufferRead, typeIndex, currentNumElements);
					cudaError_t err = cudaMemcpy(getPtr() + offset, reinterpret_cast<void*>(bufferCast), currentNumElements*sizeof(T), cudaMemcpyHostToDevice);
					if(err!=cudaSuccess)
						throw static_cast<Exception>(CudaExceptionsOffset + err);
				}
				offset += currentNumElements;
			}
		}
		catch(Exception& e)
		{
			delete bufferRead;
			delete bufferCast;
			throw e;
		}
		delete bufferRead;
		delete bufferCast;
	}

	template<typename T>
	__host__ void Array<T>::readFromFile(const std::string& filename, bool convert, size_t maxBufferSize)
	{
		std::fstream file(filename.c_str(), std::fstream::in | std::fstream::binary);

		if(!file.is_open())
		{
			file.close();
			throw InvalidFileStream;
		}

		readFromFile(file, convert, maxBufferSize);

		file.close();
	}

	template<typename T>
	__host__ void Array<T>::writeToFile(std::fstream& file, size_t maxBufferSize)
	{
		if(!isMonolithic())
			throw InvalidOperation;

		if(!file.is_open())
			throw InvalidFileStream;

		if(maxBufferSize==0)
			maxBufferSize = static_cast<size_t>(getNumElements())*sizeof(T);

		// Write the header :
		Layout::writeToFile<T>(file);
		const size_t 	numBufferElements = std::min(static_cast<size_t>(getNumElements())*sizeof(T), maxBufferSize)/sizeof(T),
				numReads = static_cast<size_t>(static_cast<float>(getNumElements())/static_cast<float>(numBufferElements) + 0.5f);
		index_t offset = 0;
		char *buffer = new char[numBufferElements*sizeof(T)];

		// Write the data :
		try
		{
			for(size_t k=0; k<numReads; k++)
			{
				const size_t currentNumElements = std::min(numBufferElements, static_cast<size_t>(getNumElements())-k*numBufferElements);

				// Copy :
				cudaError_t err = cudaMemcpy(reinterpret_cast<void*>(buffer), getPtr() + offset, currentNumElements*sizeof(T), cudaMemcpyDeviceToHost);
				if(err!=cudaSuccess)
					throw static_cast<Exception>(CudaExceptionsOffset + err);

				// Write :
				file.write(buffer, currentNumElements*sizeof(T));

				offset += currentNumElements;
			}
		}
		catch(Exception& e)
		{
			delete buffer;
			throw e;
		}
		delete buffer;
	}

	template<typename T>
	__host__ void Array<T>::writeToFile(const std::string& filename, size_t maxBufferSize)
	{
		std::fstream file(filename.c_str(), std::fstream::out | std::fstream::binary);

		if(!file.is_open())
		{
			file.close();
			throw InvalidFileStream;
		}

		writeToFile(file, maxBufferSize);

		file.close();
	}

} // Namespace Kartet

#endif 

