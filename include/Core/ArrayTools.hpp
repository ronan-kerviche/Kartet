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

	#include <cmath>

namespace Kartet
{
// Tools :
	inline Direction getDirection(const Location& from, const Location& to)
	{
		if(from==AnySide || to==AnySide)
			throw InvalidDirection;
		else
			return 	(from==HostSide && to==HostSide) ?	HostToHost : 
				(from==HostSide && to==DeviceSide) ?	HostToDevice : 
				(from==DeviceSide && to==DeviceSide) ?	DeviceToDevice :
									DeviceToHost;
	}

	#ifdef __CUDACC__
		inline cudaMemcpyKind getCudaDirection(const Direction& direction)
		{
			return 	(direction==HostToHost) ?	cudaMemcpyHostToHost :
				(direction==HostToDevice) ?	cudaMemcpyHostToDevice :
				(direction==DeviceToHost) ?	cudaMemcpyDeviceToHost :
								cudaMemcpyDeviceToDevice;
		}
	#endif

// Layout::StaticContainer :
	template<typename T>
	inline size_t Layout::StaticContainer<T>::getStreamHeaderLength(void)
	{
		return sizeof(Layout::StaticContainer<T>::streamHeader);
	}

// Layout :
	/**
	\brief Layout constructor.
	\param r Number of rows.
	\param c Number of columns.
	\param s Number of slices.
	\param lc Distance between columns (LDA).
	\param ls Distance between slices.
	\param o Offset (not included in pointer calculations).
	**/
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

	/**
	\brief Layout constructor.
	\param l Original layout to be copied.
	**/
	__host__ __device__ inline Layout::Layout(const Layout& l)
	 : 	numRows(l.numRows),
		numColumns(l.numColumns),
		numSlices(l.numSlices),
		leadingColumns(l.leadingColumns),
		leadingSlices(l.leadingSlices),
		offset(l.offset)
	{ }

	/**
	\brief Returns the number of elements.
	\return The number of elements (number or rows x number of columns x number of slices).
	**/
	__host__ __device__ inline index_t Layout::getNumElements(void) const
	{
		return (numRows * numColumns * numSlices);
	}
	
	/**
	\brief Returns the number of elements per slice.
	\return The number of elements per slice (number or rows x number of columns).
	**/
	__host__ __device__ inline index_t Layout::getNumElementsPerSlice(void) const
	{
		return (numRows * numColumns);
	}

	/**
	\brief Returns the number of elements per smallest fragment.
	\return The number of elements in the smallest monoloithic fragment.
	**/
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

	/**
	\brief Returns the number of rows.
	\return The number of rows.
	**/
	__host__ __device__ inline index_t Layout::getNumRows(void) const
	{
		return numRows;
	}

	/**
	\brief Returns the number of columns.
	\return The number of columns.
	**/
	__host__ __device__ inline index_t Layout::getNumColumns(void) const
	{
		return numColumns;
	}

	/**
	\brief Returns the number of slices.
	\return The number of slices.
	**/
	__host__ __device__ inline index_t Layout::getNumSlices(void) const
	{
		return numSlices;
	}

	/**
	\brief Returns the number of monolithic fragments.
	\return The number of monolithic fragments.
	**/
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

	/**
	\brief Returns the number of columns.
	\return The number of columns.
	**/
	__host__ __device__ inline index_t Layout::getWidth(void) const
	{
		return numColumns;
	}

	/**
	\brief Returns the number of rows.
	\return The number of rows.
	**/
	__host__ __device__ inline index_t Layout::getHeight(void) const
	{
		return numRows;
	}

	/**
	\brief Returns the number of slices.
	\return The number of slices.
	**/
	__host__ __device__ inline index_t Layout::getDepth(void) const
	{
		return numSlices;
	}

	/**
	\brief Returns the distance between the beginnings of two consecutive columns.
	\return The distance between the beginnings of two consecutive columns.
	**/
	__host__ __device__ inline index_t Layout::getLeadingColumns(void) const
	{
		return leadingColumns;
	}

	/**
	\brief Returns the distance between the beginnings of two consecutive slices.
	\return The distance between the beginnings of two consecutive slices.
	**/
	__host__ __device__ inline index_t Layout::getLeadingSlices(void) const
	{
		return leadingSlices;
	}

	/**
	\brief Returns the offset of the data.
	\return The offset of the data.
	**/
	__host__ __device__ inline index_t Layout::getOffset(void) const
	{
		return offset;
	}

	/**
	\brief Modify the offset of the data.
	\param newOffset New offset value.

	This functions does not change in any way pointers to the data. It is purely an indication of the layout.
	
	\return The old offset.
	**/
	__host__ __device__ inline index_t Layout::setOffset(index_t newOffset)
	{
		index_t oldOffset = offset;
		offset = newOffset;
		return oldOffset;
	}

	#ifdef __CUDACC__
	/**
	\brief Returns the dimensions of the array in a dim3 struct.
	\return The dimensions of the array in a dim3 struct.
	**/
	__host__ __device__ inline dim3 Layout::getDimensions(void) const
	{
		return dim3(numRows, numColumns, numSlices);
	}
	
	/**
	\brief Returns the strides of the array in a dim3 struct.
	
	Note that the first stride is guaranteed to be 1.

	\return The strides of the array in a dim3 struct.
	**/
	__host__ __device__ inline dim3 Layout::getStride(void) const
	{
		return dim3(1, leadingColumns, leadingSlices);
	}
	#endif

	/**
	\brief Test if the layout is monolithic.
		
	This is true if and only if all of its elements are contiguous (equivalent to the number of fragments being equal to 1).

	\return True if the layout is monolithic.
	**/
	__host__ __device__ inline bool Layout::isMonolithic(void) const
	{
		return (leadingColumns==numRows || numColumns==1) && (leadingSlices==(numRows*numColumns) || numSlices==1);
	}

	/**
	\brief Test if all the slices in the layout are monolithic.
		
	This is true if and only if all of the elements inside the slices are contiguous.

	\return True if all the slices in the layout are monolithic.
	**/
	__host__ __device__ inline bool Layout::isSliceMonolithic(void) const
	{
		return (leadingColumns==numRows || numColumns==1);
	}

	/**
	\brief Modify the layout without changing the number of elements.
	\param r New row count.
	\param c New column count.
	\param s New slice count.

	This modification is only valid if the layout structure allow it (it is in agreement with the fragments layout).
	For instance :
	It is possible to reinterpret a layout from (2,6) to any of (12, 1), (1, 12), (3, 4), (4, 3), etc...
	It is not possible to reinterpret a layout from (2,6) to (11, 1) because they do not have the same number of elements.
	It is not possible to reinterpret a layout from (2,6, leading columns = 4) to (3,4) because the fragments are non contiguous.
	It is possible to reinterpret a layout from (2,6, leading columns = 4) to (2,1,6) because the fragments are kept.

	\throw InvalidLayoutChange In case of a failure.
	**/
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

	/**
	\brief Modify the layout without changing the number of elements.
	\param other New layout (only the number of rows, columns and slices are considered).

	\throw InvalidLayoutChange In case of a failure.
	**/
	__host__ inline void Layout::reinterpretLayout(const Layout& other)
	{
		reinterpretLayout(other.numRows, other.numColumns, other.numSlices);
	}

	/**
	\brief Reinterpret the layout by concatenating all slices columns in-place.

	The layout becomes of size (R, C*S, 1).

	\throw InvalidLayoutChange In case of a failure.
	**/
	__host__ inline void Layout::flatten(void)
	{
		reinterpretLayout(numRows, numColumns*numSlices, 1);
	}

	/**
	\brief Reinterpret the layout by concatenating all columns elements in-place.

	The layout becomes of size (R*C, S, 1).

	\throw InvalidLayoutChange In case of a failure.
	**/
	__host__ inline void Layout::stretch(void)
	{
		reinterpretLayout(numRows*numColumns, numSlices, 1);
	}

	/**
	\brief Reinterpret the layout by concatenating all elements in-place.
	
	The layout becomes of size (R*C*S, 1, 1).

	\throw InvalidLayoutChange In case of a failure.
	**/
	__host__ inline void Layout::vectorize(void)
	{
		reinterpretLayout(numRows*numColumns*numSlices, 1, 1);
	}

	/**
	\brief Generates a series of layout, each containing a specified number of columns of the original layout.
	\param jBegin Origin column index.
	\param nColumns Number of columns in each split.

	Note : the number of slices is preserved in the background of each split. The last split might contain less than nColumns columns (gracious).
	
	\throw OutOfRange If the index of the origin is invalid or nColumns is strictly less than 1.
	\return A vector containing all the splits (ordered).
	**/
	__host__ inline std::vector<Layout> Layout::splitLayoutColumns(index_t jBegin, index_t nColumns) const
	{
		std::vector<Layout> pages;

		if(!validColumnIndex(jBegin) || nColumns<1)
			throw OutOfRange;
	
		for(index_t j=jBegin; j<numColumns; j+=nColumns)
			pages.push_back(Layout(numRows, std::min(nColumns, numColumns-j), numSlices, getLeadingColumns(), getLeadingSlices(), getOffset()+getIndex(0,j,0)));

		return pages;
	}

	/**
	\brief Generates a series of layout, each containing a specified number of slices of the original layout.
	\param kBegin Origin slice index.
	\param nSlices Number of slices in each split.

	Note : the last split might contain less than nSlices slices (gracious).
	
	\throw OutOfRange If the index of the origin is invalid or nSlices is strictly less than 1.
	\return A vector containing all the splits (ordered).
	**/
	__host__ inline std::vector<Layout> Layout::splitLayoutSlices(index_t kBegin, index_t nSlices) const
	{
		std::vector<Layout> pages;

		if(!validSliceIndex(kBegin) || nSlices<1)
			throw OutOfRange;
	
		for(index_t k=kBegin; k<numSlices; k+=nSlices)
			pages.push_back(Layout(numRows, numColumns, std::min(nSlices, numSlices-k), getLeadingColumns(), getLeadingSlices(), getOffset()+getIndex(0,0,k)));

		return pages;
	}
	
	/**
	\brief Generates a series of layout, each containing a specified block layout.
	\param iBegin Origin row index.
	\param jBegin Origin column index.
	\param kBegin Origin slice index.
	\param nRows Number of rows in the block split.
	\param nColumns Number of columns in the block split.
	\param nSlices Number of slices in the block split.
	
	\throw OutOfRange If the index of the origin in any dimension is invalid or the size in any dimension is strictly less than 1.
	\return A vector containing all the splits (column-major ordered).
	**/
	__host__ inline std::vector<Layout> Layout::splitLayoutSubArrays(index_t iBegin, index_t jBegin, index_t kBegin, index_t nRows, index_t nColumns, index_t nSlices) const
	{
		std::vector<Layout> pages;

		// Fill missing arguments
		if(nRows==0)	
			nRows = getNumRows();
		if(nColumns==0)
			nColumns = getNumColumns();
		if(nSlices==0)
			nSlices = getNumSlices();

		if(!validRowIndex(iBegin) || nRows<1 || !validColumnIndex(jBegin) || nColumns<1 || !validSliceIndex(kBegin) || nSlices<1)
			throw OutOfRange;
		
		for(index_t k=kBegin; k<numSlices; k+=nSlices)
			for(index_t j=jBegin; j<numColumns; j+=nColumns)
				for(index_t i=iBegin; i<numRows; i+=nRows)
					pages.push_back(Layout(std::min(nRows, numRows-i), std::min(nColumns, numColumns-j), std::min(nSlices, numSlices-k), getLeadingColumns(), getLeadingSlices(), getOffset()+getIndex(i,j,k)));

		return pages;
	}

	/**
	\brief Generates a series of layout, each containing a specified block layout.
	\param iBegin Origin row index.
	\param jBegin Origin column index.
	\param kBegin Origin slice index.
	\param layout Block split size.
	
	\throw OutOfRange If the index of the origin in any dimension is invalid or the size in any dimension is strictly less than 1.
	\return A vector containing all the splits (column-major ordered).
	**/
	__host__ inline std::vector<Layout> Layout::splitLayoutSubArrays(index_t iBegin, index_t jBegin, index_t kBegin, const Layout& layout) const
	{
		return splitLayoutSubArrays(iBegin, jBegin, kBegin, layout.getNumRows(), layout.getNumColumns(), layout.getNumSlices());
	}

	/**
	\brief Test if this layout is identical to another layout.
	\param other Other layout object to be compared with.
	\return True if this layout has the same sizes and same strides as the other layout.
	**/
	__host__ __device__ inline bool Layout::sameLayoutAs(const Layout& other) const
	{
		return (numRows==other.numRows && numColumns==other.numColumns && numSlices==other.numSlices && leadingColumns==other.leadingColumns && leadingSlices==other.leadingSlices); 
	}

	/**
	\brief Test if the layout a slice from this object is identical to those of another layout.
	\param other Other layout object to be compared with.
	\return True if the slice layout of this object has the same sizes and same strides as for the other layout.
	**/
	__host__ __device__ inline bool Layout::sameSliceLayoutAs(const Layout& other) const
	{
		return (numRows==other.numRows && numColumns==other.numColumns && leadingColumns==other.leadingColumns); 
	}

	#ifdef __CUDACC__
	/**
	\brief Get the current row index.
	\return The current row index.
	**/
	__device__ inline index_t Layout::getI(void)
	{
		return blockIdx.x*blockDim.x+threadIdx.x;
	}

	/**
	\brief Get the current column index.
	\return The current column index.
	**/
	__device__ inline index_t Layout::getJ(void)
	{
		return blockIdx.y*blockDim.y+threadIdx.y;
	}

	/**
	\brief Get the current slice index.
	\return The current slice index.
	**/
	__device__ inline index_t Layout::getK(void)
	{
		return blockIdx.z*blockDim.z+threadIdx.z;
	}
	#endif

	/**
	\brief Get the normalized row index.
	\param i The original row index.

	Note : this function does not perform bound checking.
	
	\return The normalized row index (i/nRows).
	**/
	template<typename TOut>
	__host__ __device__ inline TOut Layout::getINorm(index_t i) const
	{
		return static_cast<TOut>(i)/static_cast<TOut>(numRows);
	}

	/**
	\brief Get the normalized column index.
	\param j The original column index.

	Note : this function does not perform bound checking.
	
	\return The normalized column index (j/nColumns).
	**/
	template<typename TOut>
	__host__ __device__ inline TOut Layout::getJNorm(index_t j) const
	{
		return static_cast<TOut>(j)/static_cast<TOut>(numColumns);
	}

	/**
	\brief Get the normalized slice index.
	\param k The original slice index.

	Note : this function does not perform bound checking.
	
	\return The normalized slice index (k/nSlices).
	**/
	template<typename TOut>
	__host__ __device__ inline TOut Layout::getKNorm(index_t k) const
	{
		return static_cast<TOut>(k)/static_cast<TOut>(numSlices);
	}

	#ifdef __CUDACC__
	/**
	\brief Get the current normalized row index.
	\return The normalized row index (i/nRows).
	**/
	template<typename TOut>
	__device__ inline TOut Layout::getINorm(void) const
	{
		return getINorm<TOut>(getI());
	}

	/**
	\brief Get the current normalized column index.
	\return The normalized column index (j/nColumns).
	**/
	template<typename TOut>
	__device__ inline TOut Layout::getJNorm(void) const
	{
		return getJNorm<TOut>(getJ());
	}

	/**
	\brief Get the current normalized slice index.
	\return The normalized slice index (k/nSlices).
	**/
	template<typename TOut>
	__device__ inline TOut Layout::getKNorm(void) const
	{
		return getKNorm<TOut>(getK());
	}
	#endif

	/**
	\brief Get the inclusive normalized row index.
	\param i The original row index.

	Note : this function does not perform bound checking.
	
	\return The inclusive normalized row index (i/(nRows-1)).
	**/
	template<typename TOut>
	__host__ __device__ inline TOut Layout::getINormIncl(index_t i) const
	{
		return static_cast<TOut>(i)/static_cast<TOut>(numRows-1);
	}

	/**
	\brief Get the inclusive normalized column index.
	\param j The original column index.

	Note : this function does not perform bound checking.
	
	\return The inclusive normalized column index (i/(nColumns-1)).
	**/
	template<typename TOut>
	__host__ __device__ inline TOut Layout::getJNormIncl(index_t j) const
	{
		return static_cast<TOut>(j)/static_cast<TOut>(numColumns-1);
	}

	/**
	\brief Get the inclusive normalized slice index.
	\param k The original slice index.

	Note : this function does not perform bound checking.
	
	\return The inclusive normalized slice index (i/(nSlices-1)).
	**/
	template<typename TOut>
	__host__ __device__ inline TOut Layout::getKNormIncl(index_t k) const
	{
		return static_cast<TOut>(k)/static_cast<TOut>(numSlices-1);
	}

	#ifdef __CUDACC__
	/**
	\brief Get the current inclusive normalized row index.
	\return The inclusive normalized row index (i/(nRows-1)).
	**/
	template<typename TOut>
	__device__ inline TOut Layout::getINormIncl(void) const
	{
		return getINormIncl<TOut>(getI());
	}

	/**
	\brief Get the current inclusive normalized column index.
	\return The inclusive normalized column index (i/(nColumns-1)).
	**/
	template<typename TOut>
	__device__ inline TOut Layout::getJNormIncl(void) const
	{
		return getJNormIncl<TOut>(getJ());
	}

	/**
	\brief Get the current inclusive normalized slice index.
	\return The inclusive normalized slice index (i/(nSlices-1)).
	**/
	template<typename TOut>
	__device__ inline TOut Layout::getKNormIncl(void) const
	{
		return getKNormIncl<TOut>(getK());
	}
	#endif

	/**
	\brief Get the clamped row index.
	\param i The original row index.
	\return The row index clamped to the range [0, nRows-1].
	**/
	__host__ __device__ inline index_t Layout::getIClamped(index_t i) const
	{
		return min( max(static_cast<index_t>(0), i), numRows-1);
	}

	/**
	\brief Get the clamped column index.
	\param j The original column index.
	\return The column index clamped to the range [0, nColumns-1].
	**/
	__host__ __device__ inline index_t Layout::getJClamped(index_t j) const
	{
		return min( max(static_cast<index_t>(0), j), numColumns-1);
	}

	/**
	\brief Get the clamped slice index.
	\param k The original slice index.
	\return The slice index clamped to the range [0, nSlices-1].
	**/
	__host__ __device__ inline index_t Layout::getKClamped(index_t k) const
	{
		return min( max(static_cast<index_t>(0), k), numSlices-1);
	}

	/**
	\brief Get the wrapped row index.
	\param i The original row index.
	\return The row index wraped to the range [0, nRows-1] (not mirrored).
	**/
	__host__ __device__ inline index_t Layout::getIWrapped(index_t i) const
	{
		return (i % numRows);
	}

	/**
	\brief Get the wrapped column index.
	\param j The original column index.
	\return The column index wraped to the range [0, nColumns-1] (not mirrored).
	**/
	__host__ __device__ inline index_t Layout::getJWrapped(index_t j) const
	{
		return (j % numColumns);
	}

	/**
	\brief Get the wrapped slice index.
	\param k The original slice index.
	\return The slice index wraped to the range [0, nSlices-1] (not mirrored).
	**/
	__host__ __device__ inline index_t Layout::getKWrapped(index_t k) const
	{
		return (k % numSlices);	
	}

	/**
	\brief Get the index of an element from coordinates.
	\param i The row index.
	\param j The column index.
	\param k The slice index.

	Note the function does not perform bound checking.

	\return The index of the element corresponding to the coordinates.
	**/
	__host__ __device__ inline index_t Layout::getIndex(index_t i, index_t j, index_t k) const
	{
		return k*leadingSlices + j*leadingColumns + i;
	}

	#ifdef __CUDACC__
	/**
	\brief Get the index of the current element.
	\return The index of the current element.
	**/
	__device__ inline index_t Layout::getIndex(void) const
	{
		return getIndex(getI(), getJ(), getK());
	}
	#endif

	/**
	\brief Get the index of an element from coordinates, after fftshift. Modify the arguments.
	\param i The row index. Will be modified.
	\param j The column index. Will be modified.
	\param k The slice index. Will be modified.

	Note the function does not perform bound checking.

	\return The index of the element corresponding to the fftshift coordinates.
	**/
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

	/**
	\brief Get the index of an element from coordinates, after fftshift.
	\param i The row index.
	\param j The column index.
	\param k The slice index.

	Note the function does not perform bound checking.

	\return The index of the element corresponding to the fftshift coordinates.
	**/
	__host__ __device__ inline index_t Layout::getIndexFFTShift(index_t i, index_t j, index_t k) const
	{
		return getIndicesFFTShift(i, j, k);
	}

	/**
	\brief Get the index of an element from coordinates, after ifftshift. Modify the arguments.
	\param i The row index. Will be modified.
	\param j The column index. Will be modified.
	\param k The slice index. Will be modified.

	Note the function does not perform bound checking.

	\return The index of the element corresponding to the ifftshift coordinates.
	**/
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

	/**
	\brief Get the index of an element from coordinates, after ifftshift.
	\param i The row index.
	\param j The column index.
	\param k The slice index.

	Note the function does not perform bound checking.

	\return The index of the element corresponding to the ifftshift coordinates.
	**/
	__host__ __device__ inline index_t Layout::getIndexFFTInverseShift(index_t i, index_t j, index_t k) const
	{
		return getIndicesFFTInverseShift(i, j, k);
	}

	/**
	\brief Get the index of an element from coordinates, after clamping.
	\param i The row index.
	\param j The column index.
	\param k The slice index.
	\return The index of the element corresponding to the clamped coordinates.
	**/
	__host__ __device__ inline index_t Layout::getIndexClampedToEdge(index_t i, index_t j, index_t k) const
	{
		return getIndex(getIClamped(i), getJClamped(j), getKClamped(k));
	}

	/**
	\brief Get the index of an element from coordinates, after wrapping.
	\param i The row index.
	\param j The column index.
	\param k The slice index.
	\return The index of the element corresponding to the wrapped coordinates.
	**/
	__host__ __device__ inline index_t Layout::getIndexWarped(index_t i, index_t j, index_t k) const
	{
		return getIndex(getIWrapped(i), getJWrapped(j), getKWrapped(k));
	}

	#ifdef __CUDACC__
	/**
	\brief Get the index of the current element, after fftshift.
	\return The index of the current element, after fftshift.
	**/
	__device__ inline index_t Layout::getIndexFFTShift(void) const
	{
		return getIndexFFTShift(getI(), getJ(), getK());
	}

	/**
	\brief Get the index of the current element, after inverse fftshift.
	\return The index of the current element, after inverse fftshift.
	**/
	__device__ inline index_t Layout::getIndexFFTInverseShift(void) const
	{
		return getIndexFFTInverseShift(getI(), getJ(), getK());
	}
	#endif
	
	/**
	\brief Test if coordinates are inside a layout.
	\param i The row index.
	\param j The column index.
	\param k The slice index.
	\return True if the coordinates are inside the layout.
	**/
	__host__ __device__ inline bool Layout::isInside(index_t i, index_t j, index_t k) const
	{
		return (i>=0 && i<numRows && j>=0 && j<numColumns && k>=0 && k<numSlices);
	}

	#ifdef __CUDACC__
	/**
	\brief Test if the current coordinates are inside a layout.
	\return True if the coordinates are inside the layout.
	**/
	__device__ inline bool Layout::isInside(void) const
	{
		return  isInside(getI(), getJ(), getK());
	}
	#endif

	/**
	\brief Test if a row index is valid.
	\param i Row index.
	\return True if the row index is valid.
	**/
	__host__ __device__ inline bool Layout::validRowIndex(index_t i) const	
	{
		return (i>=0 && i<numRows);
	}

	/**
	\brief Test if a column index is valid.
	\param j Column index.
	\return True if the column index is valid.
	**/
	__host__ __device__ inline bool Layout::validColumnIndex(index_t j) const
	{
		return (j>=0 && j<numColumns);
	}

	/**
	\brief Test if a slice index is valid.
	\param k Slice index.
	\return True if the slice index is valid.
	**/
	__host__ __device__ inline bool Layout::validSliceIndex(index_t k) const
	{
		return (k>=0 && k<numSlices);
	}

	/**
	\brief Unpack element index to coordinates.
	\param index Element index.
	\param i Output, row index.
	\param j Output, column index.
	\param k Output, slice index.
	**/
	__host__ __device__ inline void Layout::unpackIndex(index_t index, index_t& i, index_t& j, index_t& k) const
	{
		k = index / leadingSlices;
		j = (index - k*leadingSlices) / leadingColumns;
		i = index - k*leadingSlices - j*leadingColumns;
	}

	/**
	\brief Move to next coordinate in the layout.
	\param i Input : current row index. Output : next row index.
	\param j Input : current column index. Output : next column index.
	\param k Input : current slice index. Output : next slice index.
	**/
	__host__ __device__ inline void Layout::moveToNextIndex(index_t& i, index_t& j, index_t& k) const
	{
		/*
		// This version is the "protected version" 
		// It will safely warp bad coordinates.
		i = ((i+1) % numRows);
		j = (((i==0) ? (j+1) : j) % numColumns);
		k = ((i==0 && j==0) ? (k+1) : k);
		*/

		// This version is the "unprotected version"
		// It is also faster. 
		i = (i+1);
		i = (i>=numRows) ? 0 : i;
		j = (i==0) ? (j+1) : j;
		j = (j>=numColumns) ? 0 : j;
		k = (i==0 && j==0) ? (k+1) : k;
	}

	#ifdef __CUDACC__
	/**
	\brief Get the block size for CUDA computation.
	\return dim3 struct with correct block settings.
	**/
	__host__ inline dim3 Layout::getBlockSize(void) const
	{
		dim3 d;
		// From inner most dimension (I <-> X, J <-> Y, K <-> Z) :
		d.x = min(StaticContainer<void>::numThreads, numRows);
		d.y = min(StaticContainer<void>::numThreads/d.x, numColumns);
		d.z = min(min(StaticContainer<void>::numThreads/(d.x*d.y), numSlices), StaticContainer<void>::maxZThreads);
		//std::cout << "Layout::getBlockSize : " << d.x << ", " << d.y << ", " << d.z << std::endl;
		return d;
	}

	/**
	\brief Get the grid size for CUDA computation.
	\return dim3 struct with correct grid settings.
	**/
	__host__ inline dim3 Layout::getNumBlock(void) const
	{
		dim3 d;
		// From inner most dimension (I <-> X, J <-> Y, K <-> Z) :
		const dim3 blockSize = getBlockSize();
		d.x = (numRows + blockSize.x - 1)/blockSize.x;
		d.y = (numColumns + blockSize.y - 1)/blockSize.y;
		d.z = (numSlices + blockSize.z - 1)/blockSize.z;
		//std::cout << "Layout::getNumBlock : " << d.x << ", " << d.y << ", " << d.z << std::endl;
		return d;
	}
	#endif

	/**
	\brief Get the layout corresponding to a vector.
	\return The layout of a vector.
	**/
	__host__ inline Layout Layout::getVectorLayout(void) const
	{
		return Layout(numRows);
	}

	/**
	\brief Get the layout corresponding to a slice.
	\return The layout of a slice.
	**/
	__host__ inline Layout Layout::getSliceLayout(void) const
	{
		return Layout(numRows, numColumns, 1, getLeadingColumns());
	}

	/**
	\brief Get the monolithic layout corresponding to this object.
	\return The monolithic layout corresponding to this object.
	**/
	__host__ inline Layout Layout::getMonolithicLayout(void) const
	{
		return Layout(numRows, numColumns, numSlices);
	}

	/**
	\brief Perform a single scan operation for this layout using the provided operator.
	\param ptr Data pointer.
	\param op Operator object.
	
	Operator must have the following member function : 
	\code{.unparsed}
	void apply(const Layout& mainLayout, const Layout& currentAccessLayout, T* ptr, size_t offset, index_t i, index_t j, index_t k) const;
			mainLayout          : the original layout.
			currentAccessLayout : the layout where access is currently granted.
			ptr                 : the original pointer. DOES NOT CONTAIN THE OFFSET.
			offset              : the offset leading to the right portion of the memory.
			i, j, k             : the corresponding coordinates.
	\endcode
	**/
	template<class Op, typename T>
	__host__ void Layout::singleScan(T* ptr, const Op& op) const
	{
		if(numRows==getLeadingColumns() && getNumElementsPerSlice()==getLeadingSlices())
			op.apply(*this, *this, ptr, 0, 0, 0, 0);		
		else if(numRows==getLeadingColumns())
		{
			const Layout sliceLayout = getSliceLayout();
			for(index_t k=0; k<numSlices; k++)
				op.apply(*this, sliceLayout, ptr, k*getLeadingSlices(), 0, 0, k);	
		}
		else
		{
			const Layout vectorLayout = getVectorLayout();
			for(index_t k=0; k<numSlices; k++)
			{
				for(index_t j=0; j<numColumns; j++)
					op.apply(*this, vectorLayout, ptr, (k*getLeadingSlices() + j*getLeadingColumns()), 0, j, k);
			}
		}
	}

	/**
	\brief Perform a dual scan operation on the layouts provided.
	\param layoutA First layout.
	\param ptrA First data pointer.
	\param layoutB Second layout.
	\param ptrB Second data pointer.
	\param op Operator object.
	
	Operator must have the following member function :
	\code{.unparsed}
	void apply(const Layout& mainLayout, const Layout& currentAccessLayout, T* ptrA, T* ptrB, size_t offsetA, size_t offsetB, index_t i, index_t j, index_t k) const;
			mainLayout          : the original layout.
			currentAccessLayout : the layout where access is currently granted.
			ptrA, ptrB          : the original pointers. DOES NOT CONTAIN THE OFFSET.
			offsetA, offsetB    : the offset leading to the right portion of the memory.
			i, j, k             : the corresponding coordinates.
	\endcode
	**/
	template<class Op, typename T>
	__host__ void Layout::dualScan(const Layout& layoutA, T* ptrA, const Layout& layoutB, T* ptrB, const Op& op)
	{
		if(!layoutA.getMonolithicLayout().sameLayoutAs(layoutB.getMonolithicLayout()))
			throw IncompatibleLayout;

		if(	(layoutA.getNumRows()==layoutA.getLeadingColumns() && layoutA.getNumElementsPerSlice()==layoutA.getLeadingSlices()) &&
			(layoutB.getNumRows()==layoutB.getLeadingColumns() && layoutB.getNumElementsPerSlice()==layoutB.getLeadingSlices()) )
		{
			op.apply(layoutA, layoutA, ptrA, ptrB, 0, 0, 0, 0, 0);
		}
		else if((layoutA.getNumRows()==layoutA.getLeadingColumns()) &&
			(layoutB.getNumRows()==layoutB.getLeadingColumns()) )
		{
			const Layout sliceLayout = layoutA.getSliceLayout();
			for(index_t k=0; k<layoutA.getNumSlices(); k++)
				op.apply(layoutA, sliceLayout, ptrA, ptrB, k*layoutA.getLeadingSlices(), k*layoutB.getLeadingSlices(), 0, 0, k);
		}
		else
		{
			const Layout vectorLayout = layoutA.getVectorLayout();
			for(index_t k=0; k<layoutA.getNumSlices(); k++)
			{
				for(index_t j=0; j<layoutB.getNumColumns(); j++)
					op.apply(layoutA, vectorLayout, ptrA, ptrB, (k*layoutA.getLeadingSlices() + j*layoutA.getLeadingColumns()), (k*layoutB.getLeadingSlices() + j*layoutB.getLeadingColumns()), 0, j, k);
			}
		}
	}

	/**
	\brief Read Layout from input stream.
	\param stream Input stream.
	\param typeIndex Output, type code.

	If not NULL, the destination will be written with the type code following the layout (single integer).
	
	\return The Layout built from the stream.
	**/
	__host__ inline Layout Layout::readFromStream(std::istream& stream, int* typeIndex)
	{
		if(!stream.good())
			throw InvalidInputStream;
		
		const size_t bufferLength = 32;
		if(bufferLength<StaticContainer<void>::getStreamHeaderLength())
			throw InvalidOperation;
		char headerBuffer[bufferLength];
		std::memset(headerBuffer, 0, bufferLength);
		stream.read(headerBuffer, StaticContainer<void>::getStreamHeaderLength()-1);
		if(strncmp(StaticContainer<void>::streamHeader, headerBuffer, StaticContainer<void>::getStreamHeaderLength()-1)!=0)
			throw InvalidStreamHeader;
	
		// Read the type :
		int dummyType;
		if(typeIndex==NULL)
			typeIndex = &dummyType;
		stream.read(reinterpret_cast<char*>(typeIndex), sizeof(int)); if(!stream.good()) throw InvalidInputStream;

		// Read the sizes :
		index_t	r = 0,
			c = 0,
			s = 0;
		stream.read(reinterpret_cast<char*>(&r), sizeof(index_t)); if(!stream.good()) throw InvalidInputStream;
		stream.read(reinterpret_cast<char*>(&c), sizeof(index_t)); if(!stream.good()) throw InvalidInputStream;
		stream.read(reinterpret_cast<char*>(&s), sizeof(index_t)); if(stream.fail()) throw InvalidInputStream; // Do not test for EOF after this read.

		// Return :
		return Layout(r, c, s);
	}

	/**
	\brief Read Layout from a file.
	\param filename Filename to load.
	\param typeIndex Output, type code.

	If not NULL, the destination will be written with the type code following the layout (single integer).
	
	\return The Layout built from the file.
	**/
	__host__ inline Layout Layout::readFromFile(const std::string& filename, int* typeIndex)
	{
		std::ifstream file(filename.c_str(), std::ios::in | std::ios::binary);

		if(!file.is_open())
			throw InvalidInputStream;

		Layout layout = readFromStream(file, typeIndex);
		file.close();

		return layout;
	}

	/**
	\brief Write Layout to the output stream.
	\param stream Stream to write.
	\param typeIndex Type code.
	**/
	__host__ inline void Layout::writeToStream(std::ostream& stream, int typeIndex)
	{
		if(!stream.good())
			throw InvalidOutputStream;

		// Write the header :
		stream.write(StaticContainer<void>::streamHeader, StaticContainer<void>::getStreamHeaderLength()-1);
		
		// Write the type :	
		stream.write(reinterpret_cast<char*>(&typeIndex), sizeof(int));
	
		// Write the size :
		stream.write(reinterpret_cast<char*>(&numRows), sizeof(index_t));
		stream.write(reinterpret_cast<char*>(&numColumns), sizeof(index_t));
		stream.write(reinterpret_cast<char*>(&numSlices), sizeof(index_t));	
	}

	/**
	\brief Write Layout to the file.
	\param filename Filename to write to.
	\param typeIndex Type code.
	**/
	__host__ inline void Layout::writeToFile(const std::string& filename, int typeIndex)
	{
		std::ofstream file(filename.c_str(), std::ios::out | std::ios::binary);

		if(!file.is_open())
			throw InvalidOutputStream;

		writeToStream(file, typeIndex);
		file.close();
	}

	/**
	\brief Write Layout to the output stream.
	\param stream Stream to write.

	Template parameter : T, Type of the data attached to the layout.
	**/
	template<typename T>
	__host__ inline void Layout::writeToStream(std::ostream& stream)
	{
		writeToStream(stream, GetIndex<TypesSortedByAccuracy, T>::value);
	}

	/**
	\brief Write Layout to the file.
	\param filename Filename to write to.

	Template parameter : T, Type of the data attached to the layout.
	**/
	template<typename T>
	__host__ inline void Layout::writeToFile(const std::string& filename)
	{
		writeToFile(filename, GetIndex<TypesSortedByAccuracy, T>::value);
	}

	/**
	\brief Write layout to ostream object (std::cout, std::cerr, etc.).
	\param os std::ostream object to write to.
	\param layout Layout object.

	Example : 
	Layout l(3, 5, 7);
	std;;cout << l << std::endl;

	Output (strides are replaced by an underscore if elements are contiguous, and not printed if the layout is monolithic) : 
	['rows', 'columns', 'slices', 'leadingColumns', 'leadingSlices', 'offset']

	\return Reference to the modified std::ostream, for chain.
	**/
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
	template<typename T, Location l>
	__host__ __device__ Accessor<T,l>::Accessor(index_t r, index_t c, index_t s, index_t lc, index_t ls, index_t o)
	 : 	Layout(r, c, s, lc, ls, o),
		ptr(NULL)	
	{ }

	template<typename T, Location l>
	__host__ __device__ Accessor<T,l>::Accessor(const Layout& layout)
	 : 	Layout(layout), 
		ptr(NULL)
	{ }

	template<typename T, Location l>
	__host__ __device__ Accessor<T,l>::Accessor(T* ptr, index_t r, index_t c, index_t s, index_t lc, index_t ls, index_t o)
	 :	Layout(r, c, s, lc, ls, o),
		ptr(ptr)
	{ }

	template<typename T, Location l>
	__host__ __device__ Accessor<T,l>::Accessor(T* ptr, const Layout& layout)
	 :	Layout(layout),
		ptr(ptr)
	{ }
	
	template<typename T, Location l>
	__host__ Accessor<T,l>::Accessor(const Array<T,l>& a)
	 :	Layout(a), 
		ptr(a.ptr)
	{ }

	template<typename T, Location l>
	__host__ __device__ Accessor<T,l>::Accessor(const Accessor<T,l>& a)
	 : 	Layout(a),
		ptr(a.ptr)
	{ }

	template<typename T, Location l>
	__host__ __device__ Location Accessor<T,l>::getLocation(void) const
	{
		return l;
	}

	template<typename T, Location l>
	__host__ __device__ T* Accessor<T,l>::getPtr(void) const
	{
		return ptr;
	}

	template<typename T, Location l>
	__host__ __device__ size_t Accessor<T,l>::getSize(void) const
	{
		return static_cast<size_t>(getNumElements())*sizeof(T);
	}	

	template<typename T, Location l>
	__host__ __device__ inline T& Accessor<T,l>::data(index_t i, index_t j, index_t k) const
	{
		return ptr[getIndex(i, j, k)];
	}

	template<typename T, Location l>
	__host__ __device__ inline T& Accessor<T,l>::data(index_t p) const
	{
		return ptr[p];
	}

	#ifdef __CUDACC__
	template<typename T, Location l>
	__device__ inline T& Accessor<T,l>::data(void) const
	{
		return ptr[getIndex()];
	}

	template<typename T, Location l>
	__device__ inline T& Accessor<T,l>::dataInSlice(int k) const
	{
		return ptr[getIndex(getI(),getJ(),k)];
	}

	template<typename T, Location l>
	__device__ inline T& Accessor<T,l>::dataFFTShift(void) const
	{
		return ptr[getIndexFFTShift()];
	}

	template<typename T, Location l>
	__device__ inline T& Accessor<T,l>::dataFFTInverseShift(void) const
	{
		return ptr[getIndexFFTInverseShift()];
	}
	#endif

	template<typename T, Location l>
	T* Accessor<T,l>::getData(void) const
	{	
		T* ptr = new T[getNumElements()];
		getData(ptr);
		return ptr;
	}

	// Tools for the memcpy :
		template<typename T>
		struct MemCpyToolBox
		{			
			const Direction		direction;
			const T			*from;
			T			*to;
			const Layout		originalLayout,
						solidLayout;

			__host__ MemCpyToolBox(const Direction _d, T* _to, const T* _from, const Layout& _originalLayout)
			 :	direction(_d),
				from(_from),
				to(_to),
				originalLayout(_originalLayout),
				solidLayout(_originalLayout.getMonolithicLayout())
			{
				#ifdef __CUDACC__
					if(direction==HostToDevice || direction==DeviceToHost || direction==DeviceToDevice)
						cudaDeviceSynchronize();
				#endif
			}
			
			__host__ void apply(const Layout& mainLayout, const Layout& currentAccessLayout, T* ptr, size_t offset, index_t i, index_t j, index_t k) const
			{
				UNUSED_PARAMETER(mainLayout)
				UNUSED_PARAMETER(ptr)
				UNUSED_PARAMETER(offset)

				size_t	toOffset = 0,
					fromOffset = 0;
				if(direction==DeviceToHost)
				{
					toOffset	= solidLayout.getIndex(i, j, k);
					fromOffset	= originalLayout.getIndex(i, j, k);
				}
				else
				{
					toOffset	= originalLayout.getIndex(i, j, k);
					fromOffset	= solidLayout.getIndex(i, j, k);
				}

				#ifdef __CUDACC__
					const cudaMemcpyKind _direction = getCudaDirection(direction);
					cudaError_t err = cudaMemcpy((to + toOffset), (from + fromOffset), currentAccessLayout.getNumElements()*sizeof(T), _direction);
					if(err!=cudaSuccess)
						throw static_cast<Exception>(CudaExceptionsOffset + err);
				#else
					memcpy((to + toOffset), (from + fromOffset), currentAccessLayout.getNumElements()*sizeof(T));
				#endif
			}
		};

	template<typename T, Location l>
	void Accessor<T,l>::getData(T* dst, const Location lout) const
	{
		if(dst==NULL)
			throw NullPointer;
	
		const Direction direction = getDirection(l, lout);
		MemCpyToolBox<T> toolbox(direction, dst, ptr, *this);
		singleScan(toolbox);
	}

	template<typename T, Location l>
	void Accessor<T,l>::setData(const T* src, const Location lin) const
	{
		if(src==NULL)
			throw NullPointer;

		const Direction direction = getDirection(lin, l);
		MemCpyToolBox<T> toolbox(direction, ptr, src, *this);
		singleScan(toolbox);
	}

	// Tool for the file input : 
		template<typename T>
		struct StreamInputToolBox
		{
			std::istream& 		stream;
			const Location		destinationLocation;
			const int		sourceTypeIndex;
			const bool		conversion;
			const size_t		sourceTypeSize,
						numBufferElements;
			char			*bufferRead,
						*bufferCast;

			__host__ StreamInputToolBox(std::istream& _stream, const Location _destinationLocation, const int _sourceTypeIndex, const size_t _sourceTypeSize, const size_t _numBufferElements)
			 :	stream(_stream),
				destinationLocation(_destinationLocation),
				sourceTypeIndex(_sourceTypeIndex),
				conversion(_sourceTypeIndex!=GetIndex<TypesSortedByAccuracy, T>::value),				
				sourceTypeSize(_sourceTypeSize),
				numBufferElements(_numBufferElements),
				bufferRead(NULL),
				bufferCast(NULL)
			{
				// Allocation of the write buffer :
				if(destinationLocation==DeviceSide)
				{
					#ifdef __CUDACC__
						bufferRead = new char[numBufferElements*sourceTypeSize];
						if(conversion)
							bufferCast = new char[numBufferElements*sizeof(T)];
						cudaDeviceSynchronize();
					#else
						throw NotSupported;
					#endif
				}
				else
				{
					if(conversion)
						bufferRead = new char[numBufferElements*sourceTypeSize];
				}
			}

			__host__ ~StreamInputToolBox(void)
			{
				delete[] bufferRead;
				delete[] bufferCast;
			}
			
			__host__ void apply(const Layout& mainLayout, const Layout& currentAccessLayout, T* ptr, size_t offset, index_t i, index_t j, index_t k) const
			{
				UNUSED_PARAMETER(mainLayout)
				UNUSED_PARAMETER(i)
				UNUSED_PARAMETER(j)
				UNUSED_PARAMETER(k)

				if(destinationLocation==HostSide && !conversion)
				{
					stream.read(reinterpret_cast<char*>(ptr + offset), currentAccessLayout.getNumElements()*sizeof(T));
					if(stream.fail())
						throw InvalidInputStream;
				}
				else
				{
					for(index_t offsetCopied=0; offsetCopied<currentAccessLayout.getNumElements(); )
					{
						const index_t currentCopyNumElements = std::min(currentAccessLayout.getNumElements()-offsetCopied, static_cast<index_t>(numBufferElements));
						stream.read(bufferRead, currentCopyNumElements*sourceTypeSize);
						if(stream.fail())
							throw InvalidInputStream;
		
						char* tmpSrc = bufferRead;
						if(conversion)
						{
							char* castPtr = (destinationLocation==HostSide) ? reinterpret_cast<char*>(ptr + offset + offsetCopied) : bufferCast;
							dynamicCopy(reinterpret_cast<T*>(castPtr), bufferRead, sourceTypeIndex, currentCopyNumElements);
							tmpSrc = castPtr;
						}

						if(destinationLocation==DeviceSide)
						{
							#ifdef __CUDACC__
								cudaError_t err = cudaMemcpy((ptr + offset + offsetCopied), tmpSrc, currentCopyNumElements*sizeof(T), cudaMemcpyHostToDevice);
								if(err!=cudaSuccess)
									throw static_cast<Exception>(CudaExceptionsOffset + err);
							#else
								throw NotSupported;
							#endif
						}

						offsetCopied += currentCopyNumElements;
					}
				}
			}
		};

	template<typename T, Location l>
	__host__ void Accessor<T,l>::readFromStream(std::istream& stream, bool convert, const size_t maxBufferSize, const bool skipHeader, int sourceTypeIndex)
	{
		if(maxBufferSize<sizeof(T))
			throw InvalidOperation;
		if(!stream.good())
			throw InvalidInputStream;
		
		Layout layout = getLayout();
		if(!skipHeader)
			layout = Layout::readFromStream(stream, &sourceTypeIndex);
		if(!layout.sameLayoutAs(*this))
			throw IncompatibleLayout;
	
		const bool conversion = (sourceTypeIndex!=GetIndex<TypesSortedByAccuracy, T>::value);
		if(!convert && conversion)
			throw InvalidOperation;
		
		const size_t 	sourceTypeSize = sizeOfType(sourceTypeIndex),
				maxSize = conversion ? (sourceTypeSize + sizeof(T)) : sizeof(T),
				numBufferElements = std::min(static_cast<size_t>(layout.getNumElements())*maxSize, maxBufferSize)/maxSize;
		
		StreamInputToolBox<T> toolbox(stream, l, sourceTypeIndex, sourceTypeSize, numBufferElements);
		singleScan(toolbox);
	}

	template<typename T, Location l>
	__host__ void Accessor<T,l>::readFromFile(const std::string& filename, bool convert, const size_t maxBufferSize)
	{
		std::ifstream file(filename.c_str(), std::ios::in | std::ios::binary);
		if(!file.is_open() || file.fail())
			throw InvalidInputStream;

		readFromStream(file, convert, maxBufferSize);
		file.close();
	}

	// Tools for the file output :
		template<typename T>
		struct StreamOutputToolBox
		{
			std::ostream& 		stream;
			const Location		sourceLocation;
			T			*buffer;
			const size_t		numBufferElements;

			__host__ StreamOutputToolBox(std::ostream& _stream, const Location _sourceLocation, const size_t _numBufferElements)
			 :	stream(_stream),
				sourceLocation(_sourceLocation),
				buffer(NULL),
				numBufferElements(_numBufferElements)
			{
				// Allocation of the write buffer :
				if(sourceLocation==DeviceSide)
				{
					#ifdef __CUDACC__
						
						buffer = new T[numBufferElements];
						cudaDeviceSynchronize();
					#else
						throw NotSupported;
					#endif
				}
			}

			__host__ ~StreamOutputToolBox(void)
			{
				delete[] buffer;
			}
			
			__host__ void apply(const Layout& mainLayout, const Layout& currentAccessLayout, T* ptr, size_t offset, index_t i, index_t j, index_t k) const
			{
				UNUSED_PARAMETER(mainLayout)
				UNUSED_PARAMETER(i)
				UNUSED_PARAMETER(j)
				UNUSED_PARAMETER(k)

				if(sourceLocation==DeviceSide)
				{
					#ifdef __CUDACC__
						// Copy to the buffer :
						for(index_t offsetCopied=0; offsetCopied<currentAccessLayout.getNumElements(); )
						{
							const index_t currentCopyNumElements = std::min(currentAccessLayout.getNumElements()-offsetCopied, static_cast<index_t>(numBufferElements));
							cudaError_t err = cudaMemcpy(buffer, (ptr + offset + offsetCopied), currentCopyNumElements*sizeof(T), cudaMemcpyDeviceToHost);
							if(err!=cudaSuccess)
								throw static_cast<Exception>(CudaExceptionsOffset + err);
							stream.write(reinterpret_cast<char*>(buffer), currentCopyNumElements*sizeof(T));
							if(stream.fail())
								throw InvalidOutputStream;
							offsetCopied += currentCopyNumElements;
						}
					#else
						throw NotSupported;
					#endif
				}
				else
				{
					// Write the full data directly :
					stream.write(reinterpret_cast<char*>(ptr + offset), currentAccessLayout.getNumElements()*sizeof(T));
					if(stream.fail())
						throw InvalidOutputStream;
				}
			}
		};

	template<typename T, Location l>
	__host__ void Accessor<T,l>::writeToStream(std::ostream& stream, const size_t maxBufferSize)
	{
		if(maxBufferSize<sizeof(T))
			throw InvalidOperation;

		const size_t numBufferElements = std::min(static_cast<size_t>(getNumElements())*sizeof(T), maxBufferSize)/sizeof(T);

		// Write the header :
		Layout::writeToStream<T>(stream);
		StreamOutputToolBox<T> toolbox(stream, l, numBufferElements);
		singleScan(toolbox);
	}

	template<typename T, Location l>
	__host__ void Accessor<T,l>::writeToFile(const std::string& filename, const size_t maxBufferSize)
	{
		std::ofstream file(filename.c_str(), std::ios::out | std::ios::binary);
		if(!file.is_open() || file.fail())
			throw InvalidOutputStream;

		writeToStream(file, maxBufferSize);
		file.close();
	}

	template<typename T, Location l>
	__host__ const Layout& Accessor<T,l>::getLayout(void) const
	{
		return (*this);
	}

	template<typename T, Location l>
	__host__ Accessor<T,l> Accessor<T,l>::element(index_t i, index_t j, index_t k) const
	{
		if(!isInside(i, j, k))
			throw OutOfRange;		

		return Accessor<T,l>(ptr + getIndex(i, j, k), 1, 1, 1, 1, 1, getOffset()+getIndex(i, j, k));
	}

	template<typename T, Location l>
	__host__ Accessor<T,l> Accessor<T,l>::elements(index_t p, index_t numElements) const
	{
		if(p<0 || numElements<0 || p>=getNumElements() || (p+numElements)>getNumElements())
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
		return Accessor<T,l>(ptr + p, numElements, 1, 1, numElements, numElements, getOffset()+p);
	}

	template<typename T, Location l>
	__host__ Accessor<T,l> Accessor<T,l>::elements(void) const
	{
		if(!isMonolithic())
			throw InvalidOperation;
		return Accessor<T,l>(ptr, getNumElements(), 1, 1, getNumElements(), getNumElements(), getOffset());
	}

	template<typename T, Location l>
	__host__ Accessor<T,l> Accessor<T,l>::vector(index_t j) const
	{
		if(!validColumnIndex(j))
			throw OutOfRange;

		return Accessor<T,l>(ptr + getIndex(0,j,0), getNumRows(), 1, getNumSlices(), getNumRows(), getLeadingSlices(), getOffset()+getIndex(0,j,0));
	}

	template<typename T, Location l>
	__host__ Accessor<T,l> Accessor<T,l>::endVector(void) const
	{
		return vector(getNumColumns()-1);
	}

	template<typename T, Location l>
	__host__ Accessor<T,l> Accessor<T,l>::vectors(index_t jBegin, index_t numVectors, index_t jStep) const
	{
		if(jStep<=0 || numVectors<=0)
			throw InvalidNegativeStep;
		if(!validColumnIndex(jBegin) || !validColumnIndex(jBegin+(numVectors-1)*jStep))
			throw OutOfRange;

		return Accessor<T,l>(ptr + getIndex(0,jBegin,0), getNumRows(), numVectors, getNumSlices(), jStep*getLeadingColumns(), getLeadingSlices(), getOffset()+getIndex(0,jBegin,0));
	}

	template<typename T, Location l>
	__host__ Accessor<T,l> Accessor<T,l>::slice(index_t k) const
	{
		if(!validSliceIndex(k))
			throw OutOfRange;

		return Accessor<T,l>(ptr + getIndex(0,0,k), getNumRows(), getNumColumns(), 1, getLeadingColumns(), getOffset()+getIndex(0,0,k));
	}

	template<typename T, Location l>
	__host__ Accessor<T,l> Accessor<T,l>::endSlice(void) const
	{
		return slice(getNumSlices()-1);
	}

	template<typename T, Location l>
	__host__ Accessor<T,l> Accessor<T,l>::slices(index_t kBegin, index_t numSlices, index_t kStep) const
	{
		if(kStep<0 || numSlices<=0)
			throw InvalidNegativeStep;
		if(!validSliceIndex(kBegin) || !validSliceIndex(kBegin+(numSlices-1)*kStep))
			throw OutOfRange;

		return Accessor<T,l>(ptr + getIndex(0,0,kBegin), getNumRows(), getNumColumns(), numSlices, getLeadingColumns(), kStep*getLeadingSlices(), getOffset()+getIndex(0,0,kBegin));
	}

	template<typename T, Location l>
	__host__ Accessor<T,l> Accessor<T,l>::subArray(index_t iBegin, index_t jBegin, index_t numRows, index_t numColumns) const
	{
		if(numRows<=0 || numColumns<=0)
			throw InvalidNegativeStep;
		if(!validRowIndex(iBegin) || !validRowIndex(iBegin+numRows-1) || !validColumnIndex(jBegin) || !validColumnIndex(jBegin+numColumns-1))
			throw OutOfRange;
		
		return Accessor<T,l>(ptr + getIndex(iBegin,jBegin,0), numRows, numColumns, getNumSlices(), getLeadingColumns(), getLeadingSlices(), getOffset()+getIndex(iBegin,jBegin,0));
	}

	template<typename T, Location l>
	__host__ Accessor<T,l> Accessor<T,l>::flattened(void) const
	{
		Accessor<T, l> result = (*this);
		result.flatten();
		return result;
	}

	template<typename T, Location l>
	__host__ Accessor<T,l> Accessor<T,l>::stretched(void) const
	{
		Accessor<T, l> result = (*this);
		result.stretch();
		return result;
	}

	template<typename T, Location l>
	__host__ Accessor<T,l> Accessor<T,l>::vectorized(void) const
	{
		Accessor<T, l> result = (*this);
		result.vectorize();
		return result;
	}

	template<typename T, Location l>
	__host__  std::vector< Accessor<T,l> > Accessor<T,l>::splitColumns(index_t jBegin, index_t nColumns) const
	{
		std::vector< Accessor<T,l> > pages;
		Layout tmp = this->getLayout();
		tmp.setOffset(0);
		const std::vector<Layout> pagesLayout = tmp.splitLayoutColumns(jBegin, nColumns);

		for(std::vector<Layout>::const_iterator it=pagesLayout.begin(); it!=pagesLayout.end(); it++)	
		{
			pages.push_back( Accessor<T,l>(ptr + it->getOffset(), *it) );
			pages.back().setOffset(getOffset() + pages.back().getOffset());
		}

		return pages;
	}

	template<typename T, Location l>
	__host__  std::vector< Accessor<T,l> > Accessor<T,l>::splitSlices(index_t kBegin, index_t nSlices) const
	{
		std::vector< Accessor<T,l> > pages;
		Layout tmp = this->getLayout();
		tmp.setOffset(0);
		const std::vector<Layout> pagesLayout = tmp.splitLayoutSlices(kBegin, nSlices);

		for(std::vector<Layout>::const_iterator it=pagesLayout.begin(); it!=pagesLayout.end(); it++)	
		{
			pages.push_back( Accessor<T,l>(ptr + it->getOffset(), *it) );
			pages.back().setOffset(getOffset() + pages.back().getOffset());
		}

		return pages;
	}

	template<typename T, Location l>
	__host__ std::vector< Accessor<T,l> > Accessor<T,l>::splitSubArrays(index_t iBegin, index_t jBegin, index_t kBegin, index_t nRows, index_t nColumns, index_t nSlices) const
	{
		std::vector< Accessor<T,l> > pages;
		Layout tmp = this->getLayout();
		tmp.setOffset(0);
		const std::vector<Layout> pagesLayout = tmp.splitLayoutSubArrays(iBegin, jBegin, kBegin, nRows, nColumns, nSlices);

		for(std::vector<Layout>::const_iterator it=pagesLayout.begin(); it!=pagesLayout.end(); it++)	
		{
			pages.push_back( Accessor<T,l>(ptr + it->getOffset(), *it) );
			pages.back().setOffset(getOffset() + pages.back().getOffset());
		}

		return pages;
	}

	template<typename T, Location l>
	__host__ std::vector< Accessor<T,l> > Accessor<T,l>::splitSubArrays(index_t iBegin, index_t jBegin, index_t kBegin, const Layout& layout) const
	{
		return Accessor<T,l>::splitSubArrays(iBegin, jBegin, kBegin, layout.getNumRows(), layout.getNumColumns(), layout.getNumSlices());
	}

	template<typename T, Location l>
	template<class Op>
	__host__ void Accessor<T,l>::singleScan(const Op& op) const
	{
		Layout::singleScan(ptr, op);
	}

	template<typename T, Location l>
	__host__ std::ostream& operator<<(std::ostream& os, const Accessor<T,l>& A)
	{
		typedef typename StaticIf<SameTypes<T, char>::test || SameTypes<T, unsigned char>::test, int, T>::TValue CastType;

		#define FMT std::right << std::setfill(fillCharacter) << std::setw(maxWidth)
		const int precision = 4,
			  maxWidth = precision+3;
		const char fillCharacter = ' ';
		const char* spacing = "  ";
		const Kartet::Layout layout = A.getMonolithicLayout();

		T* tmp = (A.getLocation()==DeviceSide) ? A.getData() : A.getPtr();

		// Get old parameter :
		const int oldPrecision = os.precision(precision);

		os << "(Array of size " << A.getLayout() << ", " << ((A.getLocation()==DeviceSide) ? "DeviceSide" : "HostSide") << ')' << std::endl;
		for(int k=0; k<layout.getNumSlices(); k++)
		{
			if(layout.getNumSlices()>1)
				os << "Slice " << k << " : "<< std::endl;
	
			for(int i=0; i<layout.getNumRows(); i++)
			{
				for(int j=0; j<(layout.getNumColumns()-1); j++)
					os << FMT << static_cast<CastType>(tmp[layout.getIndex(i,j,k)]) << spacing;
				os << FMT << static_cast<CastType>(tmp[layout.getIndex(i,layout.getNumColumns()-1,k)]) << std::endl;
			}
		}
		#undef FMT

		os.precision(oldPrecision);

		if(A.getLocation()==DeviceSide)
			delete[] tmp;
		
		return os;
	}

// Array :
	template<typename T, Location l>
	__host__ Array<T,l>::Array(index_t r, index_t c, index_t s)
	 :	Accessor<T,l>(r, c, s)
	{
		allocateMemory();	
	}
	
	template<typename T, Location l>
	__host__ Array<T,l>::Array(const Layout& layout)
	 :	Accessor<T,l>(layout.getMonolithicLayout())
	{
		allocateMemory();
	}

	template<typename T, Location l>
	__host__ Array<T,l>::Array(const T* ptr, index_t r, index_t c, index_t s)
	 :	Accessor<T,l>(r, c, s)
	{
		allocateMemory();
		setData(ptr);
	}

	template<typename T, Location l>
	__host__ Array<T,l>::Array(const T* ptr, const Layout& layout)
	 :	Accessor<T,l>(layout.getMonolithicLayout())
	{
		allocateMemory();
		setData(ptr);
	}

	template<typename T, Location l>
	__host__ Array<T,l>::Array(const Array<T,l>& A)
	 :	Accessor<T,l>(A.getMonolithicLayout())
	{
		allocateMemory();
		(*this) = A;
	}

	template<typename T, Location l>
	template<typename TIn, Location lin>
	__host__ Array<T,l>::Array(const Accessor<TIn,lin>& A)
	 : 	Accessor<T,l>(A.getMonolithicLayout())
	{
		allocateMemory();
		(*this) = A;
	}

	template<typename T, Location l>
	__host__ Array<T,l>::~Array(void)
	{
		switch(l)
		{
			case DeviceSide :
				{
				#ifdef __CUDACC__
					cudaError_t err = cudaSuccess;
					cudaDeviceSynchronize();	
					err = cudaFree(this->ptr);
					if(err!=cudaSuccess)
						throw static_cast<Exception>(CudaExceptionsOffset + err);
				#else
					throw NotSupported;
				#endif
				}
				break;
			case HostSide :
				delete[] this->ptr;
				break;
			default :
				throw InvalidLocation;
		}
		this->ptr = NULL;
	}

	template<typename T, Location l>
	__host__ void Array<T,l>::allocateMemory(void)
	{
		StaticAssert<(l==DeviceSide) || (l==HostSide)>();

		switch(l)
		{
			case DeviceSide :
				{
				#ifdef __CUDACC__
					cudaError_t err = cudaSuccess;
					err = cudaMalloc(reinterpret_cast<void**>(&this->ptr), getNumElements()*sizeof(T));
					if(err!=cudaSuccess)
						throw static_cast<Exception>(CudaExceptionsOffset + err);
					break;
				#else
					throw NotSupported;
				#endif
				}
			case HostSide :
				this->ptr = new T[getNumElements()];
				break;
			default :
				throw InvalidLocation;
		}
	}

	template<typename T, Location l>
	__host__ Accessor<T,l>& Array<T,l>::accessor(void)
	{
		return (*this);
	}

	template<typename T, Location l>
	__host__ const Accessor<T,l>& Array<T,l>::accessor(void) const
	{
		return (*this);
	}

	template<typename T, Location l>
	__host__ Array<T,l>* Array<T,l>::buildFromStream(std::istream& stream, bool convert, size_t maxBufferSize)
	{
		int sourceTypeIndex = 0; // void;
		const Layout layout = Layout::readFromStream(stream, &sourceTypeIndex);
		Array<T,l>* result = new Array<T,l>(layout);
		
		try
		{
			result->readFromStream(stream, convert, maxBufferSize, true, sourceTypeIndex);
		}
		catch(Kartet::Exception& e)
		{
			delete result;
			throw e;
		}
		return result;
	}

	template<typename T, Location l>
	__host__ Array<T,l>* Array<T,l>::buildFromFile(const std::string& filename, bool convert, size_t maxBufferSize)
	{
		std::ifstream file(filename.c_str(), std::ios::in | std::ios::binary);
		if(!file.is_open() || file.fail())
			throw InvalidInputStream;

		Array<T,l>* result = buildFromStream(file, convert, maxBufferSize);
		file.close();
		return result;
	}

} // Namespace Kartet

#endif 

