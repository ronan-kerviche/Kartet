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

/**
	\file    ArrayTools.hpp
	\brief   Array classes implementations.
	\author  R. Kerviche
	\date    November 1st 2009
**/

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
	inline size_t Layout::StaticContainer<T>::streamHeaderLength(void)
	{
		return sizeof(Layout::StaticContainer<T>::streamHeader);
	}

// Layout :
	/**
	\brief Layout constructor.
	\param r Number of rows.
	\param c Number of columns.
	\param s Number of slices.
	\param lc Distance between columns.
	\param ls Distance between slices.
	\param o Offset (not included in pointer calculations).
	**/
	__host__ __device__ inline Layout::Layout(index_t r, index_t c, index_t s, index_t lc, index_t ls, index_t o)
	 : 	nRows(r),
		nColumns(c),
		nSlices(s),
		sColumns(lc),
		sSlices(ls),
		ofs(o)
	{ }

	/**
	\brief Layout constructor.
	\param r Number of rows.
	\param c Number of columns.
	\param s Number of slices.

	The strides are automatically computed, assuming a monolithic layout.
	**/
	__host__ __device__ inline Layout::Layout(index_t r, index_t c, index_t s)
	 :	nRows(r),
		nColumns(c),
		nSlices(s),
		sColumns(r),
		sSlices(r*c),
		ofs(0)
	{ }

	/**
	\brief Layout constructor.
	\param l Original layout to be copied.
	**/
	__host__ __device__ inline Layout::Layout(const Layout& l)
	 : 	nRows(l.nRows),
		nColumns(l.nColumns),
		nSlices(l.nSlices),
		sColumns(l.sColumns),
		sSlices(l.sSlices),
		ofs(l.ofs)
	{ }

	/**
	\brief Test if the layout is valid.

	The layout is considered valid if it meets all of the following criteria : 
	- All of the sizes are larger or equal to 1.
	- The column stride is larger or equal to the number of rows.
	- The slice stride is larger or equal to the number of elements per slice.

	\return True if the layout is valid.
	**/
	__host__ __device__ inline bool Layout::isValid(void) const
	{
		return (nRows>0 && nColumns>0 && nSlices>0 && 
			sColumns>=nRows && 
			sSlices>=(nRows*nColumns));
			//(nColumns!=1 || sColumns==sSlices) &&	- The column stride is equal to the slice stride if there is only one column.
			//(nSlices!=1 || sSlices==numElements())	- The slice stride is equal to the number of elements if there is only one slice.
	}

	/**
	\brief Returns the number of elements.
	\return The number of elements (number or rows x number of columns x number of slices).
	**/
	__host__ __device__ inline index_t Layout::numElements(void) const
	{
		return (nRows * nColumns * nSlices);
	}
	
	/**
	\brief Returns the number of elements per slice.
	\return The number of elements per slice (number or rows x number of columns).
	**/
	__host__ __device__ inline index_t Layout::numElementsPerSlice(void) const
	{
		return (nRows * nColumns);
	}

	/**
	\brief Returns the number of elements per smallest fragment.
	\return The number of elements in the smallest monoloithic fragment.
	**/
	__host__ __device__ inline index_t Layout::numElementsPerFragment(void) const
	{
		if(sColumns==nRows)
		{
			if(sSlices==(nRows * nColumns))
				return (nRows * nColumns * nSlices);
			else
				return (nRows * nColumns);
		}
		else
			return nRows;
	}

	/**
	\brief Returns the number of rows.
	\return The number of rows.
	**/
	__host__ __device__ inline index_t Layout::numRows(void) const
	{
		return nRows;
	}

	/**
	\brief Returns the number of columns.
	\return The number of columns.
	**/
	__host__ __device__ inline index_t Layout::numColumns(void) const
	{
		return nColumns;
	}

	/**
	\brief Returns the number of slices.
	\return The number of slices.
	**/
	__host__ __device__ inline index_t Layout::numSlices(void) const
	{
		return nSlices;
	}

	/**
	\brief Returns the number of monolithic fragments.
	\return The number of monolithic fragments.
	**/
	__host__ __device__ inline index_t Layout::numFragments(void) const
	{
		if(sColumns==nRows)
		{
			if(sSlices==(nRows * nColumns))
				return 1;
			else
				return nSlices;
		}
		else
			return (nSlices * nColumns);
	}

	/**
	\brief Returns the number of columns.
	\return The number of columns.
	**/
	__host__ __device__ inline index_t Layout::width(void) const
	{
		return nColumns;
	}

	/**
	\brief Returns the number of rows.
	\return The number of rows.
	**/
	__host__ __device__ inline index_t Layout::height(void) const
	{
		return nRows;
	}

	/**
	\brief Returns the number of slices.
	\return The number of slices.
	**/
	__host__ __device__ inline index_t Layout::depth(void) const
	{
		return nSlices;
	}

	/**
	\brief Returns the distance between the beginnings of two consecutive columns.
	\return The distance between the beginnings of two consecutive columns.
	**/
	__host__ __device__ inline index_t Layout::columnsStride(void) const
	{
		return sColumns;
	}

	/**
	\brief Returns the distance between the beginnings of two consecutive slices.
	\return The distance between the beginnings of two consecutive slices.
	**/
	__host__ __device__ inline index_t Layout::slicesStride(void) const
	{
		return sSlices;
	}

	/**
	\brief Returns the offset of the data.
	\return The offset of the data.
	**/
	__host__ __device__ inline index_t Layout::offset(void) const
	{
		return ofs;
	}

	/**
	\brief Modify the offset of the data.
	\param newOffset New offset value.

	This functions does not change in any way pointers to the data. It is purely an indication of the layout.
	
	\return The old offset.
	**/
	__host__ __device__ inline index_t Layout::setOffset(index_t newOffset)
	{
		index_t oldOffset = ofs;
		ofs = newOffset;
		return oldOffset;
	}

	#ifdef __CUDACC__
	/**
	\brief Returns the dimensions of the array in a dim3 struct.
	\return The dimensions of the array in a dim3 struct.
	**/
	__host__ __device__ inline dim3 Layout::dimensions(void) const
	{
		return dim3(nRows, nColumns, nSlices);
	}
	
	/**
	\brief Returns the strides of the array in a dim3 struct.
	
	Note that the first stride is guaranteed to be 1.

	\return The strides of the array in a dim3 struct.
	**/
	__host__ __device__ inline dim3 Layout::strides(void) const
	{
		return dim3(1, sColumns, sSlices);
	}
	#endif

	/**
	\brief Test if the layout is monolithic.
		
	This is true if and only if all of its elements are contiguous (equivalent to the number of fragments being equal to 1).

	\return True if the layout is monolithic.
	**/
	__host__ __device__ inline bool Layout::isMonolithic(void) const
	{
		return (sColumns==nRows || nColumns==1) && (sSlices==(nRows*nColumns) || nSlices==1);
	}

	/**
	\brief Test if all the slices in the layout are monolithic.
		
	This is true if and only if all of the elements inside the slices are contiguous.

	\return True if all the slices in the layout are monolithic.
	**/
	__host__ __device__ inline bool Layout::isSliceMonolithic(void) const
	{
		return (sColumns==nRows || nColumns==1);
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

	\throw InvalidLayoutReinterpretation In case of a failure.
	**/
	__host__ inline void Layout::reinterpretLayout(index_t r, index_t c, index_t s)
	{
		if(!isValid())
			throw InvalidLayout;
		else if(r<=0 || c<=0 || s<=0 || r*c*s!=numElements())
			throw InvalidLayoutReinterpretation;

		const bool 	im = isMonolithic(),
				ism = isSliceMonolithic();
		const index_t	nPerSlice = numElementsPerSlice(),
				newPerSlice = r*c;
		index_t lc = 0,
			ls = 0;

		// Test if the new number of rows is valid : 
		if((r>nRows && !ism) || (r>nPerSlice && !im)) // If it does not go outside a continuous fragment (column if not ism, slice if not im, the last condition is handled in the first test).
			throw InvalidLayoutReinterpretation;
		
		// Find the corresponding distance between columns for this new number of rows :
		if(r==nRows && !ism)
			lc = sColumns;
		else if(r==numElementsPerSlice() && !im)
			lc = sSlices;
		else
			lc = r;

		// If the new slices are monolithic : 
		if(lc==r)
		{
			if((newPerSlice>nRows && !ism) || (newPerSlice>nPerSlice && !im))
				throw InvalidLayoutReinterpretation;
			else if(newPerSlice==nRows && !ism)
				ls = sColumns;
			else if(newPerSlice==nPerSlice && !im)
				ls = sSlices;
			else
				ls = newPerSlice;
		}
		else // if(lc>r) // The new slices are not monolithic
		{
			// Test if it is possible to retract or expand the number of columns : 
			if(c!=nColumns && c>1 && nColumns>1 && s!=1 && nColumns*sColumns!=sSlices)
				throw InvalidLayoutReinterpretation;
			else
				ls = c*lc;
		}

		// Set : 
		nRows		= r;
		nColumns	= c;
		nSlices 	= s;
		sColumns	= lc;
		sSlices		= ls;
	}

	/**
	\brief Modify the layout without changing the number of elements.
	\param other New layout (only the number of rows, columns and slices are considered).

	\throw InvalidLayoutReinterpretation In case of a failure.
	**/
	__host__ inline void Layout::reinterpretLayout(const Layout& other)
	{
		reinterpretLayout(other.nRows, other.nColumns, other.nSlices);
	}

	/**
	\brief Reinterpret the layout by concatenating all slices columns in-place.

	The layout becomes of size (R, C*S, 1).

	\throw InvalidLayoutReinterpretation In case of a failure.
	**/
	__host__ inline void Layout::flatten(void)
	{
		reinterpretLayout(nRows, nColumns*nSlices, 1);
	}

	/**
	\brief Reinterpret the layout by concatenating all columns elements in-place.

	The layout becomes of size (R*C, S, 1).

	\throw InvalidLayoutReinterpretation In case of a failure.
	**/
	__host__ inline void Layout::stretch(void)
	{
		reinterpretLayout(nRows*nColumns, nSlices, 1);
	}

	/**
	\brief Reinterpret the layout by concatenating all elements in-place.
	
	The layout becomes of size (R*C*S, 1, 1).

	\throw InvalidLayoutReinterpretation In case of a failure.
	**/
	__host__ inline void Layout::vectorize(void)
	{
		reinterpretLayout(nRows*nColumns*nSlices, 1, 1);
	}

	/**
	\brief Generates a series of layout, each containing a specified number of columns of the original layout.
	\param jBegin Origin column index.
	\param c Number of columns in each split.

	Note : the number of slices is preserved in the background of each split. The last split might contain less than c columns (gracious).

	\throw Kartet::InvalidSize If the number of columns is strictly less than 1.
	\throw Kartet::OutOfRange If the first targeted column is not in range.
	\return A vector containing all the splits (ordered).
	**/
	__host__ inline std::vector<Layout> Layout::splitLayoutColumns(index_t jBegin, index_t c) const
	{
		std::vector<Layout> pages;

		if(c<1)
			throw InvalidSize;
		else if(!isColumnValid(jBegin))
			throw OutOfRange;
	
		for(index_t j=jBegin; j<nColumns; j+=c)
			pages.push_back(Layout(nRows, std::min(c, nColumns-j), nSlices, sColumns, sSlices, ofs+getPosition(0,j,0)));

		return pages;
	}

	/**
	\brief Generates a series of layout, each containing a specified number of slices of the original layout.
	\param kBegin Origin slice index.
	\param s Number of slices in each split.

	Note : the last split might contain less than s slices (gracious).
	
	\throw Kartet::InvalidSize If the number of slices is strictly less than 1.
	\throw Kartet::OutOfRange If the first targeted slice is not in range.
	\return A vector containing all the splits (ordered).
	**/
	__host__ inline std::vector<Layout> Layout::splitLayoutSlices(index_t kBegin, index_t s) const
	{
		std::vector<Layout> pages;

		if(s<1)
			throw InvalidSize;
		else if(!isSliceValid(kBegin))
			throw OutOfRange;
	
		for(index_t k=kBegin; k<nSlices; k+=s)
			pages.push_back(Layout(nRows, nColumns, std::min(s, nSlices-k), sColumns, sSlices, ofs+getPosition(0,0,k)));

		return pages;
	}
	
	/**
	\brief Generates a series of layout, each containing a specified block layout.
	\param iBegin Origin row index.
	\param jBegin Origin column index.
	\param kBegin Origin slice index.
	\param r Number of rows in the block split. If equals to 0, it will be replaced with the number or rows of the layout.
	\param c Number of columns in the block split. If equals to 0, it will be replaced with the number or columns of the layout.
	\param s Number of slices in the block split. If equals to 0, it will be replaced with the number or slices of the layout.
	
	\throw Kartet::InvalidSize If any of the sizes is strictly less than 1.
	\throw Kartet::OutOfRange If any index is not in range
	\return A vector containing all the splits (column-major ordered).
	**/
	__host__ inline std::vector<Layout> Layout::splitLayoutSubArrays(index_t iBegin, index_t jBegin, index_t kBegin, index_t r, index_t c, index_t s) const
	{
		std::vector<Layout> pages;

		// Fill missing arguments
		if(r==0)	
			r = numRows();
		if(c==0)
			c = numColumns();
		if(s==0)
			s = numSlices();

		if(r<1 || c<1 ||  s<1)
			throw InvalidSize;
		else if(!isRowValid(iBegin) || !isColumnValid(jBegin) ||  !isSliceValid(kBegin))
			throw OutOfRange;
			
		for(index_t k=kBegin; k<nSlices; k+=s)
			for(index_t j=jBegin; j<nColumns; j+=c)
				for(index_t i=iBegin; i<nRows; i+=r)
					pages.push_back(Layout(std::min(r, nRows-i), std::min(c, nColumns-j), std::min(s, nSlices-k), sColumns, sSlices, ofs+getPosition(i,j,k)));

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
		return splitLayoutSubArrays(iBegin, jBegin, kBegin, layout.numRows(), layout.numColumns(), layout.numSlices());
	}

	/**
	\brief Test if this layout is identical to another layout.
	\param other Other layout object to be compared with.
	\return True if this layout has the same sizes and same strides as the other layout.
	**/
	__host__ __device__ inline bool Layout::sameLayoutAs(const Layout& other) const
	{
		return (nRows==other.nRows && nColumns==other.nColumns && nSlices==other.nSlices && sColumns==other.sColumns && sSlices==other.sSlices); 
	}

	/**
	\brief Test if the layout a slice from this object is identical to those of another layout.
	\param other Other layout object to be compared with.
	\return True if the slice layout of this object has the same sizes and same strides as for the other layout.
	**/
	__host__ __device__ inline bool Layout::sameSliceLayoutAs(const Layout& other) const
	{
		return (nRows==other.nRows && nColumns==other.nColumns && sColumns==other.sColumns); 
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

	Note : this function does not perform boundary checking.
	
	\return The normalized row index (i/nRows).
	**/
	template<typename TOut>
	__host__ __device__ inline TOut Layout::getINorm(index_t i) const
	{
		return static_cast<TOut>(i)/static_cast<TOut>(nRows);
	}

	/**
	\brief Get the normalized column index.
	\param j The original column index.

	Note : this function does not perform boundary checking.
	
	\return The normalized column index (j/nColumns).
	**/
	template<typename TOut>
	__host__ __device__ inline TOut Layout::getJNorm(index_t j) const
	{
		return static_cast<TOut>(j)/static_cast<TOut>(nColumns);
	}

	/**
	\brief Get the normalized slice index.
	\param k The original slice index.

	Note : this function does not perform boundary checking.
	
	\return The normalized slice index (k/nSlices).
	**/
	template<typename TOut>
	__host__ __device__ inline TOut Layout::getKNorm(index_t k) const
	{
		return static_cast<TOut>(k)/static_cast<TOut>(nSlices);
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

	Note : this function does not perform boundary checking.
	
	\return The inclusive normalized row index (i/(nRows-1)).
	**/
	template<typename TOut>
	__host__ __device__ inline TOut Layout::getINormIncl(index_t i) const
	{
		return static_cast<TOut>(i)/static_cast<TOut>(nRows-1);
	}

	/**
	\brief Get the inclusive normalized column index.
	\param j The original column index.

	Note : this function does not perform boundary checking.
	
	\return The inclusive normalized column index (i/(nColumns-1)).
	**/
	template<typename TOut>
	__host__ __device__ inline TOut Layout::getJNormIncl(index_t j) const
	{
		return static_cast<TOut>(j)/static_cast<TOut>(nColumns-1);
	}

	/**
	\brief Get the inclusive normalized slice index.
	\param k The original slice index.

	Note : this function does not perform boundary checking.
	
	\return The inclusive normalized slice index (i/(nSlices-1)).
	**/
	template<typename TOut>
	__host__ __device__ inline TOut Layout::getKNormIncl(index_t k) const
	{
		return static_cast<TOut>(k)/static_cast<TOut>(nSlices-1);
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
		return min( max(static_cast<index_t>(0), i), nRows-1);
	}

	/**
	\brief Get the clamped column index.
	\param j The original column index.
	\return The column index clamped to the range [0, nColumns-1].
	**/
	__host__ __device__ inline index_t Layout::getJClamped(index_t j) const
	{
		return min( max(static_cast<index_t>(0), j), nColumns-1);
	}

	/**
	\brief Get the clamped slice index.
	\param k The original slice index.
	\return The slice index clamped to the range [0, nSlices-1].
	**/
	__host__ __device__ inline index_t Layout::getKClamped(index_t k) const
	{
		return min( max(static_cast<index_t>(0), k), nSlices-1);
	}

	/**
	\brief Get the wrapped row index.
	\param i The original row index.
	\return The row index wraped to the range [0, nRows-1] (not mirrored).
	**/
	__host__ __device__ inline index_t Layout::getIWrapped(index_t i) const
	{
		return (i % nRows);
	}

	/**
	\brief Get the wrapped column index.
	\param j The original column index.
	\return The column index wraped to the range [0, nColumns-1] (not mirrored).
	**/
	__host__ __device__ inline index_t Layout::getJWrapped(index_t j) const
	{
		return (j % nColumns);
	}

	/**
	\brief Get the wrapped slice index.
	\param k The original slice index.
	\return The slice index wraped to the range [0, nSlices-1] (not mirrored).
	**/
	__host__ __device__ inline index_t Layout::getKWrapped(index_t k) const
	{
		return (k % nSlices);
	}

	/**
	\brief Get the index of an element from coordinates.
	\param i The row index.
	\param j The column index.
	\param k The slice index.

	The index does not take stides into account. For a monolithic array index and position are the same.
	Note : the function does not perform boundary checking.

	\return The index of the element corresponding to the coordinates.
	**/
	__host__ __device__ inline index_t Layout::getIndex(index_t i, index_t j, index_t k) const
	{
		return k*nSlices + j*nColumns + i;
	}

	/**
	\brief Get the position offset of an element from coordinates : 
	\param i The row index.
	\param j The column index.
	\param k The slice index.

	The position takes stides into account. For a monolithic array index and position are the same.
	Note : the function does not perform boundary checking.

	\return The position of the element corresponding to the coordinates.
	**/
	__host__ __device__ inline index_t Layout::getPosition(index_t i, index_t j, index_t k) const
	{
		return k*sSlices + j*sColumns + i;
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

	/**
	\brief Get the position of the current element.
	\return The position of the current element.
	**/
	__device__ inline index_t Layout::getPosition(void) const
	{
		return getPosition(getI(), getJ(), getK());
	}
	#endif

	/**
	\brief Get the position of an element from coordinates, after fftshift. Modify the arguments.
	\param i The row index. Will be modified.
	\param j The column index. Will be modified.
	\param k The slice index. Will be modified.

	Note the function does not perform boundary checking.

	\return The position of the element corresponding to the fftshift coordinates.
	**/
	__host__ __device__ inline index_t Layout::getPositionFFTShift(index_t& i, index_t& j, index_t& k) const
	{
		const index_t	hi = nRows % 2,
				hj = nColumns % 2;

		if(i<(nRows-hi)/2) 	j = j + (nRows+hi)/2;
		else			j = j - (nRows-hi)/2;

		if(j<(nColumns-hj)/2) i = i + (nColumns+hj)/2;
		else 			i = i - (nColumns-hj)/2;

		return getPosition(i, j, k);
	}

	/**
	\brief Get the position of an element from coordinates, after ifftshift. Modify the arguments.
	\param i The row index. Will be modified.
	\param j The column index. Will be modified.
	\param k The slice index. Will be modified.

	Note the function does not perform bound checking.

	\return The position of the element corresponding to the ifftshift coordinates.
	**/
	__host__ __device__ inline index_t Layout::getPositionFFTInverseShift(index_t& i, index_t& j, index_t& k) const
	{
		const index_t	hi = nRows % 2,
				hj = nColumns % 2;

		if(i<(nRows+hi)/2) 	i = i + (nRows-hi)/2;
		else			i = i - (nRows+hi)/2;

		if(j<(nColumns+hj)/2) j = j + (nColumns-hj)/2;
		else 			j = j - (nColumns+hj)/2;

		return getPosition(i, j, k);
	}

	/**
	\brief Get the position of an element from coordinates, after clamping.
	\param i The row index.
	\param j The column index.
	\param k The slice index.
	\return The position of the element corresponding to the clamped coordinates.
	**/
	__host__ __device__ inline index_t Layout::getPositionClampedToEdge(index_t& i, index_t& j, index_t& k) const
	{
		i = getIClamped(i);
		j = getJClamped(j);
		k = getKClamped(k);
		return getIndex(i, j, k);
	}

	/**
	\brief Get the position of an element from coordinates, after wrapping.
	\param i The row index.
	\param j The column index.
	\param k The slice index.
	\return The position of the element corresponding to the wrapped coordinates.
	**/
	__host__ __device__ inline index_t Layout::getPositionWarped(index_t& i, index_t& j, index_t& k) const
	{
		i = getIWrapped(i);
		j = getJWrapped(j);
		k = getKWrapped(k);
		return getIndex(i, j, k);
	}

	#ifdef __CUDACC__
	/**
	\brief Get the index of the current element, after fftshift.
	\return The position of the current element, after fftshift.
	**/
	__device__ inline index_t Layout::getPositionFFTShift(void) const
	{
		index_t	i = getI(),
			j = getJ(),
			k = getK();
		return getPositionFFTShift(i, j, k);
	}

	/**
	\brief Get the index of the current element, after inverse fftshift.
	\return The position of the current element, after inverse fftshift.
	**/
	__device__ inline index_t Layout::getPositionFFTInverseShift(void) const
	{
		index_t	i = getI(),
			j = getJ(),
			k = getK();
		return getPositionFFTInverseShift(i, j, k);
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
		return (i>=0 && i<nRows && j>=0 && j<nColumns && k>=0 && k<nSlices);
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
	__host__ __device__ inline bool Layout::isRowValid(index_t i) const	
	{
		return (i>=0 && i<nRows);
	}

	/**
	\brief Test if a column index is valid.
	\param j Column index.
	\return True if the column index is valid.
	**/
	__host__ __device__ inline bool Layout::isColumnValid(index_t j) const
	{
		return (j>=0 && j<nColumns);
	}

	/**
	\brief Test if a slice index is valid.
	\param k Slice index.
	\return True if the slice index is valid.
	**/
	__host__ __device__ inline bool Layout::isSliceValid(index_t k) const
	{
		return (k>=0 && k<nSlices);
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
		const index_t n = nRows*nColumns;
		k = index / n;
		j = (index - k*n) / nRows;
		i = index - k*n - j*nRows;
	}

	/**
	\brief Unpack element position to coordinates.
	\param position Element position.
	\param i Output, row index.
	\param j Output, column index.
	\param k Output, slice index.
	**/
	__host__ __device__ inline void Layout::unpackPosition(index_t position, index_t& i, index_t& j, index_t& k) const
	{
		k = position / sSlices;
		j = (position - k*sSlices) / sColumns;
		i = position - k*sSlices - j*sColumns;
	}

	/**
	\brief Move to next coordinate in the layout.
	\param i Input : current row index. Output : next row index.
	\param j Input : current column index. Output : next column index.
	\param k Input : current slice index. Output : next slice index.
	**/
	__host__ __device__ inline void Layout::moveToNext(index_t& i, index_t& j, index_t& k) const
	{
		// This version is the "protected version" 
		// It will safely warp bad coordinates.
		//i = ((i+1) % nRows);
		//j = (((i==0) ? (j+1) : j) % nColumns);
		//k = ((i==0 && j==0) ? (k+1) : k);

		// This version is the "unprotected version"
		// It is also much faster by avoiding index_t modulos. 
		i = (i+1);
		i = (i>=nRows) ? 0 : i;
		j = (i==0) ? (j+1) : j;
		j = (j>=nColumns) ? 0 : j;
		k = (i==0 && j==0) ? (k+1) : k;
	}

	#ifdef __CUDACC__
	/**
	\brief Get the block size for CUDA computation.
	\return dim3 struct with correct block settings.
	**/
	__host__ inline dim3 Layout::blockSize(void) const
	{
		dim3 d;
		// From inner most dimension (I <-> X, J <-> Y, K <-> Z) :
		d.x = min(StaticContainer<void>::numThreads, nRows);
		d.y = min(StaticContainer<void>::numThreads/d.x, nColumns);
		d.z = min(min(StaticContainer<void>::numThreads/(d.x*d.y), nSlices), StaticContainer<void>::maxZThreads);
		//std::cout << "Layout::getBlockSize : " << d.x << ", " << d.y << ", " << d.z << std::endl;
		return d;
	}

	/**
	\brief Get the grid size for CUDA computation.
	\return dim3 struct with correct grid settings.
	**/
	__host__ inline dim3 Layout::numBlocks(void) const
	{
		dim3 d;
		// From inner most dimension (I <-> X, J <-> Y, K <-> Z) :
		const dim3 b = blockSize();
		d.x = (nRows + b.x - 1)/b.x;
		d.y = (nColumns + b.y - 1)/b.y;
		d.z = (nSlices + b.z - 1)/b.z;
		//std::cout << "Layout::numBlock : " << d.x << ", " << d.y << ", " << d.z << std::endl;
		return d;
	}
	#endif

	/**
	\brief Get the layout corresponding to a column. Retains strides.
	\param includeSlices If true also include the depth of the current layout (number of slices).
	\return The layout of a column.
	**/
	__host__ inline Layout Layout::columnLayout(const bool& includeSlices) const
	{
		if(includeSlices)
			return Layout(nRows, 1, nSlices, sColumns, sSlices);
		else
			return Layout(nRows, 1, 1, sColumns, sSlices);
	}

	/**
	\brief Get the layout corresponding to a slice. Retains strides.
	\return The layout of a slice.
	**/
	__host__ inline Layout Layout::sliceLayout(void) const
	{
		return Layout(nRows, nColumns, 1, sColumns, sSlices);
	}

	/**
	\brief Get the monolithic layout corresponding to this object.
	\return The monolithic layout corresponding to this object.
	**/
	__host__ inline Layout Layout::monolithicLayout(void) const
	{
		return Layout(nRows, nColumns, nSlices);
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
		if(nRows==columnsStride() && numElementsPerSlice()==slicesStride())
			op.apply(*this, *this, ptr, 0, 0, 0, 0);		
		else if(nRows==columnsStride())
		{
			const Layout sl = sliceLayout();
			for(index_t k=0; k<nSlices; k++)
				op.apply(*this, sl, ptr, k*sSlices, 0, 0, k);	
		}
		else
		{
			const Layout cl = columnLayout();
			for(index_t k=0; k<nSlices; k++)
			{
				for(index_t j=0; j<nColumns; j++)
					op.apply(*this, cl, ptr, (k*sSlices + j*sColumns), 0, j, k);
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
		if(!layoutA.monolithicLayout().sameLayoutAs(layoutB.monolithicLayout()))
			throw IncompatibleLayout;

		if(	(layoutA.numRows()==layoutA.columnsStride() && layoutA.numElementsPerSlice()==layoutA.slicesStride()) &&
			(layoutB.numRows()==layoutB.columnsStride() && layoutB.numElementsPerSlice()==layoutB.slicesStride()) )
		{
			op.apply(layoutA, layoutA, ptrA, ptrB, 0, 0, 0, 0, 0);
		}
		else if((layoutA.numRows()==layoutA.columnsStride()) &&
			(layoutB.numRows()==layoutB.columnsStride()) )
		{
			const Layout sliceLayout = layoutA.sliceLayout();
			for(index_t k=0; k<layoutA.numSlices(); k++)
				op.apply(layoutA, sliceLayout, ptrA, ptrB, k*layoutA.slicesStride(), k*layoutB.slicesStride(), 0, 0, k);
		}
		else
		{
			const Layout cl = layoutA.columnLayout();
			for(index_t k=0; k<layoutA.numSlices(); k++)
			{
				for(index_t j=0; j<layoutB.numColumns(); j++)
					op.apply(layoutA, cl, ptrA, ptrB, (k*layoutA.slicesStride() + j*layoutA.columnsStride()), (k*layoutB.slicesStride() + j*layoutB.columnsStride()), 0, j, k);
			}
		}
	}

	/**
	\brief Read Layout from input stream.
	\param stream Input stream.
	\param typeIndex Output, type code.

	If not NULL, the destination will be written with the type code following the layout (single integer).
	
	\throw InsufficientIndexingDepth If the stream refers to data wider than the available indexing depth.
	\return The Layout built from the stream.
	**/
	__host__ inline Layout Layout::readFromStream(std::istream& stream, int* typeIndex)
	{
		if(!stream.good())
			throw InvalidInputStream;
		
		const size_t bufferLength = 32;
		if(bufferLength<StaticContainer<void>::streamHeaderLength())
			throw InvalidOperation;
		char headerBuffer[bufferLength];
		std::memset(headerBuffer, 0, bufferLength);
		stream.read(headerBuffer, StaticContainer<void>::streamHeaderLength()-1);
		if(strncmp(StaticContainer<void>::streamHeader, headerBuffer, StaticContainer<void>::streamHeaderLength()-1)!=0)
			throw InvalidStreamHeader;
	
		// Read the type :
		int dummyType;
		if(typeIndex==NULL)
			typeIndex = &dummyType;
		stream.read(reinterpret_cast<char*>(typeIndex), sizeof(int)); if(!stream.good()) throw InvalidInputStream;

		// Read the sizes :
		index64_t r = 0,
			  c = 0,
			  s = 0;
		stream.read(reinterpret_cast<char*>(&r), sizeof(index64_t)); if(!stream.good()) throw InvalidInputStream;
		stream.read(reinterpret_cast<char*>(&c), sizeof(index64_t)); if(!stream.good()) throw InvalidInputStream;
		stream.read(reinterpret_cast<char*>(&s), sizeof(index64_t)); if(stream.fail()) throw InvalidInputStream; // Do not test for EOF after this read.

		// Cast to the index_t : 
		if(r>=std::numeric_limits<index_t>::max() || c>=std::numeric_limits<index_t>::max() || s>=std::numeric_limits<index_t>::max())
			throw InsufficientIndexingDepth;

		// Return :
		return Layout(static_cast<index_t>(r), static_cast<index_t>(c), static_cast<index_t>(s));
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
		stream.write(StaticContainer<void>::streamHeader, StaticContainer<void>::streamHeaderLength()-1);
		
		// Write the type :	
		stream.write(reinterpret_cast<char*>(&typeIndex), sizeof(int));
	
		// Write the size :
		index64_t _nRows = nRows,
			  _nColumns = nColumns,
			  _nSlices = nSlices;
		stream.write(reinterpret_cast<char*>(&_nRows), sizeof(index64_t));
		stream.write(reinterpret_cast<char*>(&_nColumns), sizeof(index64_t));
		stream.write(reinterpret_cast<char*>(&_nSlices), sizeof(index64_t));
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
	\related Kartet::Layout
	\brief Write layout to ostream object (std::cout, std::cerr, etc.).
	\param os std::ostream object to write to.
	\param layout Layout object.

	Example : 
	\code
	Layout l(3, 5, 7);
	std::cout << l << std::endl;
	\endcode

	Output (strides are replaced by an underscore if elements are contiguous, and not printed if the layout is monolithic) : 
	\code
	['rows', 'columns', 'slices', 'sColumns', 'sSlices', 'offset']
	\endcode

	\return Reference to the modified std::ostream, for chain.
	**/
	__host__ inline std::ostream& operator<<(std::ostream& os, const Layout& layout)
	{
		if(layout.numSlices()==1)
			os << '[' << layout.numRows() << ", " << layout.numColumns();
		else
			os << '[' << layout.numRows() << ", " << layout.numColumns() << ", " << layout.numSlices();

		if(layout.columnsStride()>layout.numRows() || layout.slicesStride()>layout.numElementsPerSlice())
		{
			os << "; ";
			if(layout.columnsStride()==layout.numRows() && layout.numColumns()>1)
				os << '_';
			else
				os << '+' << layout.columnsStride();
			os << ", ";
			if(layout.slicesStride()==layout.numElementsPerSlice() && layout.numSlices()>1)
				os << '_';
			else
				os << '+' << layout.slicesStride();
			os << ", ";
			if(layout.offset()==0)
				os << '_';
			else
				os << '+' << layout.offset();
		}
		os << ']';

		return os;
	}

// Accessor :
	/**
	\brief Accessor constructor.
	\param r Number of rows.
	\param c Number of columns.
	\param s Number of slices.
	\param lc Distance between two consecutive columns.
	\param ls Distance between two consecutive slices.
	\param o Pointer offset.
	**/
	template<typename T, Location l>
	__host__ __device__ Accessor<T,l>::Accessor(index_t r, index_t c, index_t s, index_t lc, index_t ls, index_t o)
	 : 	Layout(r, c, s, lc, ls, o),
		ptr(NULL)	
	{ }

	/**
	\brief Accessor constructor.
	\param r Number of rows.
	\param c Number of columns.
	\param s Number of slices.
	
	Assumes monolithic layout.
	**/
	template<typename T, Location l>
	__host__ __device__ Accessor<T,l>::Accessor(index_t r, index_t c, index_t s)
	 :	Layout(r, c, s),
		ptr(NULL)
	{ }

	/**
	\brief Accessor constructor.
	\param layout Layout of the accessor.
	**/
	template<typename T, Location l>
	__host__ __device__ Accessor<T,l>::Accessor(const Layout& layout)
	 : 	Layout(layout), 
		ptr(NULL)
	{ }

	/**
	\brief Accessor constructor.
	\param ptr Data pointer.
	\param r Number of rows.
	\param c Number of columns.
	\param s Number of slices.
	\param lc Distance between two consecutive columns.
	\param ls Distance between two consecutive slices.
	\param o Pointer offset.

	The data pointer provided will not be released by the accessor.
	**/
	template<typename T, Location l>
	__host__ __device__ Accessor<T,l>::Accessor(T* ptr, index_t r, index_t c, index_t s, index_t lc, index_t ls, index_t o)
	 :	Layout(r, c, s, lc, ls, o),
		ptr(ptr)
	{ }

	/**
	\brief Accessor constructor.
	\param ptr Data pointer.
	\param r Number of rows.
	\param c Number of columns.
	\param s Number of slices.
	
	Assumes monolithic layout.
	**/
	template<typename T, Location l>
	__host__ __device__ Accessor<T,l>::Accessor(T* ptr, index_t r, index_t c, index_t s)
	 :	Layout(r, c, s),
		ptr(ptr)
	{ }

	/**
	\brief Accessor constructor.
	\param ptr Data pointer.
	\param layout Layout of the accessor.
	
	The data pointer provided will not be released by the accessor.
	**/
	template<typename T, Location l>
	__host__ __device__ Accessor<T,l>::Accessor(T* ptr, const Layout& layout)
	 :	Layout(layout),
		ptr(ptr)
	{ }
	
	/**
	\brief Accessor constructor.
	\param a Array to be accessed.

	The accessor will shadow the array provided.
	**/
	template<typename T, Location l>
	__host__ Accessor<T,l>::Accessor(const Array<T,l>& a)
	 :	Layout(a), 
		ptr(a.ptr)
	{ }

	/**
	\brief Accessor constructor.
	\param a Another accessor.

	The accessor will shadow the accessor provided.
	**/
	template<typename T, Location l>
	__host__ __device__ Accessor<T,l>::Accessor(const Accessor<T,l>& a)
	 : 	Layout(a),
		ptr(a.ptr)
	{ }

	/**
	\brief Get the location of the data manipulated by this accessor.
	\return Location value (see Kartet::Location).
	**/
	template<typename T, Location l>
	__host__ __device__ Location Accessor<T,l>::location(void) const
	{
		return l;
	}

	/**
	\brief Get the underlying data pointer (might be on device side).
	\return The data pointer.
	**/
	template<typename T, Location l>
	__host__ __device__ T* Accessor<T,l>::dataPtr(void) const
	{
		return ptr;
	}

	/**
	\brief Get the size of the data manipulated.
	\return The size of the layout in bytes.
	**/
	template<typename T, Location l>
	__host__ __device__ size_t Accessor<T,l>::size(void) const
	{
		return static_cast<size_t>(numElements())*sizeof(T);
	}

	/**
	\brief Access the underlying data directly.
	\param i Row index.
	\param j Column index.
	\param k Slice index.
	
	Warnings : this function does not perform bound checking. If the data is on the device side, the reference is invalid when used from the host side.

	\return A reference to the data at the given coordinates.
	**/
	template<typename T, Location l>
	__host__ __device__ inline T& Accessor<T,l>::data(index_t i, index_t j, index_t k) const
	{
		return ptr[getPosition(i, j, k)];
	}

	/**
	\brief Access the underlying data directly.
	\param p Element index.
	
	Warnings : this function does not perform bound checking. This function will likely return wrong reference for non monolithic layout. If the data is on the device side, the reference is invalid when used from the host side.

	\return A reference to the data at the given index.
	**/
	template<typename T, Location l>
	__host__ __device__ inline T& Accessor<T,l>::data(index_t p) const
	{
		return ptr[p];
	}

	#ifdef __CUDACC__
	/**
	\brief Access the underlying data directly, at the current coordinates.
	\return A reference to the data at the current coordinates.
	**/
	template<typename T, Location l>
	__device__ inline T& Accessor<T,l>::data(void) const
	{
		return ptr[getPosition()];
	}

	/**
	\brief Access the underlying data directly, at the current coordinates, in another slice.
	\param k Slice index.
	\return A reference to the data at the current coordinates, in the specified slice.
	**/
	template<typename T, Location l>
	__device__ inline T& Accessor<T,l>::dataInSlice(int k) const
	{
		return ptr[getPosition(getI(),getJ(),k)];
	}

	/**
	\brief Access the underlying data directly, at the current fftshift coordinates.
	\return A reference to the data at the current fftshift coordinates.
	**/
	template<typename T, Location l>
	__device__ inline T& Accessor<T,l>::dataFFTShift(void) const
	{
		return ptr[getPositionFFTShift()];
	}

	/**
	\brief Access the underlying data directly, at the current inverse fftshift coordinates.
	\return A reference to the data at the current inverse fftshift coordinates.
	**/
	template<typename T, Location l>
	__device__ inline T& Accessor<T,l>::dataFFTInverseShift(void) const
	{
		return ptr[getPositionFFTInverseShift()];
	}
	#endif

	/**
	\brief Get a copy of the data (regardless of its location).
	
	The user takes the responsability of releasing the memory.

	\return A copy of the data in the host side.
	**/
	template<typename T, Location l>
	T* Accessor<T,l>::getData(void) const
	{	
		T* ptr = new T[numElements()];
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
				solidLayout(_originalLayout.monolithicLayout())
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
					toOffset	= solidLayout.getPosition(i, j, k);
					fromOffset	= originalLayout.getPosition(i, j, k);
				}
				else
				{
					toOffset	= originalLayout.getPosition(i, j, k);
					fromOffset	= solidLayout.getPosition(i, j, k);
				}

				#ifdef __CUDACC__
					const cudaMemcpyKind _direction = getCudaDirection(direction);
					cudaError_t err = cudaMemcpy((to + toOffset), (from + fromOffset), currentAccessLayout.numElements()*sizeof(T), _direction);
					if(err!=cudaSuccess)
						throw static_cast<Exception>(CudaExceptionsOffset + err);
				#else
					memcpy((to + toOffset), (from + fromOffset), currentAccessLayout.numElements()*sizeof(T));
				#endif
			}
		};

	/**
	\brief Copy the data (read, regardless of its location).
	\param dst Pointer to destination memory. The space must be sufficient.
	\param lout Location of the memory to be written.
	**/
	template<typename T, Location l>
	void Accessor<T,l>::getData(T* dst, const Location lout) const
	{
		if(dst==NULL)
			throw NullPointer;
	
		const Direction direction = getDirection(l, lout);
		MemCpyToolBox<T> toolbox(direction, dst, ptr, *this);
		singleScan(toolbox);
	}

	/**
	\brief Copy the data (write, regardless of its location).
	\param src Pointer to source memory.
	\param lin Location of the memory to be read.
	**/
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
					stream.read(reinterpret_cast<char*>(ptr + offset), currentAccessLayout.numElements()*sizeof(T));
					if(stream.fail())
						throw InvalidInputStream;
				}
				else
				{
					for(index_t offsetCopied=0; offsetCopied<currentAccessLayout.numElements(); )
					{
						const index_t currentCopyNumElements = std::min(currentAccessLayout.numElements()-offsetCopied, static_cast<index_t>(numBufferElements));
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
						else
							UNUSED_PARAMETER(tmpSrc)							

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

	/**
	\brief Read data from stream.
	\param stream Input stream.
	\param convert True if the data from the stream must be converted to the type of the accessor.
	\param maxBufferSize In the case of a conversion, the 
	\param skipHeader Skip header read if True. The data read must be of the same type. No conversion will be performed.
	\param sourceTypeIndex Force the data type of the input stream (useful if the header is skipped).
	\throw Kartet::InvalidOperation If maxBufferSize is insufficient or if the source type is different than the accessor type and the conversion is disabled.
	\throw Kartet::InvalidInputStream If the stream cannot be read.
	\throw Kartet::IncompatibleLayout If the layouts of the source and the accessor are not compatible.
	**/
	template<typename T, Location l>
	__host__ void Accessor<T,l>::readFromStream(std::istream& stream, bool convert, const size_t maxBufferSize, const bool skipHeader, int sourceTypeIndex)
	{
		if(maxBufferSize<sizeof(T))
			throw InvalidOperation;
		if(!stream.good())
			throw InvalidInputStream;
		
		Layout lt = layout();
		if(!skipHeader)
			lt = Layout::readFromStream(stream, &sourceTypeIndex);
		if(!lt.sameLayoutAs(*this))
			throw IncompatibleLayout;
	
		const bool conversion = (sourceTypeIndex!=GetIndex<TypesSortedByAccuracy, T>::value);
		if(!convert && conversion)
			throw InvalidOperation;
		
		const size_t 	sourceTypeSize = sizeOfType(sourceTypeIndex),
				maxSize = conversion ? (sourceTypeSize + sizeof(T)) : sizeof(T),
				numBufferElements = std::min(static_cast<size_t>(lt.numElements())*maxSize, maxBufferSize)/maxSize;
		
		StreamInputToolBox<T> toolbox(stream, l, sourceTypeIndex, sourceTypeSize, numBufferElements);
		singleScan(toolbox);
	}

	/**
	\brief Read data from file.
	\param filename Filename to load.
	\param convert True if the data from the stream must be converted to the type of the accessor.
	\param maxBufferSize In the case of a conversion, the 
	\throw Kartet::InvalidOperation If maxBufferSize is insufficient or if the source type is different than the accessor type and the conversion is disabled.
	\throw Kartet::InvalidInputStream If the stream cannot be read.
	\throw Kartet::IncompatibleLayout If the layouts of the source and the accessor are not compatible.
	**/
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
						for(index_t offsetCopied=0; offsetCopied<currentAccessLayout.numElements(); )
						{
							const index_t currentCopyNumElements = std::min(currentAccessLayout.numElements()-offsetCopied, static_cast<index_t>(numBufferElements));
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
					stream.write(reinterpret_cast<char*>(ptr + offset), currentAccessLayout.numElements()*sizeof(T));
					if(stream.fail())
						throw InvalidOutputStream;
				}
			}
		};

	/**
	\brief Write data to stream (including layout header).
	\param stream Stream to write to.
	\param maxBufferSize Buffer size used for transfers from the device.
	**/
	template<typename T, Location l>
	__host__ void Accessor<T,l>::writeToStream(std::ostream& stream, const size_t maxBufferSize)
	{
		if(maxBufferSize<sizeof(T))
			throw InvalidOperation;

		const size_t numBufferElements = std::min(static_cast<size_t>(numElements())*sizeof(T), maxBufferSize)/sizeof(T);

		// Write the header :
		Layout::writeToStream<T>(stream);
		StreamOutputToolBox<T> toolbox(stream, l, numBufferElements);
		singleScan(toolbox);
	}

	/**
	\brief Write data to file (including layout header).
	\param filename Filename to write to.
	\param maxBufferSize Buffer size used for transfers from the device.
	**/
	template<typename T, Location l>
	__host__ void Accessor<T,l>::writeToFile(const std::string& filename, const size_t maxBufferSize)
	{
		std::ofstream file(filename.c_str(), std::ios::out | std::ios::binary);
		if(!file.is_open() || file.fail())
			throw InvalidOutputStream;

		writeToStream(file, maxBufferSize);
		file.close();
	}

	/**
	\brief Get the layout of this accessor.
	\return The layout, include stride parameters.
	**/
	template<typename T, Location l>
	__host__ const Layout& Accessor<T,l>::layout(void) const
	{
		return (*this);
	}

	/**
	\brief Get an accessor to the specified element.
	\param i Row index.
	\param j Column index.
	\param k Slice index.

	Note that this method can be used to modify a single element of data on either host or device side, but is slow.

	\throw Kartet::OutOfRange If the coordinates are not valid.
	\return An accessor for a single element.
	**/
	template<typename T, Location l>
	__host__ Accessor<T,l> Accessor<T,l>::element(index_t i, index_t j, index_t k) const
	{
		if(!isInside(i, j, k))
			throw OutOfRange;		

		return Accessor<T,l>(ptr + getPosition(i, j, k), 1, 1, 1, 1, 1, offset()+getPosition(i, j, k));
	}

	/**
	\brief Get an accessor to the specified elements.
	\param p Starting index.
	\param n Number of elements.
	\throw Kartet::OutofRange If the starting index is not valid or if the number of elements is larger than the number of contiguous elements.
	\return An accessor to the contiguous elements.
	**/
	template<typename T, Location l>
	__host__ Accessor<T,l> Accessor<T,l>::elements(index_t p, index_t n) const
	{
		if(p<0 || n<0 || p>=numElements() || (p+n)>numElements())
			throw OutOfRange;

		// Test if the block is fully in this accessor :
		if(!isMonolithic())
		{
			index_t i = 0,
				j = 0,
				k = 0;
			unpackPosition(p, i, j, k);
			if(numRows()!=columnsStride())
			{
				if(n>numRows())
					throw OutOfRange;
			}
			else if(numElementsPerSlice()!=columnsStride())
			{
				if(n>numElementsPerSlice())
					throw OutOfRange;
			}
		}
		return Accessor<T,l>(ptr + p, n, 1, 1, n, n, offset()+p);
	}

	/**
	\brief Get an accessor to all the underlying elements.
	\throw Kartet::InvalidOperation If the accessor is not monolithic.
	\returns An accessor to all the elements.
	**/
	template<typename T, Location l>
	__host__ Accessor<T,l> Accessor<T,l>::elements(void) const
	{
		if(!isMonolithic())
			throw InvalidOperation;
		return Accessor<T,l>(ptr, numElements(), 1, 1, numElements(), numElements(), offset());
	}

	/**
	\brief Get an accesor to a particular column.
	\param j Column index.
	\throw Kartet::OutOfRange If the column index is out of range.
	\return An accessor to the specified column.
	**/
	template<typename T, Location l>
	__host__ Accessor<T,l> Accessor<T,l>::column(index_t j) const
	{
		if(!isColumnValid(j))
			throw OutOfRange;

		return Accessor<T,l>(ptr + getPosition(0,j,0), numRows(), 1, numSlices(), numRows(), slicesStride(), offset()+getPosition(0,j,0));
	}

	/**
	\brief Get an accessor to the last column.
	\return An accessor to the last column.
	**/
	template<typename T, Location l>
	__host__ Accessor<T,l> Accessor<T,l>::endColumn(void) const
	{
		return column(numColumns()-1);
	}

	/**
	\brief Get an accessor to the specified columns.
	\param jBegin Starting index of the column.
	\param c Number of columns.
	\param jStep Step between two columns.
	\throw Kartet::InvalidNegativeStep If the number of columns is less or equal to 0 or the step is less or equal to 0.
	\throw Kartet::OutOfRange If any column has an invalid index.
	\return An accessor to the specified columns.
	**/
	template<typename T, Location l>
	__host__ Accessor<T,l> Accessor<T,l>::columns(index_t jBegin, index_t c, index_t jStep) const
	{
		if(jStep<=0 || c<=0)
			throw InvalidNegativeStep;
		if(!isColumnValid(jBegin) || !isColumnValid(jBegin+(c-1)*jStep))
			throw OutOfRange;

		return Accessor<T,l>(ptr + getPosition(0,jBegin,0), numRows(), c, numSlices(), jStep*columnsStride(), slicesStride(), offset()+getPosition(0,jBegin,0));
	}

	/**
	\brief Get an accessor to a slice.
	\param k Slice index.
	\throw Kartet::OutOfRange If the index of the slice is invalid.
	\return An accessor to the required slice.
	**/
	template<typename T, Location l>
	__host__ Accessor<T,l> Accessor<T,l>::slice(index_t k) const
	{
		if(!isSliceValid(k))
			throw OutOfRange;

		return Accessor<T,l>(ptr + getPosition(0,0,k), numRows(), numColumns(), 1, columnsStride(), offset()+getPosition(0,0,k));
	}

	/**
	\brief Get an accessor to the last slice.
	\return An accessor to the last slice.
	**/
	template<typename T, Location l>
	__host__ Accessor<T,l> Accessor<T,l>::endSlice(void) const
	{
		return slice(numSlices()-1);
	}

	/**
	\brief Get an accessor to multiple slices.
	\param kBegin Starting index of the slices.
	\param s Number of slices.
	\param kStep Step between two slices.
	\throw Kartet::InvalidNegativeStep If the number of slices is less or equal to 0 or the step is less or equal to 0.
	\throw Kartet::OutOfRange If any slice has an invalid index.
	\return An accessor to the specified slices.
	**/
	template<typename T, Location l>
	__host__ Accessor<T,l> Accessor<T,l>::slices(index_t kBegin, index_t s, index_t kStep) const
	{
		if(kStep<0 || s<=0)
			throw InvalidNegativeStep;
		if(!isSliceValid(kBegin) || !isSliceValid(kBegin+(s-1)*kStep))
			throw OutOfRange;

		return Accessor<T,l>(ptr + getPosition(0,0,kBegin), numRows(), numColumns(), s, columnsStride(), kStep*slicesStride(), offset()+getPosition(0,0,kBegin));
	}

	/**
	\brief Get an accessor to a sub-array.
	\param iBegin Starting row index.
	\param jBegin Starting column index.
	\param r Number of rows.
	\param c Number of columns.
	\throw Kartet::InvalidNegativeStep If the number of rows or columns is less or equal to 0.
	\throw Kartet::OutOfRange If any index is invalid.
	\return An accessor to the specified sub-array.
	**/
	template<typename T, Location l>
	__host__ Accessor<T,l> Accessor<T,l>::subArray(index_t iBegin, index_t jBegin, index_t r, index_t c) const
	{
		if(r<=0 || c<=0)
			throw InvalidNegativeStep;
		if(!isRowValid(iBegin) || !isRowValid(iBegin+r-1) || !isColumnValid(jBegin) || !isColumnValid(jBegin+c-1))
			throw OutOfRange;
		
		return Accessor<T,l>(ptr + getPosition(iBegin,jBegin,0), r, c, numSlices(), columnsStride(), slicesStride(), offset()+getPosition(iBegin,jBegin,0));
	}

	/**
	\brief Get an accessor to a sub-array.
	\param iBegin Starting row index.
	\param jBegin Starting column index.
	\param kBegin Starting slice index.
	\param r Number of rows.
	\param c Number of columns.
	\param s Number of slices.
	\throw Kartet::InvalidNegativeStep If the number of rows, or columns; or slices is less or equal to 0.
	\throw Kartet::OutOfRange If any index is invalid.
	\return An accessor to the specified sub-array.
	**/
	template<typename T, Location l>
	__host__ Accessor<T,l> Accessor<T,l>::subArray(index_t iBegin, index_t jBegin, index_t kBegin, index_t r, index_t c, index_t s) const
	{
		if(r<=0 || c<=0 || s<=0)
			throw InvalidNegativeStep;
		if(!isRowValid(iBegin) || !isRowValid(iBegin+r-1) || !isColumnValid(jBegin) || !isColumnValid(jBegin+c-1) || !isSliceValid(kBegin+s-1))
			throw OutOfRange;

		return Accessor<T,l>(ptr + getPosition(iBegin,jBegin,kBegin), r, c, s, columnsStride(), slicesStride(), offset()+getPosition(iBegin,jBegin,kBegin));
	}

	/**
	\brief Return the flattened version of this accessor (see Kartet::Layout::flatten()).
	\return The flattened version of this accessor.
	**/
	template<typename T, Location l>
	__host__ Accessor<T,l> Accessor<T,l>::flattened(void) const
	{
		Accessor<T, l> result = (*this);
		result.flatten();
		return result;
	}

	/**
	\brief Return the stretched version of this accessor (see Kartet::Layout::stretch()).
	\return The stretched version of this accessor.
	**/
	template<typename T, Location l>
	__host__ Accessor<T,l> Accessor<T,l>::stretched(void) const
	{
		Accessor<T, l> result = (*this);
		result.stretch();
		return result;
	}

	/**
	\brief Return the vectorized version of this accessor (see Kartet::Layout::vectorize()).
	\return The vectorized version of this accessor.
	**/
	template<typename T, Location l>
	__host__ Accessor<T,l> Accessor<T,l>::vectorized(void) const
	{
		Accessor<T, l> result = (*this);
		result.vectorize();
		return result;
	}

	/**
	\brief Split columns into several accessors.
	\param jBegin Starting index of the columns.
	\param c Number of columns per sub-accessor.
	\return A standard vector containing the sub-accessors (ordered).
	**/
	template<typename T, Location l>
	__host__  std::vector< Accessor<T,l> > Accessor<T,l>::splitColumns(index_t jBegin, index_t c) const
	{
		std::vector< Accessor<T,l> > pages;
		Layout tmp = this->layout();
		tmp.setOffset(0);
		const std::vector<Layout> pagesLayout = tmp.splitLayoutColumns(jBegin, c);

		for(std::vector<Layout>::const_iterator it=pagesLayout.begin(); it!=pagesLayout.end(); it++)	
		{
			pages.push_back( Accessor<T,l>(ptr + it->offset(), *it) );
			pages.back().setOffset(offset() + pages.back().offset());
		}

		return pages;
	}

	/**
	\brief Split slices into several accessors.
	\param kBegin Starting index of the slices.
	\param s Number of slices per sub-accessor.
	\return A standard vector containing the sub-accessors (ordered).
	**/
	template<typename T, Location l>
	__host__  std::vector< Accessor<T,l> > Accessor<T,l>::splitSlices(index_t kBegin, index_t s) const
	{
		std::vector< Accessor<T,l> > pages;
		Layout tmp = this->layout();
		tmp.setOffset(0);
		const std::vector<Layout> pagesLayout = tmp.splitLayoutSlices(kBegin, s);

		for(std::vector<Layout>::const_iterator it=pagesLayout.begin(); it!=pagesLayout.end(); it++)	
		{
			pages.push_back( Accessor<T,l>(ptr + it->offset(), *it) );
			pages.back().setOffset(offset() + pages.back().offset());
		}

		return pages;
	}

	/**
	\brief Split sub-arrays into several accessors.	
	\param iBegin Starting index of the rows.
	\param jBegin Starting index of the columns.
	\param kBegin Starting index of the columns.
	\param r Number of rows per sub-accessor.
	\param c Number of columns per sub-accessor.
	\param s Number of slices per sub-accessor.
	\return A standard vector containing the sub-accessors (ordered).
	**/
	template<typename T, Location l>
	__host__ std::vector< Accessor<T,l> > Accessor<T,l>::splitSubArrays(index_t iBegin, index_t jBegin, index_t kBegin, index_t r, index_t c, index_t s) const
	{
		std::vector< Accessor<T,l> > pages;
		Layout tmp = this->layout();
		tmp.setOffset(0);
		const std::vector<Layout> pagesLayout = tmp.splitLayoutSubArrays(iBegin, jBegin, kBegin, r, c, s);

		for(std::vector<Layout>::const_iterator it=pagesLayout.begin(); it!=pagesLayout.end(); it++)	
		{
			pages.push_back( Accessor<T,l>(ptr + it->offset(), *it) );
			pages.back().setOffset(offset() + pages.back().offset());
		}

		return pages;
	}

	/**
	\brief Split sub-arrays into several accessors.	
	\param iBegin Starting index of the rows.
	\param jBegin Starting index of the columns.
	\param kBegin Starting index of the columns.
	\param layout Layout of the sub-arrays.
	\return A standard vector containing the sub-accessors (ordered).
	**/
	template<typename T, Location l>
	__host__ std::vector< Accessor<T,l> > Accessor<T,l>::splitSubArrays(index_t iBegin, index_t jBegin, index_t kBegin, const Layout& layout) const
	{
		return Accessor<T,l>::splitSubArrays(iBegin, jBegin, kBegin, layout.numRows(), layout.numColumns(), layout.numSlices());
	}

	/**
	\brief Perform a single scan operation on the accessor.
	\param op Operator object (see Layout::singleScan for more information).
	**/
	template<typename T, Location l>
	template<class Op>
	__host__ void Accessor<T,l>::singleScan(const Op& op) const
	{
		Layout::singleScan(ptr, op);
	}

	/**
	\related Kartet::Accessor
	\brief Output accessor data on stream.
	\param os Output stream (std::cout, std::cerr, etc.)
	\param A Accessor object.
	
	Example : 
	\code
	Kartet::Array<int> a(4, 4);
	a = Kartet::IndexI() + Kartet::IndexJ();
	std::cout << "Array data : " << a << std::endl;
	std::cout << "Second vector : " << a.column(1) << std::endl;
	std::cout << "Last two vectors : " << a.columns(2,2) << std::endl;
	\endcode

	\return Output stream object for follow-up.
	**/
	template<typename T, Location l>
	__host__ std::ostream& operator<<(std::ostream& os, const Accessor<T,l>& A)
	{
		typedef typename StaticIf<SameTypes<T, char>::test || SameTypes<T, unsigned char>::test, int, T>::TValue CastType;

		#define FMT std::right << std::setfill(fillCharacter) << std::setw(maxWidth)
		const int precision = 4,
			  maxWidth = precision+3;
		const char fillCharacter = ' ';
		const char* spacing = "  ";
		const Kartet::Layout layout = (A.location()==DeviceSide) ? A.monolithicLayout() : A.layout();
		T* tmp = (A.location()==DeviceSide) ? A.getData() : A.dataPtr();

		// Get old parameter :
		const int oldPrecision = os.precision(precision);

		os << "(Array of size " << A.layout() << ", " << ((A.location()==DeviceSide) ? "DeviceSide" : "HostSide") << ')' << std::endl;
		for(int k=0; k<layout.numSlices(); k++)
		{
			if(layout.numSlices()>1)
				os << "Slice " << k << " : "<< std::endl;
	
			for(int i=0; i<layout.numRows(); i++)
			{
				for(int j=0; j<(layout.numColumns()-1); j++)
					os << FMT << static_cast<CastType>(tmp[layout.getPosition(i,j,k)]) << spacing;
				os << FMT << static_cast<CastType>(tmp[layout.getPosition(i,layout.numColumns()-1,k)]) << std::endl;
			}
		}
		#undef FMT

		os.precision(oldPrecision);

		if(A.location()==DeviceSide)
			delete[] tmp;
		
		return os;
	}

// Array :
	/**
	\brief Array constructor.
	\param r Number of rows.
	\param c Number of columns.
	\param s Number of slices.
	\throw Kartet::InvalidLayout If the layout is invalid (incorrect sizes).
	**/
	template<typename T, Location l>
	__host__ Array<T,l>::Array(index_t r, index_t c, index_t s)
	 :	Accessor<T,l>(r, c, s)
	{
		if(!layout().isValid())
			throw InvalidLayout;
		allocateMemory();	
	}
	
	/**
	\brief Array constructor.
	\param lt Layout of the array.
	\throw Kartet::InvalidLayout If the layout is invalid (incorrect sizes).
	**/
	template<typename T, Location l>
	__host__ Array<T,l>::Array(const Layout& lt)
	 :	Accessor<T,l>(lt.monolithicLayout())
	{
		if(!layout().isValid())
			throw InvalidLayout;
		allocateMemory();
	}

	/**
	\brief Array constructor.
	\param ptr Data to be copied. This memory space will not be used once the constructor finishes.
	\param r Number of rows.
	\param c Number of columns.
	\param s Number of slices.
	\throw Kartet::InvalidLayout If the layout is invalid (incorrect sizes).
	**/
	template<typename T, Location l>
	__host__ Array<T,l>::Array(const T* ptr, index_t r, index_t c, index_t s)
	 :	Accessor<T,l>(r, c, s)
	{
		if(!layout().isValid())
			throw InvalidLayout;
		allocateMemory();
		setData(ptr);
	}

	/**
	\brief Array constructor.
	\param ptr Data to be copied. This memory space will not be used once the constructor finishes.
	\param lt Layout of the array.
	\throw Kartet::InvalidLayout If the layout is invalid (incorrect sizes).
	**/
	template<typename T, Location l>
	__host__ Array<T,l>::Array(const T* ptr, const Layout& lt)
	 :	Accessor<T,l>(layout.monolithicLayout())
	{
		if(!layout().isValid())
			throw InvalidLayout;
		allocateMemory();
		setData(ptr);
	}

	/**
	\brief Copy constructor.
	\param A Array to be copied.
	**/
	template<typename T, Location l>
	__host__ Array<T,l>::Array(const Array<T,l>& A)
	 :	Accessor<T,l>(A.monolithicLayout())
	{
		allocateMemory();
		(*this) = A;
	}

	/**
	\brief Copy constructor.
	\param A Accessor to be copied (can also be an array with a different location parameter).
	**/
	template<typename T, Location l>
	template<typename TIn, Location lin>
	__host__ Array<T,l>::Array(const Accessor<TIn,lin>& A)
	 : 	Accessor<T,l>(A.monolithicLayout())
	{
		allocateMemory();
		(*this) = A;
	}

	/**
	\brief Array constructor.
	\param stream Input stream to read from.
	\param convert Set to true if the data can be converted to the current type.
	\param maxBufferSize The maximum buffer size to be allocated to perform the requested conversion.
	\param sourceTypeIndex For internal use only.
	**/
	template<typename T, Location l>
	__host__ Array<T,l>::Array(std::istream& stream, const bool convert, const size_t maxBufferSize, int sourceTypeIndex)
	 :	Accessor<T,l>(Layout::readFromStream(stream, &sourceTypeIndex))
	{
		allocateMemory();
		readFromStream(stream, convert, maxBufferSize, true, sourceTypeIndex);
	}

	/**
	\brief Array constructor.
	\param filename Filename to read from.
	\param convert Set to true if the data can be converted to the current type.
	\param maxBufferSize The maximum buffer size to be allocated to perform the requested conversion.
	\param sourceTypeIndex For internal use only.
	**/
	template<typename T, Location l>
	__host__ Array<T,l>::Array(const std::string& filename, const bool convert, const size_t maxBufferSize, int sourceTypeIndex)
	 :	Accessor<T,l>(Layout::readFromFile(filename, &sourceTypeIndex))
	{
		allocateMemory();
		readFromFile(filename, convert, maxBufferSize);
	}

	/**
	\brief Array destructor.
	**/
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

	/**
	\brief Automatic memory allocation.
	**/
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
					err = cudaMalloc(reinterpret_cast<void**>(&this->ptr), numElements()*sizeof(T));
					if(err!=cudaSuccess)
						throw static_cast<Exception>(CudaExceptionsOffset + err);
					break;
				#else
					throw NotSupported;
				#endif
				}
			case HostSide :
				this->ptr = new T[numElements()];
				break;
			default :
				throw InvalidLocation;
		}
	}

	/**
	\brief Get the accessor corresponding to the full array.
	\return An accessor object.
	**/
	template<typename T, Location l>
	__host__ Accessor<T,l>& Array<T,l>::accessor(void)
	{
		return (*this);
	}

	/**
	\brief Get the accessor corresponding to the full array.
	\return An accessor object.
	**/
	template<typename T, Location l>
	__host__ const Accessor<T,l>& Array<T,l>::accessor(void) const
	{
		return (*this);
	}

	/**
	\brief Build the array with data from a stream.
	\param stream The input stream.
	\param convert Set to true if the data can be converted to the current type.
	\param maxBufferSize The maximum buffer size to be allocated to perform the requested conversion.
	\return A pointer to the newly created array.
	**/
	template<typename T, Location l>
	__host__ Array<T,l>* Array<T,l>::buildFromStream(std::istream& stream, const bool convert, const size_t maxBufferSize)
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

	/**
	\brief Build the array with data from a stream.
	\param filename Filename to read from.
	\param convert Set to true if the data can be converted to the current type.
	\param maxBufferSize The maximum buffer size to be allocated to perform the requested conversion.
	\return A pointer to the newly created array.
	**/
	template<typename T, Location l>
	__host__ Array<T,l>* Array<T,l>::buildFromFile(const std::string& filename, const bool convert, const size_t maxBufferSize)
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

