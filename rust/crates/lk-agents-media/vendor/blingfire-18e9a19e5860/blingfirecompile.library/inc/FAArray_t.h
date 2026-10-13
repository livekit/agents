/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_ARRAY_T_H_
#define _FA_ARRAY_T_H_

#include "FAConfig.h"
#include "FAAllocatorA.h"
#include "FAException.h"
#include "FALimits.h"

namespace BlingFire
{

/// non-contiguous array, but has fewer than FAArray_cont_t relocations

template <class Ty>
class FAArray_t {

 public:

  FAArray_t () :
    m_ar (NULL),
    m_Bits (10),
    m_Mask ((1 << 10) - 1),
    m_Delta (0),
    m_ar_size (0),
    m_size (0),
    m_pAlloc (NULL)
    {
    }

  FAArray_t (
	   const int Bits, // 2^bits == size of minimal contiguous block.
	   const int Delta // The number of extra blocks to be added during
	                   // the allocation.
	   ) :
    m_ar (NULL),
    m_Bits (Bits),
    m_Mask ((1 << Bits) - 1),
    m_Delta (Delta),
    m_ar_size (0),
    m_size (0),
    m_pAlloc (NULL)
    {
    }

  FAArray_t (const FAArray_t <Ty>& A) :
    m_Bits (A.m_Bits),
    m_Mask (A.m_Mask),
    m_Delta (A.m_Delta)
    {
      // allocate memory
      FAArray_t::Create (A.size ());
      // copy data
      *this = A;
    }

  ~FAArray_t ()
    {
      FAArray_t::Clear ();
    }

 public:

  /// Sets up allocator.
  void SetAllocator (FAAllocatorA* pAlloc)
    {
      m_pAlloc = pAlloc;
    }

  /// Makes array ready (should be called after allocator have set up).
  void Create (const unsigned int Size = 0)
    {
      DebugLogAssert (NULL == m_ar && 0 == m_ar_size);
      DebugLogAssert (m_pAlloc);

      if (FALimits::MaxArrSize < Size) {
        throw FAException (FAMsg::LimitIsExceeded, __FILE__, __LINE__);
      }

      m_size = 0;

      // allocate m_ar
      unsigned int ar_size = (Size >> m_Bits) + 1 + m_Delta;
      m_ar = (Ty **) FAAlloc (m_pAlloc, ar_size * sizeof (Ty *));

      // allocate contiguous blocks
      for (unsigned int i = 0; i < ar_size; ++i) {
        m_ar_size++;
        m_ar [i] = NULL;
        Ty * ar2 = (Ty *) FAAlloc (m_pAlloc, (m_Mask + 1) * sizeof (Ty));
        m_ar [i] = ar2;
      }
    }


  /// Makes the array to be as just after a constructor is called.
  /// Frees memory.
  void Clear ()
    {
      if (0 < m_ar_size) {

	DebugLogAssert (m_pAlloc);

	// free contiguous blocks
	for (unsigned int i = 0; i < m_ar_size; ++i) {

	  Ty * ar2 = m_ar [i];
	  FAFree (m_pAlloc, ar2);
	}

	// free the main array
	FAFree (m_pAlloc, m_ar);
      }

      m_ar = NULL;
      m_ar_size = 0;
      m_size = 0;
    }

 public:

  /// Gets element by its index
  const Ty& operator[] (unsigned int Idx) const
    {
      DebugLogAssert (m_ar);
      DebugLogAssert (Idx < m_size);
      DebugLogAssert ((Idx >> m_Bits) < m_ar_size);
      DebugLogAssert (m_ar [Idx >> m_Bits]);

      return m_ar [Idx >> m_Bits][Idx & m_Mask];
    }

  /// Returns the number of elements.
  inline const unsigned int size () const
    {
      return m_size;
    }

  /// Returns true is array is empty
  inline const bool empty () const
    {
      return 0 == m_size;
    }

  /// Adds element into Array.
  inline void push_back (const Ty& Val)
    {
      // change size
      resize (m_size + 1);

      // setup the last element
      (*this)[m_size - 1] = Val;
    }

  /// Removes the last element.
  inline void pop_back()
    {
      m_size--;
    }

  /// Gets element by its index for modification.
  Ty& operator[] (const unsigned int Idx)
    {
      DebugLogAssert (m_ar);
      DebugLogAssert (Idx < m_size);
      DebugLogAssert ((Idx >> m_Bits) < m_ar_size);
      DebugLogAssert (m_ar [Idx >> m_Bits]);

      return m_ar [Idx >> m_Bits][Idx & m_Mask];
    }

  /// Inserts or erases elements at the end such that the size becomes N.
  /// (does not free memory, allocates only if needed)
  inline void resize (const unsigned int N)
    {
      const unsigned int RealSize =  m_ar_size << m_Bits;

      // check whether we have to allocate more memory
      if (RealSize < N) {

	DebugLogAssert (m_pAlloc);

        // overflow check: (N >> m_Bits) + 1 + m_Delta
        if (FALimits::MaxArrSize < N) {
            throw FAException (FAMsg::LimitIsExceeded, __FILE__, __LINE__);
        }

	const unsigned int old_ar_size = m_ar_size;

	unsigned int ar_size = (N >> m_Bits) + 1 + m_Delta;
	m_ar = (Ty **) FAReAlloc (m_pAlloc, m_ar, ar_size * sizeof (Ty *));

	// for each new element in m_ar allocate a contiguous block
	for (unsigned int i = old_ar_size; i < ar_size; ++i) {
      m_ar_size++;
      m_ar [i] = NULL;
	  Ty * ar2 = (Ty *) FAAlloc (m_pAlloc, (m_Mask + 1) * sizeof (Ty));
	  m_ar [i] = ar2;
	}
      }

      m_size = N;
    }

  /// Makes the size of the array to be 0 (does not free memory).
  inline void clear ()
    {
      m_size = 0;
    }

  /// Makes a copy of A (rellocate memory only if necessary).
  FAArray_t <Ty> & operator= (const FAArray_t <Ty>& A)
    {
      if (this != &A) {

	// copy allocator ptr
	m_pAlloc = A.m_pAlloc;

	// resize this array
	const unsigned int Size = A.size ();
	resize (Size);

	// see whether a structure of Arrays is not the same
	if (m_Bits != A.m_Bits) {

	  // make a slow copy
	  for (unsigned int i = 0; i < Size; ++i) {
	    (*this) [i] = A [i];
	  }

	} else {

	  // make a fast copy
	  const unsigned int BlocksToCopy = m_ar_size;

	  for (unsigned int i = 0; i < BlocksToCopy; ++i) {
	    memcpy (m_ar [i], A.m_ar [i], m_Mask + 1);
	  }
	}
      }

      return *this;
    }

 private:

  Ty ** m_ar;

  const int m_Bits;
  const int m_Mask;
  const int m_Delta;

  unsigned int m_ar_size;
  unsigned int m_size;

  FAAllocatorA * m_pAlloc;

};

}

#endif
