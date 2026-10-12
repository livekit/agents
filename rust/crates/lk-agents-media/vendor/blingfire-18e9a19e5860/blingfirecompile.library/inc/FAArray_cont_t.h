/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_ARRAY_CONT_T_H_
#define _FA_ARRAY_CONT_T_H_

#include "FAConfig.h"
#include "FAAllocatorA.h"
#include "FAException.h"
#include "FALimits.h"

namespace BlingFire
{

/// contiguous array

template <class Ty>
class FAArray_cont_t {

public:

    FAArray_cont_t () :
        m_ar (NULL),
        m_pAlloc (NULL),
        m_ar_size (0),
        m_size (0)
    {
    }

    FAArray_cont_t (const FAArray_cont_t <Ty>& A)
    {
        // allocate memory
        FAArray_cont_t::Create (A.size ());
        // copy data
        *this = A;
    }

    ~FAArray_cont_t ()
    {
        FAArray_cont_t::Clear ();
    }

public:

    /// Sets up allocator.
    void SetAllocator (FAAllocatorA* pAlloc)
    {
        m_pAlloc = pAlloc;
    }

    /// Makes array ready (should be called after allocator have set up).
    /// This method assumes that object memory is not initialized yet
    void Create ()
    {
        // copy size
        m_size = 0;
        m_ar = (Ty *) FAAlloc (m_pAlloc, 1 * sizeof (Ty));
        m_ar_size = 1;
    }

    /// Makes array ready (should be called after allocator have set up).
    /// ( this method assumes that 0 < Size )
    /// This method assumes that object memory is not initialized yet
    void Create (const unsigned int Size)
    {
        DebugLogAssert (0 < Size);

        // copy size
        m_size = 0;

        // overflow check: Size * sizeof (Ty)
        FAAssert (FALimits::MaxArrSize >= Size, FAMsg::LimitIsExceeded);

        m_ar = (Ty *) FAAlloc (m_pAlloc, Size * sizeof (Ty));
        m_ar_size = Size;
    }

    /// Makes the array to be as just after a constructor is called.
    /// Frees memory.
    void Clear ()
    {
        if (0 < m_ar_size) {

            DebugLogAssert (m_pAlloc);

            // free the main array
            FAFree (m_pAlloc, m_ar);
        }

        m_ar = NULL;
        m_ar_size = 0;
        m_size = 0;
    }

public:

    /// Gets element by its index
    const Ty& operator[] (const unsigned int Idx) const
    {
        DebugLogAssert (m_ar);
        DebugLogAssert (Idx < m_size);

        return m_ar [Idx];
    }

    /// Gets element by its index for modification.
    Ty& operator[] (const unsigned int Idx)
    {
        DebugLogAssert (m_ar);
        DebugLogAssert (Idx < m_size);

        return m_ar [Idx];
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

    /// Removes the last element.
    inline void pop_back()
    {
        m_size--;
    }

    /// Returns the random access iterator
    inline Ty * begin ()
    {
        return m_ar;
    }

    /// Returns the random access iterator
    inline Ty * end ()
    {
        return m_ar + m_size;
    }

    /// Returns the const random access iterator
    inline const Ty * begin () const
    {
        return m_ar;
    }

    /// Returns the const random access iterator
    inline const Ty * end () const
    {
        return m_ar + m_size;
    }

    /// Makes the size of the array to be 0 (does not free memory).
    inline void clear ()
    {
        m_size = 0;
    }

    /// Inserts or erases elements at the end such that the size becomes N.
    /// (does not free memory, allocates only if needed)
    inline void resize (const unsigned int N, const unsigned int Delta = 1024)
    {
        // check whether we have to allocate more memory
        if (m_ar_size < N) {

            DebugLogAssert (m_pAlloc);

            // overflow check: (N + Delta) * sizeof (Ty)
            FAAssert (FALimits::MaxArrSize >= N && \
              FALimits::MaxArrSize >= Delta, FAMsg::LimitIsExceeded);

            unsigned int ar_size = N + Delta;
            m_ar = (Ty *) FAReAlloc (m_pAlloc, m_ar, ar_size * sizeof (Ty));
            m_ar_size = ar_size;
        }

        m_size = N;
    }

    /// Adds element into Array.
    inline void push_back (const Ty& Val)
    {
        DebugLogAssert (m_ar);

        // change size
        resize (m_size + 1);

        // setup the last element
        m_ar [m_size - 1] = Val;
    }

    /// Adds element into Array.
    inline void push_back (const Ty& Val, const unsigned int Delta)
    {
        DebugLogAssert (m_ar);

        // change size
        resize (m_size + 1, Delta);

        // setup the last element
        m_ar [m_size - 1] = Val;
    }

    /// Makes a copy of A (rellocate memory only if necessary).
    FAArray_cont_t <Ty> & operator= (const FAArray_cont_t <Ty>& A)
    {
        if (this != &A) {

            // copy allocator ptr
            m_pAlloc = A.m_pAlloc;

            // resize this array
            const unsigned int Size = A.size ();
            resize (Size);

            // make a copy
            memcpy (m_ar , A.m_ar, Size * sizeof (Ty));
        }

        return *this;
    }

private:

    Ty * m_ar;
    FAAllocatorA* m_pAlloc;
    unsigned int m_ar_size;
    unsigned int m_size;

};

}

#endif
