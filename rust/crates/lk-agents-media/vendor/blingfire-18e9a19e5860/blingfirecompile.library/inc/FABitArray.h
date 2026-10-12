/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_BIT_ARRAY_H_
#define _FA_BIT_ARRAY_H_

#include "FAConfig.h"

namespace BlingFire
{

class FAAllocatorA;


class FABitArray {

public:
    FABitArray ();
    ~FABitArray ();

public:
    // sets up allocator
    void SetAllocator (FAAllocatorA* pAlloc);
    // makes array ready (should be called after allocator have set up)
    // this method assumes that object memory is not initialized yet
    void Create ();
    // makes the array to be as if the constructor has just been called
    // frees memory
    void Clear ();

public:
    // return true if the Idx-th bit was set, false otherwise
    inline const bool get_bit (const unsigned int Idx) const
    {
        DebugLogAssert (m_ar);
        DebugLogAssert (Idx < m_size);

        const unsigned int HiIdx = Idx >> 5;
        const unsigned int LoIdx = Idx & 31;
        DebugLogAssert (HiIdx < m_ar_size);

        const unsigned int Bits = m_ar [HiIdx];
        const unsigned int Mask = 1 << LoIdx;

        return 0 != (Mask & Bits);
    }

    // sets the Idx-th bit
    inline void set_bit (const unsigned int Idx, const bool Val)
    {
        DebugLogAssert (m_ar);
        DebugLogAssert (Idx < m_size);

        const unsigned int HiIdx = Idx >> 5;
        const unsigned int LoIdx = Idx & 31;
        DebugLogAssert (HiIdx < m_ar_size);

        if (Val) {

            const unsigned int Mask = 1 << LoIdx;
            m_ar [HiIdx] |= Mask;

        } else {

            const unsigned int Mask = ~(1 << LoIdx);
            m_ar [HiIdx] &= Mask;
        }
    }

    // sets all bits from the range to the specified value
    void set_bits (const unsigned int FromIdx,
                   const unsigned int ToIdx,
                   const bool Val);

    // adds element into array
    void push_back (const bool Val);
    // adds element into array
    void push_back (const bool Val, const int Delta);
    // removes last bit from the array
    void pop_back ();
    // returns the number of bits stored
    const unsigned int size () const;
    // inserts or erases elements at the end such that the size becomes N
    // (does not free memory, allocates only if needed)
    void resize (const unsigned int N, const unsigned int Delta);
    // the same as above but assumes Delta == 1024
    void resize (const unsigned int N);
    // makes the size of the array to be 0 (does not free memory)
    // bit faster than resize
    void clear ();
    // returns raw memory where bits are stored
    const unsigned int * begin () const;

private:

    // pointer to the allocated buffer
    unsigned int * m_ar;
    // the size of the allocated buffer
    unsigned int m_ar_size;
    // the number of bits stored
    unsigned int m_size;
    // allocator
    FAAllocatorA * m_pAlloc;
};

}

#endif
