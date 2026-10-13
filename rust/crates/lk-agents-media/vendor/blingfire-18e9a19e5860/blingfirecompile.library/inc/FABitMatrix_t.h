/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */



#ifndef _FA_BITMATRIX_H_
#define _FA_BITMATRIX_H_

#include "FAConfig.h"
#include "FABitArray.h"
#include "FAAllocatorA.h"

namespace BlingFire
{

///
/// Maps (X,Y) -> T|F
/// X \in 0..N, Y \in 0..K
///
/// Additionaly holds: 
///   1. y = MinY (x), s.t. y = Min ({y_i}), Pred (x,y_i) == T
///   2. y = MaxY (x), s.t. y = Max ({y_i}), Pred (x,y_i) == T
///
/// Note: It is assumed that base types to be used as X and Y
///

template < class Ty >
class FABitMatrix_t {

public:
    FABitMatrix_t (FAAllocator * pAlloc);
    ~FABitMatrix_t ();

public:
    // returns predicate value for the given pair
    inline const bool Get (const Ty X, const Ty Y) const;
    // returns min set Y for the given X
    // returns -1 if not defined
    inline const Ty GetMinY (const Ty X) const;
    // returns max set Y for the given X
    // returns -1 if not defined
    inline const Ty GetMaxY (const Ty X) const;

    // for the given X sets up all Ys s.t. (X,Y) == 1
    void Set (const Ty X, const Ty * pYs, const int YsCount);

    // returns container into the initial state
    void Clear ();

private:
    // makes sure that m_x2ys [X] contains new empty entry
    inline void AddEmpty (const Ty X);

private:

    class TPredYs {
    public:
        Ty m_min_y;
        FABitArray m_values;
    };

private:

    FAArray_cont_t < TPredYs > m_x2ys;
    FAAllocatorA * m_pAlloc;
};


template < class Ty >
FABitMatrix_t< Ty >::FABitMatrix_t (FAAllocator * pAlloc) :
    m_pAlloc (pAlloc)
{
    m_x2ys.SetAllocator (pAlloc);
    m_x2ys.Create ();
}


template < class Ty >
FABitMatrix_t< Ty >::~FABitMatrix_t ()
{
    FABitMatrix_t< Ty >::Clear ();
}


template < class Ty >
void FABitMatrix_t< Ty >::Clear ()
{
    const int XCount = m_x2ys.size ();

    for (int x = 0; x < XCount; ++x) {

        m_x2ys [x].m_values.Clear ();
    }

    m_x2ys.Clear ();
}


template < class Ty >
void FABitMatrix_t< Ty >::AddEmpty (const Ty X)
{
    const int XCount = m_x2ys.size ();

    // see whether we should allocate new X entry
    if (XCount <= X) {

        m_x2ys.resize (X + 1, 10);

        for (int x = XCount; x <= X; ++x) {

            TPredYs & ys = m_x2ys [x];

            ys.m_min_y = -1;
            ys.m_values.SetAllocator (m_pAlloc);
            ys.m_values.Create ();
        }

    } else {

        // just clean existing X entry
        TPredYs & ys = m_x2ys [X];

        ys.m_min_y = -1;
        ys.m_values.resize (0);
    }
}


template < class Ty >
void FABitMatrix_t< Ty >::Set (const Ty X, const Ty * pYs, const int YsCount)
{
    DebugLogAssert (0 <= X);
    DebugLogAssert (FAIsSortUniqed (pYs, YsCount));

    // ensure there is entry for X allocated
    AddEmpty (X);

    // store new values
    if (0 < YsCount) {

        DebugLogAssert (pYs);

        const Ty MinY = pYs [0];
        DebugLogAssert (0 <= MinY);

        TPredYs & ys = m_x2ys [X];
        ys.m_min_y = MinY;

        const Ty MaxY = pYs [YsCount - 1];

        FABitArray * pBits = &(ys.m_values);
        DebugLogAssert (pBits);

        const int BitCount = MaxY - MinY + 1;
        pBits->resize (BitCount, 0);
        pBits->set_bits (0, BitCount - 1, false);

        for (int i = 0; i < YsCount; ++i) {

            DebugLogAssert (pYs);
            const int Bit = pYs [i] - MinY;
            pBits->set_bit (Bit, true);
        }
    }
}


template < class Ty >
const bool FABitMatrix_t< Ty >::Get (const Ty X, const Ty Y) const
{
    DebugLogAssert (0 <= X);
    DebugLogAssert (0 <= Y);

    const Ty MaxX = m_x2ys.size () - 1;

    if (MaxX >= X) {

        const TPredYs & ys = m_x2ys [X];
        const Ty MinY = ys.m_min_y;

        if (MinY <= Y) {

            const FABitArray * pBits = &(ys.m_values);
            DebugLogAssert (pBits);

            const int BitCount = pBits->size ();
            const int MaxY = MinY + BitCount - 1;

            if (MaxY >= Y) {

                const bool Val = pBits->get_bit (Y - MinY);
                return Val;

            } // of if (MaxY >= Y)
        } // of if (MinY <= Y)
    } // of if (MaxX >= X)

    return false;
}


template < class Ty >
const Ty FABitMatrix_t< Ty >::GetMinY (const Ty X) const
{
    DebugLogAssert (0 <= X);

    const Ty MaxX = m_x2ys.size () - 1;

    if (MaxX >= X) {

        const TPredYs & ys = m_x2ys [X];
        const Ty MinY = ys.m_min_y;

        return MinY;
    }

    return -1;
}


template < class Ty >
const Ty FABitMatrix_t< Ty >::GetMaxY (const Ty X) const
{
    DebugLogAssert (0 <= X);

    const Ty MaxX = m_x2ys.size () - 1;

    if (MaxX >= X) {

        const TPredYs & ys = m_x2ys [X];
        const Ty MinY = ys.m_min_y;

        const FABitArray * pBits = &(ys.m_values);
        DebugLogAssert (pBits);

        const int BitCount = pBits->size ();
        const int MaxY = MinY + BitCount - 1;

        return MaxY;
    }

    return -1;
}

}

#endif
