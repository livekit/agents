/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */



#ifndef _FA_HEAP_T_H_
#define _FA_HEAP_T_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;


template < class Ty >
class FAHeap_t {

public:
    void Create (FAAllocatorA * pAlloc);

public:
    /// returns the number of elementes in the heap
    inline const int size () const;
    /// makes the heap empty
    inline void clear ();
    /// returns true if the heap is empty
    inline const bool empty () const;
    /// returns the smallest element
    inline const Ty * top () const;
    /// removes the smallest element 
    inline void pop ();
    /// adds in an element
    inline void push (const Ty& e);

private:

    inline const Ty * at (const int i) const;
    inline void swap (const int idx1, const int idx2);
    inline void heapify_down (int i);
    inline void heapify_up (int i);

private:

    FAArray_cont_t < Ty > m_array;
};


template < class Ty >
void FAHeap_t< Ty >::Create (FAAllocatorA * pAlloc)
{
    m_array.SetAllocator (pAlloc);
    m_array.Create ();

    // there must always be a first dummy element
    m_array.push_back (0);
}

template < class Ty >
const Ty * FAHeap_t< Ty >::at (const int i) const
{
    if (1 > i || m_array.size () <= (unsigned int) i)
        return NULL;

    return & m_array [i];
}

template < class Ty >
void FAHeap_t< Ty >::swap (const int idx1, const int idx2)
{
    DebugLogAssert (1 <= idx1 && m_array.size () > (unsigned int)idx1);
    DebugLogAssert (1 <= idx2 && m_array.size () > (unsigned int)idx2);

    const Ty E = m_array [idx1];
    m_array [idx1] = m_array [idx2];
    m_array [idx2] = E;
}

template < class Ty >
void FAHeap_t< Ty >::heapify_down (int i)
{
    while (true) {

        const int i_2 = i << 1;

        const Ty * e = at (i);
        const Ty * left = at (i_2);
        const Ty * right = at (i_2 + 1);

        if (NULL == left && NULL == right)
            break;

        if (NULL != left) {

            if (NULL != right) {

                if (*e < *left && *e < *right)
                    break;

                if (*right < *left) {

                    swap (i, i_2 + 1);
                    i = i_2 + 1;

                } else {

                    swap (i, i_2);
                    i = i_2;
                }

            } else {

                if (*e < *left)
                    break;

                swap (i, i_2);
                i = i_2;
            }
        }
    } // of while
}

template < class Ty >
void FAHeap_t< Ty >::heapify_up (int i)
{
    while (true) {

        if (1 == i)
            break;

        const int i_2 = i >> 1;

        const Ty * e = at (i);
        const Ty * parent = at (i_2);

        if (*e > *parent)
            break;

        swap (i, i_2);
        i = i_2;

    } // of while
}

template < class Ty >
const int FAHeap_t< Ty >::size () const
{
    return m_array.size () - 1;
}

template < class Ty >
void FAHeap_t< Ty >::clear ()
{
    if(1 < m_array.size()) {
      m_array.resize (1);
    }
}

template < class Ty >
const bool FAHeap_t< Ty >::empty () const
{
    return 1 == m_array.size ();
}

template < class Ty >
const Ty * FAHeap_t< Ty >::top () const
{
    DebugLogAssert (1 < m_array.size ());
    return & m_array [1];
}

template < class Ty >
void FAHeap_t< Ty >::pop ()
{
    DebugLogAssert (1 < m_array.size ());

    // move the last element to the begining
    m_array [1] = m_array [m_array.size () - 1];

    // delete the last element
    m_array.pop_back ();

    // heapify from the beginining
    heapify_down (1);
}

template < class Ty >
void FAHeap_t< Ty >::push (const Ty& e)
{
    const int new_idx = m_array.size ();

    // insert a new element into the end
    m_array.push_back (e);

    // heapify from the end
    heapify_up (new_idx);
}

}

#endif
