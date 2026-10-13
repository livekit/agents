/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_IWMAPPACK_H_
#define _FA_IWMAPPACK_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// This class creates dump representation for Iw2Iw mapping.
///
/// Binary Format:
/// BEGIN
/// <SizeOfInternalIw>                      : int
/// <IntervalCount>                         : int
/// <Array of BeginIws>                     : int * IntervalCount
/// <Array of <EndIw, Offset> pairs >       : 2 * int * IntervalCount
/// <InternalIwsArray>                      : SizeOfInternalIw * L
/// END
///
/// Where L is a number of elements in all the intervals.
///

class FAIwMapPack {

public:
    FAIwMapPack (FAAllocatorA * pAlloc);

public:
    // overrides default value for maximum size of gap allowed
    void SetMaxGap (const int MaxGap);
    // sets up two parallel arrays of input weights, pOldIws should be sorted
    void SetIws (const int * pOldIws, const int * pNewIws, const int Count);
    // builds memory dump
    void Process ();
    // returns built dump
    const int GetDump (const unsigned char ** ppDump) const;

private:
    // returns object into initial state
    void Clear ();
    // calculates size in bytes of maximum internal iw
    void CalcNewIwSize ();
    // adds new interval into arrays
    void AddInterval (const int IwFrom, const int IwTo);
    // builds intermediate arrays
    void BuildIntervals ();
    // returns new iw for the given old iw
    inline const int GetNewIw (const int OldIw) const;
    // builds NewIws array
    void BuildNewIwsArray ();
    // stores NewIw into the output dump
    void EncodeIw (const unsigned int NewIw);
    // builds output dump
    void BuildDump ();


private:
    // sorted array of original input weights
    const int * m_pOldIws;
    // internal input weights, s.t. m_pOldIws [i] -> m_pNewIws [i]
    const int * m_pNewIws;
    // number of input weights
    int m_Count;
    // size of internal (Iws + 1) in bytes
    int m_NewIwSize;
    // intermediate array of interval beginnings
    FAArray_cont_t < int > m_ArrStart;
    // intermediate array of interval endings
    FAArray_cont_t < int > m_ArrEnd;
    // intermediate array of indices in the output array of internal Iws
    FAArray_cont_t < int > m_ArrIdx;
    // temporary variable, keeps first avaliable index in m_ArrIdx
    int m_MaxIdx;
    // intermediate array internal (Iws + 1), 0 means no value stored
    FAArray_cont_t < int > m_NewIws;
    // output dump
    FAArray_cont_t < unsigned char > m_dump;
    unsigned char * m_pDump;
    unsigned int m_Offset;

    // compression constants
    int m_MaxGap;
    enum { DefaultMaxGap = 50 };
};

}

#endif
