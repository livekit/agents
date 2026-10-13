/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_FLOATARRAYPACK_H_
#define _FA_FLOATARRAYPACK_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Packs the array of arbitrary float numbers. The array is stored in
/// explicit form preceeded by the element count.
/// 
/// Dump format: 
///   <Count>   - int 
///   <V1>      - float
///   <V2>      - float
///   ...
///   <V_Count> - float
///

class FAFloatArrayPack {

public:
    FAFloatArrayPack (FAAllocatorA * pAlloc);

public:
    /// sets up the array of values
    void SetArray (const float * pArr, const int Size);
    /// builds memory dump
    void Process ();
    /// returns built dump
    const int GetDump (const unsigned char ** ppDump) const;

private:
    const float * m_pArr;
    int m_Size;
    // the resulting dump
    FAArray_cont_t < unsigned char > m_dump;
};

}

#endif
