/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_MULTIMAPPACK_FIXED_H_
#define _FA_MULTIMAPPACK_FIXED_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAMultiMapA;
class FAAllocatorA;

///
/// Builds memory dump for the given FAMultiMapA
///

class FAMultiMapPack_fixed {

public:
    FAMultiMapPack_fixed (FAAllocatorA * pAlloc);

public:
    // sets up input map
    void SetMultiMap (const FAMultiMapA * pMMap);
    // sets up min sizeof value, otherwise minimum possible is used
    void SetSizeOfValue (const int SizeOfValue);
    // builds dump
    void Process ();
    // returns bulit dump
    const int GetDump (const unsigned char ** ppDump) const;

private:
    const int Prepare ();

private:
    // input multi-map
    const FAMultiMapA * m_pMMap;
    // value size
    int m_ValueSize;
    // min Key
    int m_MinKey;
    // max Key
    int m_MaxKey;
    // maximum amount of values associated with a Key
    int m_MaxCount;
    // output dump
    FAArray_cont_t < unsigned char > m_dump;
    unsigned char * m_pDump;
};

}

#endif
