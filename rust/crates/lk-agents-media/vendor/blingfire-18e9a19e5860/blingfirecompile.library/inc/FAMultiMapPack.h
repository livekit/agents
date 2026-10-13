/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_MULTIMAPPACK_H_
#define _FA_MULTIMAPPACK_H_

#include "FAConfig.h"
#include "FAMultiMapA.h"
#include "FAArray_cont_t.h"
#include "FAChainsPack_triv.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Builds memory dump for the given FAMultiMapA
///

class FAMultiMapPack {

public:
    FAMultiMapPack (FAAllocatorA * pAlloc);

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
    // returns object into the initial state
    void Clear ();
    // fills in m_keys, m_offsets and packs values
    void Prepare ();
    // calculates size of (offset + 1) in bytes
    inline const unsigned int CalcOffsetSize () const;
    // calculates resulting dump size
    inline const unsigned int CalcSize () const;
    // stores offset 
    inline void EncodeOffset (
            const int Key, 
            const int Offset, 
            const int SizeOfOffset
        );
    // builds the output dump
    void BuildDump ();

private:
    // input multi-map
    const FAMultiMapA * m_pMMap;
    // values dump
    FAChainsPack_triv m_vals2dump;
    // array of keys
    FAArray_cont_t < int > m_keys;
    // array of offsets
    FAArray_cont_t < int > m_offsets;
    // output dump
    FAArray_cont_t < unsigned char > m_dump;
    unsigned char * m_pDump;
    int m_Offset;

};

}

#endif
