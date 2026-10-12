/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_TESTCMPMULTIMAP_H_
#define _FA_TESTCMPMULTIMAP_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;
class FAMultiMapCA;

///
/// Compares whether two FAMultiMapCA interfaces behave the same way.
///

class FATestCmpMultiMap {

public:
    FATestCmpMultiMap (FAAllocatorA * pAlloc);

public:
    // sets up the first map
    void SetMap1 (const FAMultiMapCA * pMMap1);
    // sets up the second map
    void SetMap2 (const FAMultiMapCA * pMMap2);
    // returns true if interfaces behave identically
    const bool Process (const int MaxKey);

private:
    inline const int GetChain1 (const int Key, const int ** ppChain);
    inline const int GetChain2 (const int Key, const int ** ppChain);
    // returns true if chains are equal
    inline const bool Equal (
            const int * pChain1, 
            const int Size1,
            const int * pChain2, 
            const int Size2
        ) const;

private:
    const FAMultiMapCA * m_pMMap1;
    const FAMultiMapCA * m_pMMap2;

    FAArray_cont_t < int > m_chain1;
    int * m_pChainBuff1;
    int m_MaxChainSize1;

    FAArray_cont_t < int > m_chain2;
    int * m_pChainBuff2;
    int m_MaxChainSize2;

};

}

#endif
