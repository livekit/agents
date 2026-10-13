/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_SPLITSETS_H_
#define _FA_SPLITSETS_H_

#include "FAConfig.h"
#include "FAChain2Num_hash.h"
#include "FAMultiMap_judy.h"
#include "FAEncoder_pref.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;


class FASplitSets {

public:
    FASplitSets (FAAllocatorA * pAlloc);

public:
    /// makes object ready 
    void Prepare (const int Count);
    /// sets up information for each element (index)
    void AddInfo (const int * pI, const int Count);
    /// makes result to be ready
    void Process ();
    /// returns element (index) -> class map
    const int GetClasses (const int ** ppC);
    /// returns object into the initial state, called automatically
    void Clear ();

private:
    inline void Classify ();
    inline static const int Info2Class (
            FAChain2Num_hash * pM, 
            const int * pInfo, 
            const int InfoSize
        );

private:
    enum { MaxAtOnce = 250, };

    FAEncoder_pref m_enc;
    FAChain2Num_hash m_info2c;
    FAMultiMap_judy m_e2v;
    int m_size;
    FAArray_cont_t < int > m_e2c;
};

}

#endif
