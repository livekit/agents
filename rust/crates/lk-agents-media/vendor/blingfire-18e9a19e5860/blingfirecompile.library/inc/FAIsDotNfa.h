/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_ISDOTNFA_H_
#define _FA_ISDOTNFA_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"
#include "FALimits.h"

namespace BlingFire
{

class FAAllocatorA;
class FARSNfaA;

///
/// This class identifies whether for all states and the given range of Iws
/// the destination state does not depend on input weight.
///

class FAIsDotNfa {

public:
    FAIsDotNfa (FAAllocatorA * pAlloc);

public:
    /// sets up input automaton
    void SetNfa (const FARSNfaA * pNfa);
    /// specifies the smallest Iw
    void SetIwBase (const int IwBase);
    /// specifies the biggest Iw
    void SetIwMax (const int IwMax);
    /// explicitly sets up alphabet
    /// 1. if -1 == ExpCount then alphabet will be re-calculated
    /// 2. if 0 <= ExpCount then IwBase and IwMax parameters are not used
    void SetExpIws (const int * pExpIws, const int ExpCount);
    /// makes processing and returns result
    const bool Process ();

private:
    // initializes m_alphabet
    void Prepare ();
    // returns true if State's alphabet equals to m_pAlphabet and
    // as a side effect has State's alphabet in m_tmp
    const bool CmpAlphabet (const int State);

private:
    const FARSNfaA * m_pNfa;
    int m_IwBase;
    int m_IwMax;

    FAArray_cont_t < int > m_alphabet;
    const int * m_pAlphabet;
    int m_AlphabetSize;
    bool m_ExpAlphabet;

    FAArray_cont_t < int > m_tmp;
};

}

#endif
