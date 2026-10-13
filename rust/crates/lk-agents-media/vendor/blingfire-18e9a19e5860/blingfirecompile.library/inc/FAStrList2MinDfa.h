/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_STRLIST2MINDFA_H_
#define _FA_STRLIST2MINDFA_H_

#include "FAConfig.h"
#include "FAChains2MinDfa.h"
#include "FAStr2Utf16.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// This class takes a 0-delimited list of strings in the specified encoding
/// and builds a corresponding Min DFA.
///

class FAStrList2MinDfa {

public:
    FAStrList2MinDfa (FAAllocatorA * pAlloc);

public:
    /// specifies input encoding name, e.g. CP1251, KOI8-R
    void SetEncodingName (const char * pEncStr);
    /// sets up input list of strings
    void SetStrList (const char * pStrList, const int Size);
    /// does actual processing
    void Process ();
    /// returns read-only interface to the corresponding Min DFA
    const FARSDfaA * GetRsDfa () const;

private:
    // input list of strings
    const char * m_pStrList;
    int m_Size;
    // str -> chain converter
    FAStr2Utf16 m_recode;
    // temporary chain
    FAArray_cont_t < int > m_chain;
    // builds min dfa from the list chains
    FAChains2MinDfa m_chains2dfa;
};

}

#endif
