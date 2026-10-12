/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_IWOWSUFFARR2PATTERNS_H_
#define _FA_IWOWSUFFARR2PATTERNS_H_

#include "FAConfig.h"
#include "FARSDfa_ar_judy.h"
#include "FARSDfa_dynamic_t.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// This class builds list patterns from the sorted suffixes of Iw/Ow chains.
/// 
/// Notes:
///
/// 1. The algorithm generates each pattern as general as possible (e.g. as
///    short as possible), but with no less than N% precision.
///
/// 2. Expects chains in the following format: [Iw_1, Ow_1, ..., Iw_n, Ow_n].
///
/// 3. State1, Dst1, Src1 are states of IwOw-trie and
///    State2, Dst2, Src2 are *corresponding* states of the Iw-trie
///

class FAIwOwSuffArr2Patterns {

public:
    FAIwOwSuffArr2Patterns (FAAllocatorA * pAlloc);
    virtual ~FAIwOwSuffArr2Patterns ();

public:
    /// sets up minimal length of pattern
    void SetMinPatLen (const int MinPatLen);
    /// sets up a lower precision bound for patterns, 100 is used by default
    void SetMinPatPrec (const float MinPatPrec);
    /// should be called for each input chain e.g. Iw_1, Ow_1, ..., Iw_n, Ow_n
    void AddChain (const int * pChain, const int Size, const int Freq);
    /// should be called after all chains added
    void Process ();

    /// callback, returns all accepted patterns
    virtual void PutPattern (const int * pPat, const int Size, const int Freq);

private:
    // returns true if the m_MinPatLen of Iws has changed
    inline const bool HasPrefChanged (const int * pChain, const int Size) const;
    // adds chain's tail to the existing State of IwOw-trie
    void AddTail_iwow (
            const int State, 
            const int * pTail, 
            const int Size, 
            const int Freq
        );
    // adds chain's tail to the existing State2 of Iw-trie
    void AddTail_iw (
            const int State1, 
            const int State2, 
            const int * pTail, 
            const int Size,
            const int Freq
        );
    // adds chain (internal)
    void AddChain_int (const int * pChain, const int Size, const int Freq);
    // updates Iws and Ows alphabets
    inline void UpdateAlphabets (const int * pChain, const int Size);
    // builds pattern text, symbol by symbol
    inline void SetIwOw (const int Iw, const int Ow, const int Pos);
    // builds all patterns
    void BuildPatterns ();
    // internal clean-up
    void Clear ();

private:
    // min pattern length
    int m_MinPatLen;
    // min pattern precision
    float m_MinPatPrec;
    // trie of all Iw:Ow suffixes (suffixes with the same m_MinPatLen prefix)
    FARSDfa_dynamic_t < FARSDfa_ar_judy > m_iwow_trie;
    // trie of all Iw suffixes (suffixes with the same m_MinPatLen prefix)
    FARSDfa_dynamic_t < FARSDfa_ar_judy > m_iw_trie;
    // IwOw Trie State -> Freq
    FAArray_cont_t < int > m_iwow_state2freq;
    // Iw Trie State -> Freq
    FAArray_cont_t < int > m_iw_state2freq;
    // Iw State -> IwOw State (or -1)
    FAArray_cont_t < int > m_state2state;
    // input alphabet
    FAArray_cont_t < int > m_iws;
    // output alphabet
    FAArray_cont_t < int > m_ows;
    // prev chain / traversal stack
    FAArray_cont_t < int > m_tmp_arr;
    // pattern buffer
    FAArray_cont_t < int > m_pat;

    enum {
        DefMinPatLen = 3,
    };
};

}

#endif
