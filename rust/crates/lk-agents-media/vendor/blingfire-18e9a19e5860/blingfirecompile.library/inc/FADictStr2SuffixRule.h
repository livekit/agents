/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_DICTSTR2SUFFIXRULE_H_
#define _FA_DICTSTR2SUFFIXRULE_H_

#include "FAConfig.h"
#include "FATransformCA_t.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;
class FAMultiMapCA;

///
/// This class converts intput dictionary entry to suffix rule, applying input
/// transformation to the word_from and output transformation to the word_to.
///
/// Input:
///   word_from\tword_to[\ttag_from[\ttag_to]]\n
/// Output:
///   word_from --> [tag_from][tag_to]-N+suff\n
/// or 
///   word_from --> [tag_from][tag_to]-N+suff\t-M+prefix\n
///

class FADictStr2SuffixRule {

public:
    FADictStr2SuffixRule (FAAllocatorA * pAlloc);

public:
    // input transformation
    void SetInTr (const FATransformCA_t < int > * pInTr);
    // output transformation
    void SetOutTr (const FATransformCA_t < int > * pOutTr);
    // specifies whether case should be ignored
    void SetIgnoreCase (const bool IgnoreCase);
    // checks whether both prefix and suffix change
    void SetUsePref (const bool UsePref);
    // sets up normalization map
    void SetCharmap (const FAMultiMapCA * pCharMap);

    // makes conversion
    const int Process (
            const int * pStr,
            const int StrLen,
            const int ** pOutStr
        );

private:
    // splits input string into fields by \t
    inline void Split (const int * pStr, const int StrLen);
    // applies input transformation
    inline void ApplyInTr ();
    // applies output transformation
    inline void ApplyOutTr ();
    // case folding, if needed
    inline void ToLower ();
    // charmap normalization, if needed
    inline void MapChars ();
    // updates m_SuffPos / m_PrefPos
    inline void CalcSuffPref ();
    // builds output
    inline void BuildOutput ();

private:
    // input transformation
    const FATransformCA_t < int > * m_pInTr;
    // output transformation
    const FATransformCA_t < int > * m_pOutTr;
    // ignore case flag
    bool m_IgnoreCase;
    // indicates wether both prefix and suffix should be cut
    bool m_UsePref;
    // fields' pointers
    const int * m_pFrom;
    const int * m_pTo;
    const int * m_pFromTag;
    const int * m_pToTag;
    // fields' lengths
    int m_FromLen;
    int m_ToLen;
    int m_FromTagLen;
    int m_ToTagLen;
    // Suffix and Prefix coordinates
    int m_SuffPos;
    int m_SuffLen;
    int m_SuffCut;
    int m_PrefPos;
    int m_PrefLen;
    int m_PrefCut;
    // output buffer
    FAArray_cont_t < int > m_tmp_from;
    FAArray_cont_t < int > m_tmp_to;
    FAArray_cont_t < int > m_out_buff;
    // charmap
    const FAMultiMapCA * m_pCharMap;
};

}

#endif
