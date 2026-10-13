/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_WRETOKENS2DIGITIZERS_H_
#define _FA_WRETOKENS2DIGITIZERS_H_

#include "FAConfig.h"
#include "FAFsmConst.h"
#include "FAArray_cont_t.h"
#include "FAWRETokens2Dicts.h"
#include "FARSNfa_wo_ro.h"
#include "FARSDfa_ro.h"
#include "FARSDfa_wo_ro.h"
#include "FAState2Ow.h"
#include "FATypesNfaList2TypeNfa.h"
#include "FATypeMinDfa2MinMooreDfa.h"
#include "FAMultiMap_judy.h"
#include "FAMultiMap_ar_uniq.h"
#include "FARegexp2MinDfa.h"
#include "FAStrList2MinDfa.h"
#include "FAStr2Utf16.h"
#include "FAAny2AnyOther_global_t.h"
#include "FANfa2Dfa_t.h"
#include "FADfa2MinDfa_hg_t.h"
#include "FATypeMinDfa2MinMooreDfa.h"
#include "FASplitSets.h"
#include "FADictInterpreter_t.h"
#include "FABitArray.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// This class, for the given list of WRE tokens, builds upto three digitizers.
///
///  1. Txt digitizer - word text references (wordlists, regexps, dicts)
///  2. Tag digitizer - disambiguated tag reference
///  3. Dct digitizer - grammatic/semantic tag reference needs tag dictionary
///
/// Note: Each of the digitizers can not exist, but at least one 
/// should not be empty.
///

class FAWRETokens2Digitizers : public FAWRETokens2Dicts {

public:
    FAWRETokens2Digitizers (FAAllocatorA * pAlloc);
    virtual ~FAWRETokens2Digitizers ();

public:
    /// sets up dictionary files location, for txt digitizer
    void SetDictRoot (const char * pDictRoot);
    /// sets up encoding name
    void SetEncodingName (const char * pEncName);
    /// sets up initial Iw value allowed to be used fot types in txt-digitizer
    void SetInitialTypeIw (const int InitialTypeIw);
    /// sets up initial Ow value for txt-digitizer
    void SetInitialOw (const int Ow);
    /// sets up tag dictionary, if any
    void SetTagDict (const FADictInterpreter_t < int > * pTagDict);
    /// returns object into the initial state
    virtual void Clear ();

public:
    /// returns common Type -> {Ows} map
    const FAMultiMapA * GetType2Ows () const;

    /// returns Txt digitizer, true if exists
    const bool GetTxtDigititizer (
            const FARSDfaA ** ppDfa,        // Word -> Ow (Moore automaton)
            const FAState2OwA ** pState2Ow  // Word -> Ow (Moore automaton)
        ) const;

    /// returns Tag digitizer, true if exists
    const bool GetTagDigitizer (
            const int ** ppTag2Ow,          // Tag -> Ow (Array)
            int * pTag2OwSize               // Tag -> Ow (Array)
        ) const;

    /// returns Dict digitizer, true if exists
    const bool GetDictDigitizer (
            const int ** ppId2Ow,           // InfoId -> Ow (Array)
            int * pId2OwSize                // InfoId -> Ow (Array)
        ) const;

private:
    static inline const int GetLength (
            const char * pBuff, 
            const int MaxBuffSize
        );

/// overriden from FAWRETokens2Dicts
private:
    void PutTxtWords (const char * pBegin, const int Length);
    void PutTxtRegexp (const char * pBegin, const int Length);
    void PutTxtDictName (const char * pBegin, const int Length);
    void PutPosTags (const int * pTags, const int Count);
    void PutDictTags (const int * pTags, const int Count);
    void PutDone ();

private:
    const bool ReadTxtDict (const char * pFileName);
    const bool ReadRgxDict (const char * pFileName);
    void BuildTxtDigitizer ();

    void BuildTagDigitizer ();

    inline void BuildTag2Ids ();
    inline void ClassifyIds ();
    inline void BuildTag2Ows ();
    void BuildDctDigitizer ();

private:
    bool m_NoTxt;
    bool m_NoTag;
    bool m_NoDct;

    // input
    const char * m_pDictRoot;

    /// text-digitizer construction stages:

    // 1. Common TyNfa construction
    int m_DynType;
    FAArray_cont_t < char > m_wl;
    FAStr2Utf16 m_recode;
    FARegexp2MinDfa m_re2dfa;
    FAStrList2MinDfa m_wl2dfa;
    FATypesNfaList2TypeNfa m_tynfa;
    // 2. global '.' - expansion
    FAAny2AnyOther_global_t < FARSNfaA, FARSNfa_wo_ro > m_dot_exp;
    FARSNfa_wo_ro m_nfa;
    // 3. NFA -> DFA
    FANfa2Dfa_t < FARSNfa_wo_ro, FARSDfa_wo_ro > m_nfa2dfa;
    FARSDfa_wo_ro m_dfa;
    // 4. DFA -> Min DFA
    FADfa2MinDfa_hg_t < FARSDfa_wo_ro, FARSDfa_ro > m_dfa2mindfa;
    FARSDfa_ro m_min_dfa;
    // 5. Ty RS DFA -> Moore DFA, Ty -> {Ows}
    FATypeMinDfa2MinMooreDfa m_tyrs2tymoore;
    FARSDfa_wo_ro m_moore_dfa;
    FAState2Ow m_moore_dfa_ows;

    /// dict-digitizer

    const int * m_pDictTags;
    int m_DictTagCount;
    const FADictInterpreter_t < int > * m_pTagDict;
    int m_IdCount;
    FASplitSets m_classify;
    FAMultiMap_judy m_tag2ids;
    FAArray_cont_t < int > m_id2ow;
    FAArray_cont_t < int > m_tmp;
    FABitArray m_tmp2;
    int m_IniOw;

    /// pos-tag digitizer
    const int * m_pPOSTags;
    int m_POSTagCount;

    enum {
        DefInitTyIw = 2000,
        DefOw = FAFsmConst::IW_EPSILON + 1,
    };

    /// Type --> Ow map

    /// in general, each type is mapped to the set of Ows
    /// if the types are initialilly non-intersecting then mapping is one to one
    FAMultiMap_judy m_type2ows;
};

}

#endif
