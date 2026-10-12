/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_STEMMERCONST_T_H_
#define _FA_STEMMERCONST_T_H_

#include "FAConfig.h"
#include "FAFsmConst.h"
#include "FALimits.h"
#include "FAStemmerLDB.h"
#include "FASuffixInterpretToolsConst_t.h"
#include "FAUtf32Utils.h"
#include "FAUtils_cl.h"
#include "FASecurity.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

///
/// PRM-based stemmer (expansive, reductive and context-sensitive stemming).
///
/// Notes:
///
/// 1. The Ty is the character type, can be any basic type enough to hold 
///    expected Unicode values.
///
/// 2. The caller of the stemmer should verify  that the actual  (returned  by
///    Process* method) size is not bigger than  provided output maximum size.
///    If it is bigger  then  the caller should provide  a buffer  of  atleast 
///    that size and  call the function again or fail if increasing the buffer
///    is not acceptable.
///

template < class Ty >
class FAStemmerConst_t {

public:
    /// creates uninitialized object
    FAStemmerConst_t ();
    ~FAStemmerConst_t ();

public:
    /// This should be a valid initialized object of an LDB with stemmer data
    /// or a NULL pointer. Call this method before any Process* methods.
    void Initialize (const FAStemmerLDB * pLDB, FAAllocatorA * pMemMgr);

public:
    /// word -> { base_form }, word can be on of the base_form too
    /// returns actual output size (see the Notes above!)
    /// returns -1 if no transformation exist
    const int ProcessW2B (
            const Ty * pIn,
            const int InSize,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        ) const;
    /// base_form -> { word }, { word } includes base_form too
    /// returns actual output size (see the Notes above!)
    /// returns -1 if no transformation exist
    const int ProcessB2W (
            const Ty * pIn,
            const int InSize,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        ) const;
    /// word -> { word }, word can be on of the base_form too
    /// returns actual output size (see the Notes above!)
    /// returns -1 if no transformation exist
    const int ProcessW2W (
            const Ty * pIn,
            const int InSize,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        ) const;

    /// word, from_tag -> { base }
    /// returns actual output size (see the Notes above!)
    /// returns -1 if no transformation exist
    const int ProcessWT2B (
            const Ty * pIn,
            const int InSize,
            const int FromTag,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        ) const;
    /// base, to_tag -> { word }
    /// returns actual output size (see the Notes above!)
    /// returns -1 if no transformation exist
    const int ProcessB2WT (
            const Ty * pIn,
            const int InSize,
            const int ToTag,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        ) const;
    /// word, from_tag, to_tag -> { word }
    /// returns actual output size (see the Notes above!)
    /// returns -1 if no transformation exist
    const int ProcessWTT2W (
            const Ty * pIn,
            const int InSize,
            const int FromTag,
            const int ToTag,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        ) const;

private:
    // returns object into initial state
    inline void Clear ();
    // makes initialization and takes care of m_Ready_w2b flag
    inline void InitW2B (FAAllocatorA * pMemMgr);
    // makes initialization and takes care of m_Ready_b2w flag
    inline void InitB2W (FAAllocatorA * pMemMgr);
    // makes initialization and takes care of m_Ready_w2w flag
    inline void InitW2W (FAAllocatorA * pMemMgr);
    // makes initialization and takes care of m_Ready_wt2b flag
    inline void InitWT2B (FAAllocatorA * pMemMgr);
    // makes initialization and takes care of m_Ready_b2wt flag
    inline void InitB2WT (FAAllocatorA * pMemMgr);
    // makes initialization and takes care of m_Ready_wtt2w flag
    inline void InitWTT2W (FAAllocatorA * pMemMgr);

private:
    // keeps morphology resources
    const FAStemmerLDB * m_pLDB;

    // true if object was initialized
    bool m_Ready_w2b;
    bool m_Ready_b2w;
    bool m_Ready_w2w;
    bool m_Ready_wt2b;
    bool m_Ready_b2wt;
    bool m_Ready_wtt2w;

    // word -> base suffix rules interpreter
    FASuffixInterpretToolsConst_t < Ty > * m_pSuffW2B;
    // base -> word suffix rules interpreter
    FASuffixInterpretToolsConst_t < Ty > * m_pSuffB2W;
    // word, tag -> base suffix rules interpreter
    FASuffixInterpretToolsConst_t < Ty > * m_pSuffWT2B;
    // base, tag -> word suffix rules interpreter
    FASuffixInterpretToolsConst_t < Ty > * m_pSuffB2WT;
};


template < class Ty >
FAStemmerConst_t< Ty >::FAStemmerConst_t () :
    m_pLDB (NULL),
    m_Ready_w2b (false),
    m_Ready_b2w (false),
    m_Ready_w2w (false),
    m_Ready_wt2b (false),
    m_Ready_b2wt (false),
    m_Ready_wtt2w (false),
    m_pSuffW2B (NULL),
    m_pSuffB2W (NULL),
    m_pSuffWT2B (NULL),
    m_pSuffB2WT (NULL)
{
}


template < class Ty >
FAStemmerConst_t< Ty >::~FAStemmerConst_t ()
{
    FAStemmerConst_t< Ty >::Clear ();
}


template < class Ty >
void FAStemmerConst_t< Ty >::Initialize (const FAStemmerLDB * pLDB, FAAllocatorA * pMemMgr)
{
    m_pLDB = pLDB;

    m_Ready_w2b = false;
    m_Ready_b2w = false;
    m_Ready_w2w = false;
    m_Ready_wt2b = false;
    m_Ready_b2wt = false;
    m_Ready_wtt2w = false;

    InitW2B (pMemMgr);
    InitB2W (pMemMgr);
    InitW2W (pMemMgr);
    InitWT2B (pMemMgr);
    InitB2WT (pMemMgr);
    InitWTT2W (pMemMgr);
}


template < class Ty >
void FAStemmerConst_t< Ty >::Clear ()
{
    m_Ready_w2b = false;
    m_Ready_b2w = false;
    m_Ready_w2w = false;
    m_Ready_wt2b = false;
    m_Ready_b2wt = false;
    m_Ready_wtt2w = false;

    if (m_pSuffW2B) {
        delete m_pSuffW2B;
        m_pSuffW2B = NULL;
    }
    if (m_pSuffB2W) {
        delete m_pSuffB2W;
        m_pSuffB2W = NULL;
    }
    if (m_pSuffWT2B) {
        delete m_pSuffWT2B;
        m_pSuffWT2B = NULL;
    }
    if (m_pSuffB2WT) {
        delete m_pSuffB2WT;
        m_pSuffB2WT = NULL;
    }
}


template < class Ty >
void FAStemmerConst_t< Ty >::InitW2B (FAAllocatorA * pMemMgr)
{
    LogAssert (!m_Ready_w2b);

    if (!m_pLDB) {
        return;
    }

    const FAWftConfKeeper * pConf = m_pLDB->GetW2BConf ();

    if (!pConf) {
        return;
    }

    const bool UseNfst = pConf->GetUseNfst ();

    LogAssert (!UseNfst);

    if (!m_pSuffW2B) {
        m_pSuffW2B = NEW FASuffixInterpretToolsConst_t < Ty > ();
        LogAssert (m_pSuffW2B);
    }
    m_pSuffW2B->SetConf (pConf, pMemMgr);

    m_Ready_w2b = pConf->GetRsDfa () && pConf->GetState2Ows () && 
        pConf->GetActs ();
}


template < class Ty >
void FAStemmerConst_t< Ty >::InitB2W (FAAllocatorA * pMemMgr)
{
    LogAssert (!m_Ready_b2w);

    if (!m_pLDB) {
        return;
    }

    const FAWftConfKeeper * pConf = m_pLDB->GetB2WConf ();

    if (!pConf) {
        return;
    }

    const bool UseNfst = pConf->GetUseNfst ();

    LogAssert (!UseNfst);

    if (!m_pSuffB2W) {
        m_pSuffB2W = NEW FASuffixInterpretToolsConst_t < Ty > ();
        LogAssert (m_pSuffB2W);
    }
    m_pSuffB2W->SetConf (pConf, pMemMgr);
    m_Ready_b2w = pConf->GetRsDfa () && pConf->GetState2Ows () &&
        pConf->GetActs ();

}


template < class Ty >
void FAStemmerConst_t< Ty >::InitWT2B (FAAllocatorA * pMemMgr)
{
    LogAssert (!m_Ready_wt2b);

    if (!m_pLDB) {
        return;
    }

    const FAWftConfKeeper * pConf = m_pLDB->GetWT2BConf ();

    if (!pConf) {
        return;
    }

    const bool UseNfst = pConf->GetUseNfst ();

    LogAssert (!UseNfst);

    if (!m_pSuffWT2B) {
        m_pSuffWT2B = NEW FASuffixInterpretToolsConst_t < Ty > ();
        LogAssert (m_pSuffWT2B);
    }
    m_pSuffWT2B->SetConf (pConf, pMemMgr);
    m_Ready_wt2b = pConf->GetRsDfa () && pConf->GetState2Ows () &&
        pConf->GetActs ();
}


template < class Ty >
void FAStemmerConst_t< Ty >::InitB2WT (FAAllocatorA * pMemMgr)
{
    LogAssert (!m_Ready_b2wt);

    if (!m_pLDB) {
        return;
    }

    const FAWftConfKeeper * pConf = m_pLDB->GetB2WTConf ();

    if (!pConf) {
        return;
    }

    const bool UseNfst = pConf->GetUseNfst ();

    LogAssert (!UseNfst);

    if (!m_pSuffB2WT) {
        m_pSuffB2WT = NEW FASuffixInterpretToolsConst_t < Ty > ();
        LogAssert (m_pSuffB2WT);
    }
    m_pSuffB2WT->SetConf (pConf, pMemMgr);
    m_Ready_b2wt = pConf->GetRsDfa () && pConf->GetState2Ows () &&
        pConf->GetActs ();
}


template < class Ty >
void FAStemmerConst_t< Ty >::InitW2W (FAAllocatorA * pMemMgr)
{
    LogAssert (!m_Ready_w2w);

    if (!m_Ready_w2b) {
        InitW2B (pMemMgr);
    }
    if (!m_Ready_b2w) {
        InitB2W (pMemMgr);
    }

    m_Ready_w2w = m_Ready_b2w && m_Ready_w2b;
}


template < class Ty >
void FAStemmerConst_t< Ty >::InitWTT2W (FAAllocatorA * pMemMgr)
{
    DebugLogAssert (!m_Ready_wtt2w);

    if (!m_Ready_wt2b) {
        InitWT2B (pMemMgr);
    }
    if (!m_Ready_b2wt) {
        InitB2WT (pMemMgr);
    }

    m_Ready_wtt2w = m_Ready_wt2b && m_Ready_b2wt;
}


template < class Ty >
const int FAStemmerConst_t< Ty >::
    ProcessW2B (
            const Ty * pIn,
            const int InSize,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        ) const
{
    // not initilized
    LogAssert (m_Ready_w2b);
    // bad input parameters
    LogAssert (0 < InSize && pIn && (pOut || 0 >= MaxOutSize));

    // check if input is too big
    if (FALimits::MaxWordSize < InSize) {
        return -1;
    }

    DebugLogAssert (m_pSuffW2B);
    const int OutSize = m_pSuffW2B->Process (pIn, InSize, pOut, MaxOutSize);

    return OutSize;
}


template < class Ty >
const int FAStemmerConst_t< Ty >::
    ProcessB2W (
            const Ty * pIn,
            const int InSize,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        ) const
{
    // not initilized
    LogAssert (m_Ready_b2w);
    // bad input parameters
    LogAssert (0 < InSize && pIn && (pOut || 0 >= MaxOutSize));

    // check if input is too big
    if (FALimits::MaxWordSize < InSize) {
        return -1;
    }

    DebugLogAssert (m_pSuffB2W);
    const int OutSize = m_pSuffB2W->Process (pIn, InSize, pOut, MaxOutSize);

    return OutSize;
}


template < class Ty >
const int FAStemmerConst_t< Ty >::
    ProcessW2W (
            const Ty * pIn,
            const int InSize,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        ) const
{
    const int MaxBaseListSize = FALimits::MaxWordLen * 2;
    Ty BaseList [MaxBaseListSize];

    // not initilized
    LogAssert (m_Ready_w2w);

    // check if input is too big
    if (FALimits::MaxWordSize < InSize) {
        return -1;
    }

    /// generate list of base forms
    const int BaseListSize =
        FAStemmerConst_t<Ty>::ProcessW2B (pIn, InSize, BaseList, MaxBaseListSize);

    /// see whether generated list of base forms does not fit the buffer
    if (BaseListSize > MaxBaseListSize) {
        /// no stemming, in this case
        return -1;
    }

    /// make interation thru the generated base forms
    int Pos = 0;
    int CurrBasePos = 0;
    int OutSize = 0;

    while (Pos < BaseListSize) {

        if (0 == BaseList [Pos]) {

            const int CurrBaseSize = Pos - CurrBasePos;
            DebugLogAssert (0 < CurrBaseSize);

            /// generate list of word-forms
            const int TmpSize =
                m_pSuffB2W->Process (
                    BaseList + CurrBasePos,
                    CurrBaseSize,
                    pOut + OutSize,
                    MaxOutSize - OutSize
                );

            if (0 < TmpSize) {
                OutSize += TmpSize;
            }

            CurrBasePos = Pos + 1;

        } // of if (0 == BaseList [Pos]) ...

        Pos++;
    }

    return OutSize;
}


template < class Ty >
const int FAStemmerConst_t< Ty >::
    ProcessWT2B (
            const Ty * pIn,
            const int InSize,
            const int FromTag,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        ) const
{
    // not initilized
    LogAssert (m_Ready_wt2b);
    // bad input parameters
    LogAssert (0 < InSize && pIn && (pOut || 0 >= MaxOutSize));

    // check if input is too big
    if (FALimits::MaxWordSize < InSize) {
        return -1;
    }

    DebugLogAssert (m_pSuffWT2B);
    const int OutSize =
        m_pSuffWT2B->Process (pIn, InSize, FromTag, pOut, MaxOutSize);

    //// do not copy the original word to the output
    if (0 >= OutSize) {
        return -1;
    } else {
        return OutSize;
    }
}


template < class Ty >
const int FAStemmerConst_t< Ty >::
    ProcessB2WT (
            const Ty * pIn,
            const int InSize,
            const int ToTag,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        ) const
{
    // not initilized
    LogAssert (m_Ready_b2wt);
    // bad input parameters
    LogAssert (0 < InSize && pIn && (pOut || 0 >= MaxOutSize));

    // check if input is too big
    if (FALimits::MaxWordSize < InSize) {
        return -1;
    }

    DebugLogAssert (m_pSuffB2WT);
    const int OutSize =
        m_pSuffB2WT->Process (pIn, InSize, ToTag, pOut, MaxOutSize);

    //// do not copy the original word to the output
    if (0 >= OutSize) {
        return -1;
    } else {
        return OutSize;
    }
}


template < class Ty >
const int FAStemmerConst_t< Ty >::
    ProcessWTT2W (
            const Ty * pIn,
            const int InSize,
            const int FromTag,
            const int ToTag,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        ) const
{
    const int MaxBaseListSize = FALimits::MaxWordLen * 2;
    Ty BaseList [MaxBaseListSize];

    // not initilized
    LogAssert (m_Ready_wtt2w);

    // check if input is too big
    if (FALimits::MaxWordSize < InSize) {
        return -1;
    }

    /// generate list of base forms
    const int BaseListSize =
        FAStemmerConst_t::ProcessWT2B (pIn, InSize, FromTag, BaseList, MaxBaseListSize);

    /// see whether generated list of base forms does not fit the buffer
    if (BaseListSize > MaxBaseListSize) {
        /// no stemming, in this case
        return -1;
    }

    /// make interation thru the generated base forms
    int Pos = 0;
    int CurrBasePos = 0;
    int OutSize = 0;

    while (Pos < BaseListSize) {

        if (0 == BaseList [Pos]) {

            const int CurrBaseSize = Pos - CurrBasePos;
            DebugLogAssert (0 < CurrBaseSize);

            /// generate list of word-forms
            const int TmpSize =
                m_pSuffB2WT->Process (
                    BaseList + CurrBasePos,
                    CurrBaseSize,
                    ToTag,
                    pOut + OutSize,
                    MaxOutSize - OutSize
                );

            if (0 < TmpSize) {
                OutSize += TmpSize;
            }

            CurrBasePos = Pos + 1;

        } // of if (0 == BaseList [Pos]) ...

        Pos++;
    }

    //// do not copy the original word to the output
    if (0 >= OutSize) {
        return -1;
    } else {
        return OutSize;
    }
}

}

#endif
