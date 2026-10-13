/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_PRMINTERPRETER_T_H_
#define _FA_PRMINTERPRETER_T_H_

#include "FAConfig.h"
#include "FAFsmConst.h"
#include "FALimits.h"
#include "FAMorphLDB_t_packaged.h"
#include "FAWordGuesser_t.h"
#include "FASuffixInterpretTools_t.h"
#include "FASegmentationTools_bf_t.h"
#include "FANfstLookupTools_t.h"
#include "FAUtf32Utils.h"
#include "FAUtils_cl.h"
#include "FASecurity.h"

namespace BlingFire
{

///
/// PRM run-time interpreter.
///

template < class Ty >
class FAPrmInterpreter_t {

public:
    /// creates uninitialized object
    FAPrmInterpreter_t (FAAllocatorA * pAlloc);
    ~FAPrmInterpreter_t ();

public:
    void SetLDB (const FAMorphLDB_t < Ty > * pLDB);

public:
    /// word -> { word-tag }
    const int ProcessW2T (
            const Ty * pIn,
            const int InSize,
            const int ** ppTags
        );

    /// base -> { word-tag }
    const int ProcessB2T (
            const Ty * pIn,
            const int InSize,
            const int ** ppTags
        );

    /// word -> { E1, ..., En }
    /// the output is a set of ending positions of segments
    const int ProcessW2S (
            const Ty * pIn,
            const int InSize,
            __out_ecount_opt(MaxOutSize) int * pOut,
            const int MaxOutSize
        );

    /// word -> { base_form }, word can be on of the base_form too
    /// returns -1 if no transformation exist
    const int ProcessW2B (
            const Ty * pIn,
            const int InSize,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        );
    /// base_form -> { word }, { word } includes base_form too
    /// returns -1 if no transformation exist
    const int ProcessB2W (
            const Ty * pIn,
            const int InSize,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        );
    /// word -> { word }, word can be on of the base_form too
    /// returns -1 if no transformation exist
    const int ProcessW2W (
            const Ty * pIn,
            const int InSize,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        );

    /// word, from_tag -> { base }
    /// returns -1 if no transformation exist
    const int ProcessWT2B (
            const Ty * pIn,
            const int InSize,
            const int FromTag,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        );
    /// base, to_tag -> { word }
    /// returns -1 if no transformation exist
    const int ProcessB2WT (
            const Ty * pIn,
            const int InSize,
            const int ToTag,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        );
    /// word, from_tag, to_tag -> { word }
    /// returns -1 if no transformation exist
    const int ProcessWTT2W (
            const Ty * pIn,
            const int InSize,
            const int FromTag,
            const int ToTag,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        );

private:
    // returns object into initial state
    inline void Clear ();
    // makes initialization and takes care of m_Ready_w2t flag
    inline void InitW2T ();
    // makes initialization and takes care of m_Ready_b2t flag
    inline void InitB2T ();
    // makes initialization and takes care of m_Ready_w2s flag
    inline void InitW2S ();
    // makes initialization and takes care of m_Ready_w2b flag
    inline void InitW2B ();
    // makes initialization and takes care of m_Ready_b2w flag
    inline void InitB2W ();
    // makes initialization and takes care of m_Ready_w2w flag
    inline void InitW2W ();
    // makes initialization and takes care of m_Ready_wt2b flag
    inline void InitWT2B ();
    // makes initialization and takes care of m_Ready_b2wt flag
    inline void InitB2WT ();
    // makes initialization and takes care of m_Ready_wtt2w flag
    inline void InitWTT2W ();

private:
    // keeps morphology resources
    const FAMorphLDB_t < Ty > * m_pLDB;
    FAAllocatorA * m_pAlloc;

    // true if object was initialized
    bool m_Ready_w2t;
    bool m_Ready_b2t;
    bool m_Ready_w2s;
    bool m_Ready_w2b;
    bool m_Ready_b2w;
    bool m_Ready_w2w;
    bool m_Ready_wt2b;
    bool m_Ready_b2wt;
    bool m_Ready_wtt2w;

    // word-guesser interpreter
    FAWordGuesser_t < Ty > * m_pW2T;
    FAWordGuesser_t < Ty > * m_pB2T;
    // detection of word segments
    FASegmentationTools_bf_t < Ty >  * m_pW2S;
    // word -> base suffix rules interpreter
    FASuffixInterpretTools_t < Ty > * m_pSuffW2B;
    // base -> word suffix rules interpreter
    FASuffixInterpretTools_t < Ty > * m_pSuffB2W;
    // word, tag -> base suffix rules interpreter
    FASuffixInterpretTools_t < Ty > * m_pSuffWT2B;
    // base, tag -> word suffix rules interpreter
    FASuffixInterpretTools_t < Ty > * m_pSuffB2WT;

    // word -> base NFST
    FANfstLookupTools_t < Ty > * m_pNfstW2B;
    // base -> word NFST
    FANfstLookupTools_t < Ty > * m_pNfstB2W;

    // buffer of temporary intermediate base forms, needed by w2w function
    enum { DefBaseListSize = FALimits::MaxWordLen * 5 };
    FAArray_cont_t < Ty > m_base_list;
    Ty * m_pBaseList;
    int m_MaxBaseListSize;
};


template < class Ty >
FAPrmInterpreter_t< Ty >::FAPrmInterpreter_t (FAAllocatorA * pAlloc) :
    m_pLDB (NULL),
    m_pAlloc (pAlloc),
    m_Ready_w2t (false),
    m_Ready_b2t (false),
    m_Ready_w2s (false),
    m_Ready_w2b (false),
    m_Ready_b2w (false),
    m_Ready_w2w (false),
    m_Ready_wt2b (false),
    m_Ready_b2wt (false),
    m_Ready_wtt2w (false),
    m_pW2T (NULL),
    m_pB2T (NULL),
    m_pW2S (NULL),
    m_pSuffW2B (NULL),
    m_pSuffB2W (NULL),
    m_pSuffWT2B (NULL),
    m_pSuffB2WT (NULL),
    m_pNfstW2B (NULL),
    m_pNfstB2W (NULL),
    m_pBaseList (NULL),
    m_MaxBaseListSize (0)
{
    m_base_list.SetAllocator (pAlloc);
    m_base_list.Create (DefBaseListSize);
    m_base_list.resize (DefBaseListSize);

    m_pBaseList = m_base_list.begin ();
    m_MaxBaseListSize = DefBaseListSize;
}


template < class Ty >
FAPrmInterpreter_t< Ty >::~FAPrmInterpreter_t ()
{
    FAPrmInterpreter_t< Ty >::Clear ();
}


template < class Ty >
void FAPrmInterpreter_t< Ty >::SetLDB (const FAMorphLDB_t < Ty > * pLDB)
{
    m_pLDB = pLDB;

    m_Ready_w2t = false;
    m_Ready_b2t = false;
    m_Ready_w2s = false;
    m_Ready_w2b = false;
    m_Ready_b2w = false;
    m_Ready_w2w = false;
    m_Ready_wt2b = false;
    m_Ready_b2wt = false;
    m_Ready_wtt2w = false;
}


template < class Ty >
void FAPrmInterpreter_t< Ty >::Clear ()
{
    m_Ready_w2t = false;
    m_Ready_b2t = false;
    m_Ready_w2s = false;
    m_Ready_w2b = false;
    m_Ready_b2w = false;
    m_Ready_w2w = false;
    m_Ready_wt2b = false;
    m_Ready_b2wt = false;
    m_Ready_wtt2w = false;

    if (m_pW2T) {
        delete m_pW2T;
        m_pW2T = NULL;
    }
    if (m_pB2T) {
        delete m_pB2T;
        m_pB2T = NULL;
    }
    if (m_pW2S) {
        delete m_pW2S;
        m_pW2S = NULL;
    }
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
    if (m_pNfstW2B) {
        delete m_pNfstW2B;
        m_pNfstW2B = NULL;
    }
    if (m_pNfstB2W) {
        delete m_pNfstB2W;
        m_pNfstB2W = NULL;
    }
}


template < class Ty >
void FAPrmInterpreter_t< Ty >::InitW2T ()
{
    DebugLogAssert (!m_Ready_w2t);

    if (!m_pLDB) {
        return;
    }

    if (!m_pW2T) {
        m_pW2T = NEW FAWordGuesser_t < Ty > ();
        LogAssert (m_pW2T);
    }

    const FAWgConfKeeper * pConf = m_pLDB->GetW2TConf (); // may be NULL
    const FATransformCA_t < Ty > * pInTr = m_pLDB->GetInTr (); // may be NULL

    m_pW2T->Initialize (pConf, pInTr);

    m_Ready_w2t = pConf && pConf->GetRsDfa () && pConf->GetState2Ows ();
}


template < class Ty >
const int FAPrmInterpreter_t< Ty >::
    ProcessW2T (
            const Ty * pIn,
            const int InSize,
            const int ** ppTags
        )
{
    if (!m_Ready_w2t) {
        InitW2T ();
        if (!m_Ready_w2t)
            return -1;
    }

    // check for limits
    if (0 >= InSize || FALimits::MaxWordSize < InSize || !pIn) {
        return -1;
    }

    DebugLogAssert (m_pW2T);
    const int TagCount = m_pW2T->Process (pIn, InSize, ppTags);

    return TagCount;
}


template < class Ty >
void FAPrmInterpreter_t< Ty >::InitB2T ()
{
    DebugLogAssert (!m_Ready_b2t);

    if (!m_pLDB) {
        return;
    }

    if (!m_pB2T) {
        m_pB2T = NEW FAWordGuesser_t < Ty > ();
        LogAssert (m_pB2T);
    }

    const FAWgConfKeeper * pConf = m_pLDB->GetB2TConf (); // may be NULL
    const FATransformCA_t < Ty > * pInTr = m_pLDB->GetInTr (); // may be NULL

    m_pB2T->Initialize (pConf, pInTr);

    m_Ready_b2t = pConf && pConf->GetRsDfa () && pConf->GetState2Ows ();
}


template < class Ty >
const int FAPrmInterpreter_t< Ty >::
    ProcessB2T (
            const Ty * pIn,
            const int InSize,
            const int ** ppTags
        )
{
    if (!m_Ready_b2t) {
        InitB2T ();
        if (!m_Ready_b2t)
            return -1;
    }

    // check for limits
    if (0 >= InSize || FALimits::MaxWordSize < InSize || !pIn)
        return -1;

    DebugLogAssert (m_pB2T);
    const int TagCount = m_pB2T->Process (pIn, InSize, ppTags);

    return TagCount;
}


template < class Ty >
void FAPrmInterpreter_t< Ty >::InitW2S ()
{
    DebugLogAssert (!m_Ready_w2s);

    if (!m_pLDB) {
        return;
    }

    const FAW2SConfKeeper * pConf = m_pLDB->GetW2SConf ();

    if (!pConf) {
        return;
    }

    const FARSDfaCA * pDfa = pConf->GetRsDfa ();
    const FAState2OwCA * pState2Ow = pConf->GetState2Ow ();

    // check whether there were enough parameters specified for operation
    if (pDfa && pState2Ow) {

        if (!m_pW2S) {
            m_pW2S = NEW FASegmentationTools_bf_t < Ty > ();
            LogAssert (m_pW2S);
        }

        m_pW2S->SetConf (pConf);

        m_Ready_w2s = true;
    }
}


template < class Ty >
const int FAPrmInterpreter_t< Ty >::
    ProcessW2S (
            const Ty * pIn,
            const int InSize,
            __out_ecount_opt(MaxOutSize) int * pOut,
            const int MaxOutSize
        )
{
    if (!m_Ready_w2s) {
        InitW2S ();
        if (!m_Ready_w2s)
            return -1;
    }

    // check for limits
    if (0 >= InSize || FALimits::MaxWordSize < InSize || !pIn || \
        (NULL == pOut && 0 != MaxOutSize))
        return -1;

    DebugLogAssert (m_pW2S);
    const int SegCount = m_pW2S->Process (pIn, InSize, pOut, MaxOutSize);

    return SegCount;
}


template < class Ty >
void FAPrmInterpreter_t< Ty >::InitW2B ()
{
    DebugLogAssert (!m_Ready_w2b);

    if (!m_pLDB) {
        return;
    }

    const FAWftConfKeeper * pConf = m_pLDB->GetW2BConf ();

    if (!pConf) {
        return;
    }

    const bool NoTrUse = pConf->GetNoTrUse ();
    const bool UseNfst = pConf->GetUseNfst ();

    if (UseNfst) {
        if (!m_pNfstW2B) {
            m_pNfstW2B = NEW FANfstLookupTools_t < Ty > (m_pAlloc);
            LogAssert (m_pNfstW2B);
        }
        if (false == NoTrUse) {
            m_pNfstW2B->SetInTr (m_pLDB->GetInTr ());
            m_pNfstW2B->SetOutTr (m_pLDB->GetOutTr ());
        }
        m_pNfstW2B->SetConf (pConf);
        m_Ready_w2b = pConf->GetRsDfa () && pConf->GetIws () && \
            pConf->GetActs ();

    } else {
        if (!m_pSuffW2B) {
            m_pSuffW2B = NEW FASuffixInterpretTools_t < Ty > ();
            LogAssert (m_pSuffW2B);
        }
        if (false == NoTrUse) {
            m_pSuffW2B->SetInTr (m_pLDB->GetInTr ());
            m_pSuffW2B->SetOutTr (m_pLDB->GetOutTr ());
        }
        m_pSuffW2B->SetConf (pConf);
        m_Ready_w2b = pConf->GetRsDfa () && pConf->GetState2Ows () && \
            pConf->GetActs ();
    }
}


template < class Ty >
const int FAPrmInterpreter_t< Ty >::
    ProcessW2B (
            const Ty * pIn,
            const int InSize,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        )
{
    if (!m_Ready_w2b) {
        InitW2B ();
        if (!m_Ready_w2b)
            return -1;
    } // of if (!m_Ready_w2b) ...

    // check for limits
    if (0 >= InSize || FALimits::MaxWordSize < InSize || !pIn || \
        (NULL == pOut && 0 != MaxOutSize))
        return -1;

    int OutSize = -1;

    if (m_pSuffW2B) {
        OutSize = m_pSuffW2B->Process (pIn, InSize, pOut, MaxOutSize);
    } else {
        DebugLogAssert (m_pNfstW2B);
        OutSize = m_pNfstW2B->Process (pIn, InSize, pOut, MaxOutSize);
    }

    return OutSize;
}


template < class Ty >
void FAPrmInterpreter_t< Ty >::InitB2W ()
{
    DebugLogAssert (!m_Ready_b2w);

    if (!m_pLDB) {
        return;
    }

    const FAWftConfKeeper * pConf = m_pLDB->GetB2WConf ();

    if (!pConf) {
        return;
    }

    const bool NoTrUse = pConf->GetNoTrUse ();
    const bool UseNfst = pConf->GetUseNfst ();

    if (UseNfst) {
        if (!m_pNfstB2W) {
            m_pNfstB2W = NEW FANfstLookupTools_t < Ty > (m_pAlloc);
            LogAssert (m_pNfstB2W);
        }
        if (false == NoTrUse) {
            m_pNfstB2W->SetInTr (m_pLDB->GetInTr ());
            m_pNfstB2W->SetOutTr (m_pLDB->GetOutTr ());
        }
        m_pNfstB2W->SetConf (pConf);
        m_Ready_b2w = pConf->GetRsDfa () && pConf->GetIws () && \
            pConf->GetActs ();

    } else {
        if (!m_pSuffB2W) {
            m_pSuffB2W = NEW FASuffixInterpretTools_t < Ty > ();
            LogAssert (m_pSuffB2W);
        }
        if (false == NoTrUse) {
            m_pSuffB2W->SetInTr (m_pLDB->GetInTr ());
            m_pSuffB2W->SetOutTr (m_pLDB->GetOutTr ());
        }
        m_pSuffB2W->SetConf (pConf);
        m_Ready_b2w = pConf->GetRsDfa () && pConf->GetState2Ows () && \
            pConf->GetActs ();
    }
}


template < class Ty >
const int FAPrmInterpreter_t< Ty >::
    ProcessB2W (
            const Ty * pIn,
            const int InSize,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        )
{
    if (!m_Ready_b2w) {
        InitB2W ();
        if (!m_Ready_b2w)
            return -1;
    }

    // check for limits
    if (0 >= InSize || FALimits::MaxWordSize < InSize || !pIn || \
        (NULL == pOut && 0 != MaxOutSize))
        return -1;

    int OutSize = -1;

    if (m_pSuffB2W) {
        OutSize = m_pSuffB2W->Process (pIn, InSize, pOut, MaxOutSize);
    } else {
        DebugLogAssert (m_pNfstB2W);
        OutSize = m_pNfstB2W->Process (pIn, InSize, pOut, MaxOutSize);
    }

    return OutSize;
}


template < class Ty >
void FAPrmInterpreter_t< Ty >::InitW2W ()
{
    DebugLogAssert (!m_Ready_w2w);

    if (!m_Ready_w2b) {
        InitW2B ();
    }
    if (!m_Ready_b2w) {
        InitB2W ();
    }

    m_Ready_w2w = m_Ready_b2w && m_Ready_w2b;
}


template < class Ty >
const int FAPrmInterpreter_t< Ty >::
    ProcessW2W (
            const Ty * pIn,
            const int InSize,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        )
{
    DebugLogAssert (m_pBaseList && DefBaseListSize <= m_MaxBaseListSize);

    if (!m_Ready_w2w) {

        InitW2W ();

        if (!m_Ready_w2w)
            return -1;
    }

    // check for limits
    if (0 >= InSize || FALimits::MaxWordSize < InSize || !pIn || \
        (NULL == pOut && 0 != MaxOutSize))
        return -1;

    /// generate list of base forms
    const int BaseListSize = \
        FAPrmInterpreter_t<Ty>::ProcessW2B (pIn, InSize, m_pBaseList, m_MaxBaseListSize);

    /// see whether generated list of base forms does not fit the buffer
    if (BaseListSize > m_MaxBaseListSize) {

        /// just extra check for reasonableness
        DebugLogAssert (BaseListSize < 10 * DefBaseListSize);

        m_base_list.resize (BaseListSize + 100);
        m_pBaseList = m_base_list.begin ();
        m_MaxBaseListSize = BaseListSize + 100;

#ifndef NDEBUG
        const int BaseListSize2 = 
#endif
        FAPrmInterpreter_t<Ty>::ProcessW2B (pIn, InSize, m_pBaseList, m_MaxBaseListSize);
        DebugLogAssert (BaseListSize2 == BaseListSize);
    }

    /// make interation thru the generated base forms
    int Pos = 0;
    int CurrBasePos = 0;
    int OutSize = 0;

    while (Pos < BaseListSize) {

        if (0 == m_pBaseList [Pos]) {

            const int CurrBaseSize = Pos - CurrBasePos;
            DebugLogAssert (0 < CurrBaseSize);

            /// generate list of word-forms
            const int TmpSize = \
                m_pSuffB2W->Process (
                    m_pBaseList + CurrBasePos,
                    CurrBaseSize,
                    pOut + OutSize,
                    MaxOutSize - OutSize
                );

            if (0 < TmpSize) {
                OutSize += TmpSize;
            }

            CurrBasePos = Pos + 1;

        } // of if (0 == m_pBaseList [Pos]) ...

        Pos++;
    }

    return OutSize;
}


template < class Ty >
void FAPrmInterpreter_t< Ty >::InitWT2B ()
{
    DebugLogAssert (!m_Ready_wt2b);

    if (!m_pLDB) {
        return;
    }

    const FAWftConfKeeper * pConf = m_pLDB->GetWT2BConf ();

    if (!pConf) {
        return;
    }

    const bool NoTrUse = pConf->GetNoTrUse ();
    const bool UseNfst = pConf->GetUseNfst ();

    LogAssert (!UseNfst);

    if (!m_pSuffWT2B) {
        m_pSuffWT2B = NEW FASuffixInterpretTools_t < Ty > ();
        LogAssert (m_pSuffWT2B);
    }
    if (false == NoTrUse) {
        m_pSuffWT2B->SetInTr (m_pLDB->GetInTr ());
        m_pSuffWT2B->SetOutTr (m_pLDB->GetOutTr ());
    }
    m_pSuffWT2B->SetConf (pConf);
    m_Ready_wt2b = pConf->GetRsDfa () && pConf->GetState2Ows () && \
        pConf->GetActs ();

}


template < class Ty >
void FAPrmInterpreter_t< Ty >::InitB2WT ()
{
    DebugLogAssert (!m_Ready_b2wt);

    if (!m_pLDB) {
        return;
    }

    const FAWftConfKeeper * pConf = m_pLDB->GetB2WTConf ();

    if (!pConf) {
        return;
    }

    const bool NoTrUse = pConf->GetNoTrUse ();
    const bool UseNfst = pConf->GetUseNfst ();

    LogAssert (!UseNfst);

    if (!m_pSuffB2WT) {
        m_pSuffB2WT = NEW FASuffixInterpretTools_t < Ty > ();
        LogAssert (m_pSuffB2WT);
    }
    if (false == NoTrUse) {
        m_pSuffB2WT->SetInTr (m_pLDB->GetInTr ());
        m_pSuffB2WT->SetOutTr (m_pLDB->GetOutTr ());
    }
    m_pSuffB2WT->SetConf (pConf);
    m_Ready_b2wt = pConf->GetRsDfa () && pConf->GetState2Ows () && \
        pConf->GetActs ();
}


template < class Ty >
const int FAPrmInterpreter_t< Ty >::
    ProcessWT2B (
            const Ty * pIn,
            const int InSize,
            const int FromTag,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        )
{
    if (!m_Ready_wt2b) {
        InitWT2B ();
        if (!m_Ready_wt2b)
            return -1;

    } // of if (!m_Ready_wt2b) ...

    // check for limits
    if (0 >= InSize || FALimits::MaxWordSize < InSize || !pIn || \
        (NULL == pOut && 0 != MaxOutSize))
        return -1;

    DebugLogAssert (m_pSuffWT2B);
    const int OutSize = \
        m_pSuffWT2B->Process (pIn, InSize, FromTag, pOut, MaxOutSize);

    //// do not copy the original word to the output
    if (0 >= OutSize) {
        return -1;
    } else {
        return OutSize;
    }
}


template < class Ty >
const int FAPrmInterpreter_t< Ty >::
    ProcessB2WT (
            const Ty * pIn,
            const int InSize,
            const int ToTag,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        )
{
    if (!m_Ready_b2wt) {
        InitB2WT ();
        if (!m_Ready_b2wt)
            return -1;
    }

    // check for limits
    if (0 >= InSize || FALimits::MaxWordSize < InSize || !pIn || \
        (NULL == pOut && 0 != MaxOutSize))
        return -1;

    DebugLogAssert (m_pSuffB2WT);
    const int OutSize = \
        m_pSuffB2WT->Process (pIn, InSize, ToTag, pOut, MaxOutSize);

    //// do not copy the original word to the output
    if (0 >= OutSize) {
        return -1;
    } else {
        return OutSize;
    }
}


template < class Ty >
void FAPrmInterpreter_t< Ty >::InitWTT2W ()
{
    DebugLogAssert (!m_Ready_wtt2w);

    if (!m_Ready_wt2b) {
        InitWT2B ();
    }
    if (!m_Ready_b2wt) {
        InitB2WT ();
    }

    m_Ready_wtt2w = m_Ready_wt2b && m_Ready_b2wt;
}


template < class Ty >
const int FAPrmInterpreter_t< Ty >::
    ProcessWTT2W (
            const Ty * pIn,
            const int InSize,
            const int FromTag,
            const int ToTag,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        )
{
    DebugLogAssert (m_pBaseList && DefBaseListSize <= m_MaxBaseListSize);

    if (!m_Ready_wtt2w) {
        InitWTT2W ();
        if (!m_Ready_wtt2w)
            return -1;
    }

    // check for limits
    if (0 >= InSize || FALimits::MaxWordSize < InSize || !pIn || \
        (NULL == pOut && 0 != MaxOutSize))
        return -1;

    /// generate list of base forms
    const int BaseListSize = \
        FAPrmInterpreter_t::ProcessWT2B (pIn, InSize, FromTag, m_pBaseList, m_MaxBaseListSize);

    /// see whether generated list of base forms does not fit the buffer
    if (BaseListSize > m_MaxBaseListSize) {

        /// just extra check for reasonableness
        DebugLogAssert (BaseListSize < 10 * DefBaseListSize);

        m_base_list.resize (BaseListSize + 100);
        m_pBaseList = m_base_list.begin ();
        m_MaxBaseListSize = BaseListSize + 100;

#ifndef NDEBUG
        const int BaseListSize2 = 
#endif
        FAPrmInterpreter_t::ProcessWT2B (pIn, InSize, FromTag, m_pBaseList, m_MaxBaseListSize);
        DebugLogAssert (BaseListSize2 == BaseListSize);
    }

    /// make interation thru the generated base forms
    int Pos = 0;
    int CurrBasePos = 0;
    int OutSize = 0;

    while (Pos < BaseListSize) {

        if (0 == m_pBaseList [Pos]) {

            const int CurrBaseSize = Pos - CurrBasePos;
            DebugLogAssert (0 < CurrBaseSize);

            /// generate list of word-forms
            const int TmpSize = \
                m_pSuffB2WT->Process (
                    m_pBaseList + CurrBasePos,
                    CurrBaseSize,
                    ToTag,
                    pOut + OutSize,
                    MaxOutSize - OutSize
                );

            if (0 < TmpSize) {
                OutSize += TmpSize;
            }

            CurrBasePos = Pos + 1;

        } // of if (0 == m_pBaseList [Pos]) ...

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
