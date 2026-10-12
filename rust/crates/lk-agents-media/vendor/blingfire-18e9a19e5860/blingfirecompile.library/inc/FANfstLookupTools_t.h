/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_NFSTLOOKUPTOOLS_T_H_
#define _FA_NFSTLOOKUPTOOLS_T_H_

#include "FAConfig.h"
#include "FARSDfaCA.h"
#include "FAGetIWsCA.h"
#include "FAMultiMapCA.h"
#include "FATransformCA_t.h"
#include "FAArray_cont_t.h"
#include "FAWftConfKeeper.h"
#include "FAUtf32Utils.h"
#include "FASecurity.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// For the given NFST (== RSDFA with <Iw, Ow> labels) and the given sequence
/// of Iws, returns a list of corresponding Ows.
///

template < class Ty >
class FANfstLookupTools_t {

public:
    FANfstLookupTools_t (FAAllocatorA * pAlloc);

public:
    /// sets up the configuration
    void SetConf (const FAWftConfKeeper * pConf);
    /// sets up input transformation, can be NULL
    void SetInTr (const FATransformCA_t < Ty > * pInTr);
    /// sets up output transformation, can be NULL
    void SetOutTr (const FATransformCA_t < Ty > * pOutTr);

    /// returns output strings for tagless input string
    /// returns -1 if no transformation exist
    const int Process (
            const Ty * pIn, 
            const int InSize, 
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        );

private:
    void SetRsDfa (const FARSDfaCA * pDfa);
    void SetGetIws (const FAGetIWsCA * pIws);
    void SetOwsMap (const FAMultiMapCA * pOwsMap);
    void SetIgnoreCase (const bool IgnoreCase);
    // applies input transformation
    inline void Normalize (
            __out_ecount(MaxOutSize) Ty * pOut, 
            const int MaxOutSize
        );
    inline void EnsurePush (const int Size);

private:
    /// data objects
    const FARSDfaCA * m_pDfa;
    const FAGetIWsCA * m_pIws;
    const FAMultiMapCA * m_pOwsMap;
    int m_Initial;

    // normalization, if any
    const FATransformCA_t < Ty > * m_pInTr;
    const FATransformCA_t < Ty > * m_pOutTr;
    bool m_IgnoreCase;

    // input sequence
    const Ty * m_pIn;
    int m_InSize;

    // Ows buffer
    FAArray_cont_t < int > m_ows;
    int m_MaxOwsSize;
    int * m_pOws;

    // stack buffer
    FAArray_cont_t < int > m_stack;
    int m_StackSize;
    int m_StackPtr; // points to the next available index in m_pStack
    int * m_pStack;

    /// charmap
    const FAMultiMapCA * m_pCharMap;

    enum {
        DuplicateOw = 2,
        EpsilonOw = FAFsmConst::IW_EPSILON,
        LeftAnchor = FAFsmConst::IW_L_ANCHOR,
        DefStackSize = 10240,
        MaxStackSize = 1000000,
     };
};


template < class Ty >
FANfstLookupTools_t < Ty >::
    FANfstLookupTools_t (FAAllocatorA * pAlloc) :
        m_pDfa (NULL),
        m_pIws (NULL),
        m_pOwsMap (NULL),
        m_Initial (0),
        m_pInTr (NULL),
        m_pOutTr (NULL),
        m_IgnoreCase (false),
        m_pIn (NULL),
        m_InSize (0),
        m_MaxOwsSize (0),
        m_pOws (NULL),
        m_StackSize (0),
        m_StackPtr (0),
        m_pStack (NULL),
        m_pCharMap (NULL)
{
    m_ows.SetAllocator (pAlloc);
    m_ows.Create ();

    m_stack.SetAllocator (pAlloc);
    m_stack.Create (DefStackSize);
    m_stack.resize (DefStackSize);

    m_StackSize = DefStackSize;
    m_pStack = m_stack.begin ();
}


template < class Ty >
void FANfstLookupTools_t < Ty >::SetConf (const FAWftConfKeeper * pConf)
{
    m_pDfa = NULL;
    m_pIws = NULL;
    m_pOwsMap = NULL;
    m_Initial = 0;
    m_IgnoreCase = false;
    m_pCharMap = NULL;

    if (!pConf)
        return;

    const FARSDfaCA * pDfa = pConf->GetRsDfa ();
    const FAMultiMapCA * pActs = pConf->GetActs ();
    const FAGetIWsCA * pIws = pConf->GetIws ();
    const bool IgnoreCase = pConf->GetIgnoreCase ();
    const bool UseNfst = pConf->GetUseNfst ();
    m_pCharMap = pConf->GetCharMap ();

    FAAssert (UseNfst, FAMsg::InitializationError);

    if (pDfa && pIws && pActs) {
        SetRsDfa (pDfa);
        SetGetIws (pIws);
        SetOwsMap (pActs);
        SetIgnoreCase (IgnoreCase);
    }
}


template < class Ty >
void FANfstLookupTools_t < Ty >::SetRsDfa (const FARSDfaCA * pDfa)
{
    DebugLogAssert (pDfa);

    m_pDfa = pDfa;

    m_Initial = pDfa->GetInitial ();
    FAAssert (0 <= m_Initial, FAMsg::InitializationError);

    m_MaxOwsSize = m_pDfa->GetIWs (NULL, 0);
    FAAssert (0 < m_MaxOwsSize, FAMsg::InitializationError);
    m_ows.resize (m_MaxOwsSize);
    m_pOws = m_ows.begin ();

}

template < class Ty >
void FANfstLookupTools_t < Ty >::SetGetIws (const FAGetIWsCA * pIws)
{
    m_pIws = pIws;
}

template < class Ty >
void FANfstLookupTools_t < Ty >::SetOwsMap (const FAMultiMapCA * pOwsMap)
{
    m_pOwsMap = pOwsMap;
}

template < class Ty >
void FANfstLookupTools_t< Ty >::SetIgnoreCase (const bool IgnoreCase)
{
    m_IgnoreCase = IgnoreCase;
}

template < class Ty >
void FANfstLookupTools_t< Ty >::
    SetInTr (const FATransformCA_t < Ty > * pInTr)
{
    m_pInTr = pInTr;
}

template < class Ty >
void FANfstLookupTools_t< Ty >::
    SetOutTr (const FATransformCA_t < Ty > * pOutTr)
{
    m_pOutTr = pOutTr;
}


template < class Ty >
inline void FANfstLookupTools_t< Ty >::
    Normalize (__out_ecount(MaxOutSize) Ty * pOut, const int MaxOutSize)
{
    DebugLogAssert (m_pIn && 0 < m_InSize && m_InSize <= FALimits::MaxWordLen);
    __analysis_assume (m_pIn && 0 < m_InSize && m_InSize <= FALimits::MaxWordLen);
    DebugLogAssert (pOut && 0 < MaxOutSize);
    __analysis_assume (pOut && 0 < MaxOutSize);
    DebugLogAssert (MaxOutSize >= m_InSize);
    __analysis_assume (MaxOutSize >= m_InSize);

    // normalize the case, if needed
    if (m_IgnoreCase) {
        for (int i = 0 ; i < m_InSize; ++i) {
            const int InSymbol = (int) m_pIn [i] ;
            const int OutSymbol = FAUtf32ToLower (InSymbol) ;
            pOut [i] = (Ty) OutSymbol ;
        }
        m_pIn = pOut;
    }

    // normalize characters
    if (m_pCharMap) {
        // in-place is fine
        m_InSize = FANormalizeWord (m_pIn, m_InSize, \
            pOut, MaxOutSize, m_pCharMap);
        m_pIn = pOut;
    }

    // apply the transformation, if needed
    if (m_pInTr) {

        // in-place is fine
        const int OutSize = \
            m_pInTr->Process (m_pIn, m_InSize, pOut, MaxOutSize);

        // see whether there was some transformation made
        if (0 < OutSize && OutSize <= MaxOutSize) {
            m_pIn = pOut;
            m_InSize = OutSize;
        }
    }
}

template < class Ty >
inline void FANfstLookupTools_t < Ty >::EnsurePush (const int Size)
{
    DebugLogAssert (0 < Size && 0 <= m_StackPtr);

    if (m_StackSize < m_StackPtr + Size) {
        m_StackSize += (Size + DefStackSize);
        m_stack.resize (m_StackSize);
        m_pStack = m_stack.begin ();
    }
}

template < class Ty >
const int FANfstLookupTools_t < Ty >::
    Process (
            const Ty * pIn, 
            const int InSize, 
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        )
{
    const int TmpBuffSize = FALimits::MaxWordLen;

    Ty TmpBuff [TmpBuffSize];
    int Out [TmpBuffSize];

    m_pIn = pIn;
    m_InSize = InSize;

    DebugLogAssert (m_pDfa && m_pOwsMap && m_pIws && -1 != m_Initial);
    __analysis_assume (m_pDfa && m_pOwsMap && m_pIws && -1 != m_Initial);
    DebugLogAssert (m_pIn && 0 < m_InSize && m_InSize <= FALimits::MaxWordLen);
    __analysis_assume (m_pIn && 0 < m_InSize && m_InSize <= FALimits::MaxWordLen);

    Normalize (TmpBuff, TmpBuffSize);

    int OutSize = 0;

    m_StackPtr = 0;

    /// fill in the stack, stack keeps <State, Ow, Pos> tuples
    EnsurePush (3);
    m_pStack [m_StackPtr + 0] = m_Initial;
    m_pStack [m_StackPtr + 1] = -1;
    m_pStack [m_StackPtr + 2] = -1;
    m_StackPtr += 3;

    while (0 < m_StackPtr) {

        DebugLogAssert (0 == m_StackPtr % 3);

        const int Pos = m_pStack [m_StackPtr - 1];
        const int Ow = m_pStack [m_StackPtr - 2];
        const int State = m_pStack [m_StackPtr - 3];

        m_StackPtr -= 3;

        /// set the Ow at the proper place
        if (0 <= Pos) {
            Out [Pos] = Ow;
        }

        /// see if the State is final and all Iws have been accepted
        if (Pos + 1 == m_InSize && m_pDfa->IsFinal (State)) {

            const int PrevOutSize = OutSize;

            for (int i = 0; i < Pos + 1; ++i) {

                // get a sequence of characters
                const int * pC;
                const int Count = m_pOwsMap->Get (Out [i], &pC);
                DebugLogAssert (0 < Count && pC);

                // process it
                for (int j = 0; j < Count; ++j) {
                    const int C = pC [j];
                    if (DuplicateOw == C) {
                        const int Iw = m_pIn [i];
                        pOut [OutSize++] = Iw;
                    } else if (EpsilonOw != C) {
                        pOut [OutSize++] = C;
                    }
                }
            } // for (int i = 0; ...

            const int Len = OutSize - PrevOutSize;

            if (0 < Len) {
                /// apply out-transformation in-place
                if (m_pOutTr) {
                    // uses - 1 as we'll need the last position for 0
                    const int NewLen = m_pOutTr->Process (pOut + PrevOutSize, \
                        Len, pOut + PrevOutSize, MaxOutSize - PrevOutSize - 1);
                    // see if transformation was successfull
                    if (-1 != NewLen && NewLen < MaxOutSize - PrevOutSize - 1){
                        OutSize = PrevOutSize + NewLen;
                    }
                }
                /// add 0-delimiter
                pOut [OutSize++] = 0;
            } // of if (0 < Len) ...

        } // of if (Pos + 1 == m_InSize ...

        /// add tuples corresponding to the next Iw
        if (Pos + 1 < m_InSize) {

            const int Iw = m_pIn [Pos + 1];
            const int NewState = m_pDfa->GetDest (State, Iw);

            if (-1 != NewState) {

                const int OwsCount = \
                    m_pIws->GetIWs (NewState, m_pOws, m_MaxOwsSize);
                DebugLogAssert (OwsCount <= m_MaxOwsSize);

                EnsurePush (3 * OwsCount);

                for (int i = 0; i < OwsCount; ++i) {

                    const int Ow2 = m_pOws [i];
                    const int NewState2 = m_pDfa->GetDest (NewState, Ow2);
                    DebugLogAssert (-1 != NewState2);

                    m_pStack [m_StackPtr + 0] = NewState2;
                    m_pStack [m_StackPtr + 1] = Ow2;
                    m_pStack [m_StackPtr + 2] = Pos + 1;
                    m_StackPtr += 3;

                } // of for (int i = 0; ...

            } // of if (-1 != NewState) {

        } // of if (IwPos + 1 < m_InSize) ...

    } // of while (0 < m_StackPtr) ...

    return OutSize;
}

}

#endif
