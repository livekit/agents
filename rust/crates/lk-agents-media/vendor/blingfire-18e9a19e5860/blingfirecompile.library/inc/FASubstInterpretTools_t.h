/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_SUBSTINTERPRETTOOLS_H_
#define _FA_SUBSTINTERPRETTOOLS_H_

#include "FAConfig.h"
#include "FABrResultA.h"
#include "FAArray_cont_t.h"
#include "FAAutInterpretTools_trbr_t.h"
#include "FAUtf8Utils.h"
#include "FASecurity.h"

namespace BlingFire
{

class FAAllocatorA;
class FARSNfaCA;

///
/// This class is responsible for substituter rules interpretation.
///
/// Note:
/// 1. All Process functions return the size of the output chain
///    if no rule matches the input then output chain size is -1.
///

template < class Ty >
class FASubstInterpretTools_t : public FABrResultA {

public:
    FASubstInterpretTools_t (FAAllocatorA * pAlloc);
    virtual ~FASubstInterpretTools_t ();

public:
    /// sets up whether rules reversed or not
    void SetReverse (const bool Reverse);
    /// sets up Moore's RS part of automaton
    void SetRsDfa (const FARSDfaCA * pDfa);
    /// sets up Moore's reaction part of automaton
    void SetState2Ows (const FAState2OwsCA * pState2Ows);
    /// sets up reversed position NFA
    void SetFollow (const FARSNfaCA * pFollow);
    /// sets up regexp-pos to openning brackets
    void SetPos2BrBegin (const FAMultiMapCA * pPos2BrBegin);
    /// sets up regexp-pos to closing brackets
    void SetPos2BrEnd (const FAMultiMapCA * pPos2BrEnd);
    /// sets up rules' actions map
    void SetActs (const FAMultiMapCA * pActs);

    /// returns output string for tagless rules
    /// returns -1 if not matched
    const int Process (
            const Ty * pIn, 
            const int InSize, 
            __out_ecount(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        );
    /// returns output string for one-tags transformation rules
    /// returns -1 if not matched
    const int Process (
            const Ty * pIn,
            const int InSize,
            const int Tag,
            __out_ecount(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        );
    /// returns output string for two-tags transformation rules
    /// returns -1 if not matched
    const int Process (
            const Ty * pIn,
            const int InSize,
            const int FromTag,
            const int ToTag,
            __out_ecount(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        );

public:
    void AddRes (const int TrBr, const int From, const int To);

private:
    inline const int Process_direct (
            const Ty * pIn,
            const int InSize,
            __out_ecount(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        );
    inline const int Process_reverse (
            const Ty * pIn, 
            const int InSize, 
            __out_ecount(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        );
    inline void Clear ();

private:

    FAAutInterpretTools_trbr_t < Ty > m_trbr_interp;
    const FAMultiMapCA * m_pActs;
    // TrBr -> (From,To); (-1,-1) stands for uninitialized pair
    FAArray_cont_t < int > m_trbr2fromto;
    FAArray_cont_t < unsigned int > m_idxs;
    // keeps action to be executed
    int m_ActNum;
    bool m_Reverse;
    FAArray_cont_t < int > m_tmp_chain;
    // special Iws
    enum { 
        AnyIw = 0, 
    };
};


template < class Ty >
FASubstInterpretTools_t< Ty >::FASubstInterpretTools_t (FAAllocatorA * pAlloc) :
    m_trbr_interp (pAlloc),
    m_pActs (NULL),
    m_ActNum (-1),
    m_Reverse (false)
{
    m_trbr2fromto.SetAllocator (pAlloc);
    m_trbr2fromto.Create ();
    m_trbr2fromto.resize (100);
    for (int i = 0; i < 100; ++i) {
        m_trbr2fromto [i] = -1;
    }

    m_idxs.SetAllocator (pAlloc);
    m_idxs.Create ();

    m_trbr_interp.SetAnyIw (AnyIw);

    m_tmp_chain.SetAllocator (pAlloc);
    m_tmp_chain.Create (100);
}


template < class Ty >
FASubstInterpretTools_t< Ty >::~FASubstInterpretTools_t ()
{}


template < class Ty >
void FASubstInterpretTools_t< Ty >::SetRsDfa (const FARSDfaCA * pDfa)
{
    m_trbr_interp.SetRsDfa (pDfa);
}


template < class Ty >
void FASubstInterpretTools_t< Ty >::
    SetState2Ows (const FAState2OwsCA * pState2Ows)
{
    m_trbr_interp.SetState2Ows (pState2Ows);
}


template < class Ty >
void FASubstInterpretTools_t< Ty >::SetFollow (const FARSNfaCA * pFollow)
{
    m_trbr_interp.SetFollow (pFollow);
}


template < class Ty >
void FASubstInterpretTools_t< Ty >::
    SetPos2BrBegin (const FAMultiMapCA * pPos2BrBegin)
{
    m_trbr_interp.SetPos2BrBegin (pPos2BrBegin);
}


template < class Ty >
void FASubstInterpretTools_t< Ty >::
    SetPos2BrEnd (const FAMultiMapCA * pPos2BrEnd)
{
    m_trbr_interp.SetPos2BrEnd (pPos2BrEnd);
}


template < class Ty >
void FASubstInterpretTools_t< Ty >::
    SetActs (const FAMultiMapCA * pActs)
{
    m_pActs = pActs;
}


template < class Ty >
void FASubstInterpretTools_t< Ty >::
    SetReverse (const bool Reverse)
{
    m_Reverse = Reverse;
}


template < class Ty >
void FASubstInterpretTools_t< Ty >::
    AddRes (const int TrBr, const int From, const int To)
{
    const unsigned int OldSize = m_trbr2fromto.size ();

    const int ActNum = (0xffff0000 & (unsigned int)TrBr) >> 16;

    if (0 < ActNum) {
        m_ActNum = ActNum;
    }

    const unsigned int Idx = (0xffff & (unsigned int)TrBr) << 1;

    if (OldSize > Idx) {

        m_trbr2fromto [Idx] = From;
        m_trbr2fromto [Idx + 1] = To;

    } else {

        m_trbr2fromto.resize (Idx + 2);

        int * pDst = m_trbr2fromto.begin ();
        DebugLogAssert (pDst);

        for (unsigned int idx = OldSize; idx < Idx; ++idx) {
            pDst [idx] = -1;
        }

        pDst [Idx] = From;
        pDst [Idx + 1] = To;
    }

    m_idxs.push_back (Idx);
}


template < class Ty >
void FASubstInterpretTools_t< Ty >::
    Clear ()
{
    m_ActNum = -1;

    const int IdxCount = m_idxs.size ();
    int * pTrbr2FromTo = m_trbr2fromto.begin ();

    for (int i = 0; i < IdxCount; ++i) {

        const unsigned int Idx = m_idxs [i];
        DebugLogAssert (Idx < m_trbr2fromto.size ());

        DebugLogAssert (pTrbr2FromTo);
        pTrbr2FromTo [Idx] = -1;
        pTrbr2FromTo [Idx + 1] = -1;
    }

    m_idxs.resize (0);
}


template < class Ty >
inline const int FASubstInterpretTools_t< Ty >::
    Process_direct (
        const Ty * pIn, 
        const int InSize, 
        __out_ecount(MaxOutSize) Ty * pOut,
        const int MaxOutSize
    )
{
    DebugLogAssert (m_pActs);
    DebugLogAssert (pIn && 0 < InSize);

    if (!m_trbr_interp.Chain2BrRes_local (pIn, InSize, this)) {
        return -1;
    }

    int OutSize = 0;

    if (0 < m_ActNum) {

        const int * pActValues;
        const int ActSize = m_pActs->Get (m_ActNum, &pActValues);

        const unsigned int IdxCount = m_trbr2fromto.size ();
        const int * pIdx2FromTo = m_trbr2fromto.begin ();
        DebugLogAssert (pIdx2FromTo);

        for (int i = 0; i < ActSize; ++i) {

            DebugLogAssert (pActValues);
            const int ActValue = pActValues [i];

            if (0 >= ActValue) {

                const int TrBr = -ActValue;
                DebugLogAssert (0 <= TrBr);

                const unsigned int Idx = ((unsigned int)TrBr) << 1;

                if (Idx < IdxCount) {

                    const int From = pIdx2FromTo [Idx];
                    const int To = pIdx2FromTo [Idx + 1];

                    for (int j = From; j <= To; ++j) {

                        OutSize++;
                        if (OutSize <= MaxOutSize) {
                            const int Symbol = pIn [j];
                            *pOut++ = Symbol;
                        }
                    }
                }

            } else {

                OutSize++;
                if (OutSize <= MaxOutSize) {
                    *pOut++ = (Ty) ActValue;
                }
            }
        } // of for (int i = 0; ...
    } // if (0 < m_ActNum) ...

    FASubstInterpretTools_t< Ty >::Clear ();

    return OutSize;
}


template < class Ty >
inline const int FASubstInterpretTools_t< Ty >::
    Process_reverse (
        const Ty * pIn,
        const int InSize,
        __out_ecount(MaxOutSize) Ty * pOut,
        const int MaxOutSize
    )
{
    DebugLogAssert (m_pActs);
    DebugLogAssert (pIn && 0 < InSize);

    if (!m_trbr_interp.Chain2BrRes_local (pIn, InSize, this)) {
        return -1;
    }

    int OutSize = 0;

    if (0 < m_ActNum) {

        const int * pActValues;
        const int ActSize = m_pActs->Get (m_ActNum, &pActValues);

        const unsigned int IdxCount = m_trbr2fromto.size ();
        const int * pIdx2FromTo = m_trbr2fromto.begin ();
        DebugLogAssert (pIdx2FromTo);

        for (int i = 0; i < ActSize; ++i) {

            DebugLogAssert (pActValues);
            const int ActValue = pActValues [i];

            if (0 >= ActValue) {

                const int TrBr = -ActValue;
                DebugLogAssert (0 <= TrBr);

                const unsigned int Idx = ((unsigned int)TrBr) << 1;

                if (Idx < IdxCount) {

                    const int From = pIdx2FromTo [Idx];
                    const int To = pIdx2FromTo [Idx + 1];

                    for (int j = To; j >= From; --j) {

                        OutSize++;
                        if (OutSize <= MaxOutSize) {
                            const int Symbol = pIn [j];
                            *pOut++ = Symbol;
                        }
                    }
                }

            } else {

                OutSize++;
                if (OutSize <= MaxOutSize) {
                    *pOut++ = (Ty) ActValue;
                }
            }
        } // of for (int i = 0; ...
    } // if (0 < m_ActNum) ...

    FASubstInterpretTools_t< Ty >::Clear ();

    return OutSize;
}


template < class Ty >
const int FASubstInterpretTools_t< Ty >::
    Process (
        const Ty * pIn,
        const int InSize,
        __out_ecount(MaxOutSize) Ty * pOut,
        const int MaxOutSize
    )
{
    if (false == m_Reverse) {

        const int OutSize = \
            Process_direct (pIn, InSize, pOut, MaxOutSize);
        return OutSize;

    } else {

        m_tmp_chain.resize (InSize);

        int * pRevChain = m_tmp_chain.begin ();
        DebugLogAssert (pRevChain);
        DebugLogAssert (pIn);

        for (int i = 0; i < InSize; ++i) {

            const int Symbol = pIn [i];
            pRevChain [InSize - i - 1] = Symbol;
        }

        const int OutSize = \
            Process_reverse (pRevChain, InSize, pOut, MaxOutSize);
        return OutSize;
    }
}


template < class Ty >
const int FASubstInterpretTools_t< Ty >::
    Process (
        const Ty * pIn,
        const int InSize,
        const int Tag,
        __out_ecount(MaxOutSize) Ty * pOut,
        const int MaxOutSize
    )
{
    DebugLogAssert (pIn && 0 < InSize);
    DebugLogAssert (-1 != Tag);

    m_tmp_chain.resize (InSize + 2);

    int * pBegin = m_tmp_chain.begin ();
    DebugLogAssert (pBegin);

    if (false == m_Reverse) {

        int * pChain = pBegin;

        *pChain++ = Tag;

        for (int i = 0; i < InSize; ++i) {
            const int Symbol = pIn [i];
            *pChain++ = Symbol;
        }

        *pChain++ = Tag;

        const int OutSize = \
            Process_direct (pBegin, InSize + 2, pOut, MaxOutSize);
        return OutSize;

    } else {

        int * pRevChain = pBegin;

        *pRevChain++ = Tag;

        for (int i = 0; i < InSize; ++i) {
            const int Symbol = pIn [i];
            pRevChain [InSize - (i + 1)] = Symbol;
        }

        pRevChain [InSize] = Tag;

        const int OutSize = \
            Process_reverse (pBegin, InSize + 2, pOut, MaxOutSize);
        return OutSize;
    }
}


template < class Ty >
const int FASubstInterpretTools_t< Ty >::
    Process (
        const Ty * pIn,
        const int InSize,
        const int FromTag,
        const int ToTag,
        __out_ecount(MaxOutSize) Ty * pOut,
        const int MaxOutSize
    )
{
    DebugLogAssert (pIn && 0 < InSize);
    DebugLogAssert (-1 != FromTag && -1 != ToTag);

    m_tmp_chain.resize (InSize + 4);

    int * pBegin = m_tmp_chain.begin ();
    DebugLogAssert (pBegin);

    if (false == m_Reverse) {

        int * pChain = pBegin;

        *pChain++ = FromTag;
        *pChain++ = ToTag;

        for (int i = 0; i < InSize; ++i) {
            const int Symbol = pIn [i];
            *pChain++ = Symbol;
        }

        *pChain++ = FromTag;
        *pChain++ = ToTag;

        const int OutSize = \
            Process_direct (pBegin, InSize + 4, pOut, MaxOutSize);
        return OutSize;

    } else {

        int * pRevChain = pBegin;

        *pRevChain++ = FromTag;
        *pRevChain++ = ToTag;

        for (int i = 0; i < InSize; ++i) {
            const int Symbol = pIn [i];
            pRevChain [InSize - (i + 1)] = Symbol;
        }

        pRevChain [InSize] = FromTag;
        pRevChain [InSize + 1] = ToTag;

        const int OutSize = \
            Process_reverse (pBegin, InSize + 4, pOut, MaxOutSize);
        return OutSize;
    }
}

}

#endif
