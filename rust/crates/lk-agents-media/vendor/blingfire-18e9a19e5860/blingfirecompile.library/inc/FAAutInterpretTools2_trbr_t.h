/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_AUTINTERPRETTOOLS2_TRBR_T_H_
#define _FA_AUTINTERPRETTOOLS2_TRBR_T_H_

#include "FAConfig.h"
#include "FAFsmConst.h"
#include "FABrResultA.h"
#include "FARSDfaCA.h"
#include "FAMealyDfaCA.h"
#include "FAMultiMapCA.h"
#include "FAArray_cont_t.h"
#include "FALimits.h"

namespace BlingFire
{

class FAAllocatorA;


template < class Ty >
class FAAutInterpretTools2_trbr_t {

public:
    FAAutInterpretTools2_trbr_t (FAAllocatorA * pAlloc);

public:
    /// sets up right-to-left automaton
    void SetMealy1 (const FARSDfaCA * pRs1, const FAMealyDfaCA * pMealy1);
    /// sets up left-to-right automaton
    void SetMealy2 (const FARSDfaCA * pRs2, const FAMealyDfaCA * pMealy2);
    /// sets up Ow -> [ TrBr ] mapping
    void SetTrBrMap (const FAMultiMapCA * pOw2TrBr);
    /// sets up tuple size, is used by Process only
    void SetTupleSize (const int TupleSize);
    /// sets up WRE token type, is used by Process2 only
    void SetTokenType (const int TokenType);

public:
    /// Returns true if the input is accepted and fills in pRes.
    const bool Process (
            const Ty * pChain,
            const int Count,
            FABrResultA * pRes
        );
    /// Returns true if the input is accepted and fills in pRes.
    /// Assumes input weight from each digitizer, but TokenType defines which
    /// actually will be used.
    const bool Process2 (
            const Ty * pChain,
            const int Size,
            FABrResultA * pRes
        );

private:
    inline void Ows2TrBr (
            const int * pOws,
            const int Size,
            FABrResultA * pRes,
            const int TupleSize
        );

private:
    const FARSDfaCA * m_pRs1;
    const FAMealyDfaCA * m_pMealy1;
    const FARSDfaCA * m_pRs2;
    const FAMealyDfaCA * m_pMealy2;
    const FAMultiMapCA * m_pOw2TrBr;
    int m_TupleSize;
    int m_TokenType;

    FAArray_cont_t < int > m_ows;
    FAArray_cont_t < int > m_stack;
};


template < class Ty >
FAAutInterpretTools2_trbr_t< Ty >::
    FAAutInterpretTools2_trbr_t (FAAllocatorA * pAlloc) :
    m_pRs1 (NULL),
    m_pMealy1 (NULL),
    m_pRs2 (NULL),
    m_pMealy2 (NULL),
    m_pOw2TrBr (NULL),
    m_TupleSize (1),
    m_TokenType (FAFsmConst::WRE_TT_DEFAULT)
{
    m_ows.SetAllocator (pAlloc);
    m_ows.Create ();

    m_stack.SetAllocator (pAlloc);
    m_stack.Create ();
}


template < class Ty >
void FAAutInterpretTools2_trbr_t< Ty >::
    SetMealy1 (const FARSDfaCA * pRs1, const FAMealyDfaCA * pMealy1)
{
    m_pRs1 = pRs1;
    m_pMealy1 = pMealy1;
}


template < class Ty >
void FAAutInterpretTools2_trbr_t< Ty >::
    SetMealy2 (const FARSDfaCA * pRs2, const FAMealyDfaCA * pMealy2)
{
    m_pRs2 = pRs2;
    m_pMealy2 = pMealy2;
}


template < class Ty >
void FAAutInterpretTools2_trbr_t< Ty >::
    SetTrBrMap (const FAMultiMapCA * pOw2TrBr)
{
    m_pOw2TrBr = pOw2TrBr;
}


template < class Ty >
void FAAutInterpretTools2_trbr_t< Ty >::
    SetTupleSize (const int TupleSize)
{
    DebugLogAssert (0 < TupleSize);
    m_TupleSize = TupleSize;
}


template < class Ty >
void FAAutInterpretTools2_trbr_t< Ty >::
    SetTokenType (const int TokenType)
{
   m_TokenType = TokenType;
}


template < class Ty >
inline void FAAutInterpretTools2_trbr_t< Ty >::
    Ows2TrBr (
        const int * pOws, 
        const int Size, 
        FABrResultA * pRes, 
        const int TupleSize
    )
{
    DebugLogAssert (m_pOw2TrBr);
    DebugLogAssert (0 < TupleSize && FAFsmConst::DIGITIZER_COUNT >= TupleSize);
    DebugLogAssert (0 < Size && pOws);
    DebugLogAssert (pRes);

    m_stack.resize (0);

    for (int i = 0; i < Size; ++i) {

        const int Ow = pOws [i];

        if (-1 != Ow) {

            /// const int Pos = i / TupleSize;
            const int Pos = (i - 1) / TupleSize;

            const int * pTrBrs;
            const int TrBrs = m_pOw2TrBr->Get (Ow, &pTrBrs);
            DebugLogAssert (0 < TrBrs && pTrBrs);

            for (int j = 0; j < TrBrs; ++j) {

                const int TrBr = pTrBrs [j];
                const bool IsLeft = (0 == (TrBr & 1));

                if (IsLeft) {

                    m_stack.push_back (Pos);

                } else {

                    DebugLogAssert (1 <= m_stack.size ());
                    const int LeftPos = m_stack [m_stack.size () - 1];
                    m_stack.pop_back ();

                    if (Pos > LeftPos) {
                        const int TrBrId = TrBr >> 1;
                        pRes->AddRes (TrBrId, LeftPos, Pos - 1);
                    }
                }

            } // of for (int j = 0; ...
        } // of if (-1 != Ow) ...
    } // of for (int i = 0; ...

    DebugLogAssert (0 == m_stack.size ());
}


template < class Ty >
const bool FAAutInterpretTools2_trbr_t< Ty >::
    Process (const Ty * pChain, const int Size, FABrResultA * pRes)
{
    DebugLogAssert (m_pRs1 && m_pMealy1);
    DebugLogAssert (pRes);

    if (0 >= Size || !pChain)
        return false;

    int i;
    const int OwsCount = Size + 2;
    m_ows.resize (OwsCount);
    int * pOws = m_ows.begin ();

    if (m_pRs2) {

        DebugLogAssert (m_pMealy2);

        int State = m_pRs1->GetInitial ();
        pOws += OwsCount;

        State = m_pMealy1->GetDestOw (State, FAFsmConst::IW_EOS, --pOws);
        DebugLogAssert (-1 != State);

        for (i = Size - 1; i >= 0; --i) {

            const int Iw = (const int) pChain [i];
            DebugLogAssert (pOws > m_ows.begin ());
            State = m_pMealy1->GetDestOw (State, Iw, --pOws);
            if (-1 == State)
                return false;
        }

        State = m_pMealy1->GetDestOw (State, FAFsmConst::IW_EOS, --pOws);
        if (-1 == State)
            return false;

        if (!m_pRs1->IsFinal (State))
            return false;

        DebugLogAssert (pOws == m_ows.begin ());
        State = m_pRs2->GetInitial ();

        for (i = 0; i < OwsCount; ++i) {

            const int Ow = pOws [i];
            State = m_pMealy2->GetDestOw (State, Ow, & (pOws [i]));
            if (-1 == State)
                return false;
        }
        if (!m_pRs2->IsFinal (State))
            return false;

    } else {

        int State = m_pRs1->GetInitial ();

        State = m_pMealy1->GetDestOw (State, FAFsmConst::IW_EOS, pOws++);
        DebugLogAssert (-1 != State);

        for (i = 0; i < Size; ++i) {

            const int Iw = (const int) pChain [i];
            State = m_pMealy1->GetDestOw (State, Iw, pOws++);
            if (-1 == State)
                return false;
        }

        State = m_pMealy1->GetDestOw (State, FAFsmConst::IW_EOS, pOws++);
        if (-1 == State)
            return false;

        pOws -= OwsCount;
        DebugLogAssert (pOws == m_ows.begin ());
        DebugLogAssert (m_pRs1->IsFinal (State));

    } // of if (m_pRs2) ...

    Ows2TrBr (pOws, OwsCount, pRes, m_TupleSize);

    return true;
}


template < class Ty >
const bool FAAutInterpretTools2_trbr_t< Ty >::
    Process2 (const Ty * pChain, const int Size, FABrResultA * pRes)
{
    DebugLogAssert (m_pRs1 && m_pMealy1);
    DebugLogAssert (pRes);
    DebugLogAssert (0 == Size % FAFsmConst::DIGITIZER_COUNT);

    if (0 >= Size || !pChain)
        return false;

    const int TupleCount = Size / FAFsmConst::DIGITIZER_COUNT;
    DebugLogAssert (0 < TupleCount);

    int TupleSize = 0;

    if (FAFsmConst::WRE_TT_TEXT & m_TokenType) {
        TupleSize++;
    }
    if (FAFsmConst::WRE_TT_TAGS & m_TokenType) {
        TupleSize++;
    }
    if (FAFsmConst::WRE_TT_DCTS & m_TokenType) {
        TupleSize++;
    }

    // +2 for the pair of FAFsmConst::IW_EOS
    const int OwsCount = (TupleCount * TupleSize) + 2;

    int i;
    m_ows.resize (OwsCount);
    int * pOws = m_ows.begin ();

    if (m_pRs2) {

        DebugLogAssert (m_pMealy2);

        int State = m_pRs1->GetInitial ();
        pOws += OwsCount;

        State = m_pMealy1->GetDestOw (State, FAFsmConst::IW_EOS, --pOws);
        DebugLogAssert (-1 != State);

        for (i = Size - 1; i >= 0; i -= FAFsmConst::DIGITIZER_COUNT) {

            if (FAFsmConst::WRE_TT_DCTS & m_TokenType) {

                const int Iw = (const int) pChain [i];
                DebugLogAssert (pOws > m_ows.begin ());
                State = m_pMealy1->GetDestOw (State, Iw, --pOws);
                if (-1 == State)
                    return false;
            }
            if (FAFsmConst::WRE_TT_TAGS & m_TokenType) {

                const int Iw = (const int) pChain [i - 1];
                DebugLogAssert (pOws > m_ows.begin ());
                State = m_pMealy1->GetDestOw (State, Iw, --pOws);
                if (-1 == State)
                    return false;
            }
            if (FAFsmConst::WRE_TT_TEXT & m_TokenType) {

                const int Iw = (const int) pChain [i - 2];
                DebugLogAssert (pOws > m_ows.begin ());
                State = m_pMealy1->GetDestOw (State, Iw, --pOws);
                if (-1 == State)
                    return false;
            }
        }

        State = m_pMealy1->GetDestOw (State, FAFsmConst::IW_EOS, --pOws);
        if (-1 == State)
            return false;

        if (!m_pRs1->IsFinal (State))
            return false;

        DebugLogAssert (pOws == m_ows.begin ());
        State = m_pRs2->GetInitial ();

        for (i = 0; i < OwsCount; ++i) {

            const int Ow = pOws [i];
            State = m_pMealy2->GetDestOw (State, Ow, & (pOws [i]));
            if (-1 == State)
                return false;
        }
        if (!m_pRs2->IsFinal (State))
            return false;

    } else {

        int State = m_pRs1->GetInitial ();

        State = m_pMealy1->GetDestOw (State, FAFsmConst::IW_EOS, pOws++);
        DebugLogAssert (-1 != State);

        for (i = 0; i < Size; i += FAFsmConst::DIGITIZER_COUNT) {

            if (FAFsmConst::WRE_TT_TEXT & m_TokenType) {

                const int Iw = (const int) pChain [i];
                State = m_pMealy1->GetDestOw (State, Iw, pOws++);
                if (-1 == State)
                    return false;
            }
            if (FAFsmConst::WRE_TT_TAGS & m_TokenType) {

                const int Iw = (const int) pChain [i + 1];
                State = m_pMealy1->GetDestOw (State, Iw, pOws++);
                if (-1 == State)
                    return false;
            }
            if (FAFsmConst::WRE_TT_DCTS & m_TokenType) {

                const int Iw = (const int) pChain [i + 2];
                State = m_pMealy1->GetDestOw (State, Iw, pOws++);
                if (-1 == State)
                    return false;
            }
        }

        State = m_pMealy1->GetDestOw (State, FAFsmConst::IW_EOS, pOws++);
        if (-1 == State)
            return false;

        pOws -= OwsCount;

        DebugLogAssert (pOws == m_ows.begin ());
        DebugLogAssert (m_pRs1->IsFinal (State));

    } // of if (m_pRs2) ...

    Ows2TrBr (pOws, OwsCount, pRes, TupleSize);

    return true;
}

}

#endif
