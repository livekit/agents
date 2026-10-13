/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */



#ifndef _FA_AUTINTERPRETTOOLS_TRBR_T_H_
#define _FA_AUTINTERPRETTOOLS_TRBR_T_H_

#include "FAConfig.h"
#include "FAAutInterpretTools_pos_t.h"
#include "FAArray_cont_t.h"
#include "FAMultiMapCA.h"
#include "FABrResultA.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// This class idendifies correspondence between positions of _matched_ text
/// and triangular brackets from the regular expression.
///

template < class Ty >
class FAAutInterpretTools_trbr_t : public FAAutInterpretTools_pos_t < Ty > {

public:
    FAAutInterpretTools_trbr_t (FAAllocatorA * pAlloc);

public:
    /// sets up brackets beginning map
    void SetPos2BrBegin (const FAMultiMapCA * pPos2BrBegin);
    /// sets up brackets ending map
    void SetPos2BrEnd (const FAMultiMapCA * pPos2BrEnd);

public:
    /// if the input chain is accepted by the automaton then
    /// returns true and a fills in bracket results else returns false.
    const bool Chain2BrRes (const Ty * pInChain,
                            const int Size,
                            FABrResultA * pBrRes);
    /// the same as above but works with locally expanded Any symbols in DFA
    const bool Chain2BrRes_local (const Ty * pInChain,
                                  const int Size,
                                  FABrResultA * pBrRes);

private:
    /// returns true if BrId ends before Pos
    inline const bool Ends (const int BrId, const int Pos) const;
    /// extracts brackets
    inline void ExtractBrackets (
            const int * pPosChain, 
            const int Size, 
            FABrResultA * pBrRes
        );

private:

    const FAMultiMapCA * m_pPos2BrBegin;
    const FAMultiMapCA * m_pPos2BrEnd;

    FAArray_cont_t < int > m_pos_chain;
    FAArray_cont_t < int > m_br_pos_stack;
};


template < class Ty >
FAAutInterpretTools_trbr_t< Ty >::
    FAAutInterpretTools_trbr_t (FAAllocatorA * pAlloc) :
        FAAutInterpretTools_pos_t < Ty > (pAlloc),
        m_pPos2BrBegin (NULL),
        m_pPos2BrEnd (NULL)
{
    m_pos_chain.SetAllocator (pAlloc);
    m_pos_chain.Create ();

    m_br_pos_stack.SetAllocator (pAlloc);
    m_br_pos_stack.Create ();
}


template < class Ty >
void FAAutInterpretTools_trbr_t< Ty >::
    SetPos2BrBegin (const FAMultiMapCA * pPos2BrBegin)
{
    m_pPos2BrBegin = pPos2BrBegin;
}


template < class Ty >
void FAAutInterpretTools_trbr_t< Ty >::
    SetPos2BrEnd (const FAMultiMapCA * pPos2BrEnd)
{
    m_pPos2BrEnd = pPos2BrEnd;
}


template < class Ty >
const bool FAAutInterpretTools_trbr_t< Ty >::
    Ends (const int BrId, const int Pos) const
{
    DebugLogAssert (m_pPos2BrEnd);

    const int * pBrs;
    const int BrCount = m_pPos2BrEnd->Get (Pos, &pBrs);

    if (0 < BrCount)
        return (-1 != FAFind_log (pBrs, BrCount, BrId));
    else
        return false;
}


template < class Ty >
inline void FAAutInterpretTools_trbr_t< Ty >::
    ExtractBrackets (const int * pPosChain, const int Size, FABrResultA * pBrRes)
{
    DebugLogAssert (pBrRes);

    int i, TopPos;
    int TopBr = -1;

    int StackSize = 0;
    m_br_pos_stack.resize (0);

    for (i = 0; i < Size; ++i) {

        const int Pos = pPosChain [i];

        if (0 < StackSize) {

            // see whether we can close any opened brackets
            TopBr = m_br_pos_stack [StackSize - 1];

            while (Ends (TopBr, Pos)) {

                TopPos = m_br_pos_stack [StackSize - 2];
                pBrRes->AddRes (TopBr, TopPos, i - 1);

                m_br_pos_stack.pop_back ();
                m_br_pos_stack.pop_back ();

                StackSize -= 2;

                if (0 != StackSize) {

                    TopBr = m_br_pos_stack [StackSize - 1];

                } else {

                    TopBr = -1;
                    break;
                }

            } // of while (Ends (TopBr, Pos)) ...
        } // of if (0 < StackSize) ...

        const int * pBrBegin;
        const int BrBeginCount = m_pPos2BrBegin->Get (Pos, &pBrBegin);

        if (0 < BrBeginCount) {

            DebugLogAssert (pBrBegin);

            for (int j = 0; j < BrBeginCount; ++j) {

                const int Br = pBrBegin [j];

                if (Br != TopBr) {

                    m_br_pos_stack.push_back (i);
                    m_br_pos_stack.push_back (Br);

                    StackSize += 2;
                }

            } // of for (int j = 0; ...
        } // of if (0 < BrBeginCount) ...
    } // of for (int i = 0; ...

    while (0 < StackSize) {

        TopBr = m_br_pos_stack [StackSize - 1];
        TopPos = m_br_pos_stack [StackSize - 2];

        pBrRes->AddRes (TopBr, TopPos, i - 1);

        m_br_pos_stack.pop_back ();
        m_br_pos_stack.pop_back ();

        StackSize -= 2;
    }

    DebugLogAssert (0 == StackSize);
}


template < class Ty >
const bool FAAutInterpretTools_trbr_t< Ty >::
    Chain2BrRes (
        const Ty * pInChain,
        const int Size,
        FABrResultA * pBrRes
    )
{
    DebugLogAssert (0 < Size && pInChain);
    DebugLogAssert (pBrRes);

    m_pos_chain.resize (Size);
    int * pPosChain = m_pos_chain.begin ();

    if (true == this->Chain2PosChain (pInChain, pPosChain, Size)) {

        ExtractBrackets (pPosChain, Size, pBrRes);

        return true;

    } else {

        return false;
    }
}


template < class Ty >
const bool FAAutInterpretTools_trbr_t< Ty >::
    Chain2BrRes_local (
        const Ty * pInChain,
        const int Size,
        FABrResultA * pBrRes
    )
{
    DebugLogAssert (0 < Size && pInChain);
    DebugLogAssert (pBrRes);

    m_pos_chain.resize (Size);
    int * pPosChain = m_pos_chain.begin ();

    if (true == Chain2PosChain_local (pInChain, pPosChain, Size)) {

        ExtractBrackets (pPosChain, Size, pBrRes);

        return true;

    } else {

        return false;
    }
}

}

#endif
