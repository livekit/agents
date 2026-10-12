/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */



#ifndef _FA_AUTINTERPRETTOOLS_FNFA_T_H_
#define _FA_AUTINTERPRETTOOLS_FNFA_T_H_

#include "FAConfig.h"
#include "FARSDfaCA.h"
#include "FAState2OwCA.h"
#include "FAState2OwsCA.h"
#include "FAArray_cont_t.h"
#include "FAUtils_cl.h"
#include "FASetUtils.h"
#include "FASecurity.h"

namespace BlingFire
{

class FAAllocatorA;


///
/// Interpreter calculates final states of intersection with factor NFA of
/// the input text see www.dialog-21.ru/Archive/2004/Olonichev.pdf for details.
///

template < class Ty >
class FAAutInterpretTools_fnfa_t {

public:

    FAAutInterpretTools_fnfa_t (FAAllocatorA * pAlloc);

public:

    /// sets up AnyIw, 0 by default
    void SetAnyIw (const int AnyIw);
    /// sets RS Dfa interface
    void SetRsDfa (const FARSDfaCA * pDfa);
    /// sets State -> Ow map
    void SetState2Ow (const FAState2OwCA * pState2Ow);
    /// sets State -> Ows map
    void SetState2Ows (const FAState2OwsCA * pState2Ows);
    /// sets up tuple size, if Iws are tuples, is not used by default
    void SetTupleSize (const int TupleSize);

public:

    /// returns true if RS Dfa accepts the chain
    const bool Chain2Bool (const Ty * pChain,  const int Size);

    /// converts chain of input weights into the chain of sets of output weights
    void Chain2OwSetChain (const Ty * pChain,
                           __out_ecount(Size) const int ** pOwsPtrs,
                           __out_ecount(Size) int * pOwsSizes,
                           const int Size);

private:
    /// makes internal structures ready to process Size of input elements
    inline void Prepare (const int Size);
    /// builds resulting reaction 
    inline void BuildResMulti (
            const int Pos,                            // position in text
            const int * pStateSet,                    // states set
            const int SetSize,                        // states set size
            __out_ecount(Size) const int ** pOwsPtrs, // reaction container
            __out_ecount(Size) int * pOwsSizes,       // reaction container
            const int Size                            // MaxPos + 1
        );

private:

    const FARSDfaCA * m_pDfa;
    const FAState2OwCA * m_pState2Ow;
    const FAState2OwsCA * m_pState2Ows;
    int m_AnyIw;
    int m_TupleSize;

    FAArray_cont_t < int > m_set1;
    FAArray_cont_t < int > m_set2;

    FAArray_cont_t < const int * > m_res_sets;
    FAArray_cont_t < int > m_res_sizes;

    FASetUtils m_pos2res;
};


template < class Ty >
FAAutInterpretTools_fnfa_t< Ty >::
  FAAutInterpretTools_fnfa_t (FAAllocatorA * pAlloc) :
    m_pDfa (NULL),
    m_pState2Ow (NULL),
    m_pState2Ows (NULL),
    m_AnyIw (0),
    m_TupleSize (1),
    m_pos2res (pAlloc)
{
    m_set1.SetAllocator (pAlloc);
    m_set1.Create ();

    m_set2.SetAllocator (pAlloc);
    m_set2.Create ();

    m_res_sets.SetAllocator (pAlloc);
    m_res_sets.Create ();

    m_res_sizes.SetAllocator (pAlloc);
    m_res_sizes.Create ();
}


template < class Ty >
void FAAutInterpretTools_fnfa_t< Ty >::SetAnyIw (const int AnyIw)
{
    m_AnyIw = AnyIw;
}


template < class Ty >
void FAAutInterpretTools_fnfa_t< Ty >::SetRsDfa (const FARSDfaCA * pDfa)
{
    m_pDfa = pDfa;
}


template < class Ty >
void FAAutInterpretTools_fnfa_t< Ty >::
  SetState2Ow (const FAState2OwCA * pState2Ow)
{
    m_pState2Ow = pState2Ow;
}


template < class Ty >
void FAAutInterpretTools_fnfa_t< Ty >::
  SetState2Ows (const FAState2OwsCA * pState2Ows)
{
    m_pState2Ows = pState2Ows;
}


template < class Ty >
void FAAutInterpretTools_fnfa_t< Ty >::
    SetTupleSize (const int TupleSize)
{
    m_TupleSize = TupleSize;
}


template < class Ty >
void FAAutInterpretTools_fnfa_t< Ty >::Prepare (const int Size)
{
    m_pos2res.SetResCount (Size);

    /// sets won't be larger than Size (as rules-fsa is deterministic)
    m_set1.resize (Size);
    m_set2.resize (Size);

    // get the initial state
    int InitialState = m_pDfa->GetInitial ();

    // initial state will always be in both sets
    m_set1 [0] = InitialState;
    m_set2 [0] = InitialState;
}


template < class Ty >
void FAAutInterpretTools_fnfa_t< Ty >::
    BuildResMulti (
            const int Pos,
            const int * pStateSet,
            const int SetSize,
            /* __out_ecount(Size) */ const int ** pOwsPtrs,
            /* __out_ecount(Size) */ int * pOwsSizes,
            const int /* Size */
    )
{
    DebugLogAssert (m_pState2Ows);
    DebugLogAssert (pOwsPtrs && pOwsSizes);

    m_pos2res.Resize (0, Pos);

    for (int i = 0; i < SetSize; ++i) {

        DebugLogAssert (pStateSet);
        const int State = pStateSet [i];

        // see whether there is a reaction here
        const int OwsCount = m_pState2Ows->GetOws (State, NULL, 0);

        if (0 < OwsCount) {

            const int OldSize = m_pos2res.GetSize (Pos);
            m_pos2res.Resize (OldSize + OwsCount, Pos);

            int * pSet;
            m_pos2res.GetRes (&pSet, Pos);
            DebugLogAssert (pSet);

            // copy results
            m_pState2Ows->GetOws (State, pSet + OldSize, OwsCount);
        }
    }
    if (0 < m_pos2res.GetSize (Pos)) {

        m_pos2res.PrepareRes (Pos);

        const int * pOws;
        const int OwsSize = m_pos2res.GetRes (&pOws, Pos);

        pOwsSizes [Pos] = OwsSize;
        pOwsPtrs [Pos] = pOws;

    } else {

        pOwsSizes [Pos] = 0;
    }
}


template < class Ty >
const bool FAAutInterpretTools_fnfa_t< Ty >::
    Chain2Bool (const Ty * pChain, const int Size)
{
    DebugLogAssert (m_pDfa);

    int i, j;

    Prepare (Size);

    int * pSrcSet = m_set1.begin ();
    int * pDstSet = m_set2.begin ();

    int SrcSetSize = 1;
    int DstSetSize = 1;

    // make iteration thru the input chain
    for (i = 0; i < Size; ++i) {

        // convert input symbol into int
        DebugLogAssert (pChain);
        const int Iw = pChain [i];

        // we have to ignore initial state, if in the middle of a tuple
        int StartIdx = 0;
        if (0 != (i % m_TupleSize)) {
            StartIdx = 1;
        }

        // make iteration thru the SrcSet
        for (j = StartIdx; j < SrcSetSize; ++j) {

            const int SrcState = pSrcSet [j];

            // get the following state
            const int DstState = m_pDfa->GetDest (SrcState, Iw);

            DebugLogAssert (DstSetSize < Size);

            // add destination state to the destination set
            if (-1 != DstState && \
                pDstSet [DstSetSize - 1] != DstState) {

                pDstSet [DstSetSize] = DstState;
                DstSetSize++;
            }

        } // of for (int j = 0; ...

        // make it to be a valid set
        if (2 < DstSetSize) {
            DstSetSize = FASortUniq (pDstSet, pDstSet + DstSetSize);
        }

        // make iteration thru the pDstSet and collect all the reactions
	// j = 1, as we do not expect initial state to be final
	for (j = 1; j < DstSetSize; ++j) {

	  DebugLogAssert (pDstSet);
	  const int DstState = pDstSet [j];

	  if (m_pDfa->IsFinal (DstState))
	    return true;
	}

        // switch Dst and Src sets
        int * pTmpPtr = pDstSet;
        pDstSet = pSrcSet;
        pSrcSet = pTmpPtr;

        SrcSetSize = DstSetSize;
        DstSetSize = 1;

    } // of for (i = 0;

    return false;
}


template < class Ty >
void FAAutInterpretTools_fnfa_t< Ty >::
    Chain2OwSetChain (
            const Ty * pChain,
            __out_ecount(Size) const int ** pOwsPtrs,
            __out_ecount(Size) int * pOwsSizes,
            const int Size
        )
{
    DebugLogAssert (m_pDfa);
    DebugLogAssert (m_pState2Ows);

    int i, j;

    Prepare (Size);

    int * pSrcSet = m_set1.begin ();
    int * pDstSet = m_set2.begin ();

    int SrcSetSize = 1;
    int DstSetSize = 1;

    // make iteration thru the input chain
    for (i = 0; i < Size; ++i) {

        // convert input symbol into int
        DebugLogAssert (pChain);
        const int Iw = pChain [i];

        // we have to ignore initial state, if in the middle of a tuple
        int StartIdx = 0;
        if (0 != (i % m_TupleSize)) {
            StartIdx = 1;
        }

        // make iteration thru the SrcSet
        for (j = StartIdx; j < SrcSetSize; ++j) {

            const int SrcState = pSrcSet [j];

            // get the following state
            const int DstState = m_pDfa->GetDest (SrcState, Iw);

            DebugLogAssert (DstSetSize < Size);

            // add destination state to the destination set
            if (-1 != DstState && \
                pDstSet [DstSetSize - 1] != DstState) {

                pDstSet [DstSetSize] = DstState;
                DstSetSize++;
            }

        } // of for (int j = 0; ...

        // make it to be a valid set
        if (2 < DstSetSize) {
            DstSetSize = FASortUniq (pDstSet, pDstSet + DstSetSize);
        }
        if (0 == (i + 1) % m_TupleSize) {
            // make iteration thru the pDstSet and collect all the reactions
            // pDstSet + 1, as we do not expect any reaction in the initial state
            BuildResMulti (i / m_TupleSize, pDstSet + 1, DstSetSize - 1, 
                pOwsPtrs, pOwsSizes, Size);
        }

        // switch Dst and Src sets
        int * pTmpPtr = pDstSet;
        pDstSet = pSrcSet;
        pSrcSet = pTmpPtr;

        SrcSetSize = DstSetSize;
        DstSetSize = 1;

    } // of for (i = 0;
}

}

#endif
