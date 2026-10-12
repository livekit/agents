/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_TOKENSEGMENTATIONTOOLS_1BEST_T_H_
#define _FA_TOKENSEGMENTATIONTOOLS_1BEST_T_H_

#include "FAConfig.h"
#include "FARSDfaCA.h"
#include "FAMealyDfaCA.h"
#include "FAArrayCA.h"
#include "FAMultiMapCA.h"
#include "FADictConfKeeper.h"
#include "FALimits.h"
#include "FASecurity.h"
#include "FATokenSegmentationToolsCA_t.h"
#include <vector>
#include <float.h>

namespace BlingFire
{

///
/// Splits input sequence into segments with smaller number of segments 
///   and maximum sum of weights.
///
/// Input:  sequence of characters
/// Output: array of tuples <TokenId, From, To>
///
/// If input sequence contains a subsequence of unknown characters then
///   this subsequence is treated as one unknown segment.
///

template < class Ty >
class FATokenSegmentationTools_1best_t : public FATokenSegmentationToolsCA_t <Ty> {

public:
    FATokenSegmentationTools_1best_t ();

public:
    /// initializes from the valid configuration object
    void SetConf (const FADictConfKeeper * pConf);

    /// writes an array of tuples <TokenId, From, To> into pOut
    /// returns the actual / needed size of the array to fit all the tuples or
    ///  -1 in case of an error
    const int Process (
            const Ty * pIn,
            const int InSize,
            __out_ecount(MaxOutSize) int * pOut,
            const int MaxOutSize,
            const int UnkId
        ) const;

private:
    // Mealy DFA keeping a map from a known segment to idx and
    // and MultiMap keeping a realtion between idx and <ID, Score> pair
    const FARSDfaCA * m_pDfa;
    const FAMealyDfaCA * m_pMealy;
    const FAArrayCA * m_pK2I;     // note this is an identify since we don't have duplicate ID's
    const FAMultiMapCA * m_pI2Info;

    // unknown segment score
    const float m_UnkScore;

    // to keep track of arc data
    struct _TArc {

        int _Begin;    // the begging position of the ssegment
        int _Id;       // ID of a segment from the vocab
        double _Score; // cumulative score

    public:
        _TArc ():
            _Begin(-1),
            _Id(-1),
            _Score(-FLT_MAX)
        {}
    };

    // a helper method to add an arc into the pArcs
    inline void AddArc (_TArc * pArcs, int start, int end, int OwSum) const;

    // a helper method to add an arc if a token is not known
    inline void AddUnknownArc (_TArc * pArcs, int start) const;
};


template < class Ty >
FATokenSegmentationTools_1best_t < Ty >::
    FATokenSegmentationTools_1best_t () :
        m_pDfa (NULL),
        m_pMealy (NULL),
        m_pK2I (NULL),
        m_pI2Info (NULL),
        m_UnkScore (-100000.0) // this is guaranteed lower than any of the segment scores
{}


template < class Ty >
void FATokenSegmentationTools_1best_t < Ty >::
    SetConf (const FADictConfKeeper * pConf)
{
    LogAssert (pConf);
    LogAssert(FAFsmConst::TYPE_MEALY_DFA == pConf->GetFsmType());

    m_pDfa = pConf->GetRsDfa ();
    m_pMealy = pConf->GetMphMealy ();
    m_pK2I = pConf->GetK2I ();
    m_pI2Info = pConf->GetI2Info ();

    LogAssert(0 < m_pK2I->GetCount ());
}


template < class Ty >
inline void FATokenSegmentationTools_1best_t < Ty >::
    AddArc (_TArc * pArcs, int start, int end, int Key) const
{
    // look up the score and the id of the segment
    const int * pValues = NULL;
    const int Count = m_pI2Info->Get (Key, &pValues);
    LogAssert (2 == Count && NULL != pValues);

    const float Score = *((const float*) &(pValues [1]));

    // compute previous score given the start
    const double prevScore = 0 < start ? pArcs [start - 1]._Score : 0;

    // get a pointer to the arc object
    _TArc * pA =  pArcs + end;

    // set the arc, if it was never set then it has smallest negative float number
    // so the condition is always true
    if (pA->_Score < Score + prevScore) {

        pA->_Begin = start;
        pA->_Id = pValues [0];
        pA->_Score = Score + prevScore;
    }
}


template < class Ty >
inline void FATokenSegmentationTools_1best_t < Ty >::
    AddUnknownArc (_TArc * pArcs, int start) const
{
    const int end = start;

    // get a pointer to the arc object and previous arc object
    _TArc * pA =  pArcs + end;
    _TArc * pPrevA =  pA - 1;

    // compute previous score given the start
    const double prevScore = 0 < start ? pPrevA->_Score : 0;

    // set the arc, if it was never set then it has smallest negative float number
    // so the condition is always true
    if (pA->_Score < m_UnkScore + prevScore) {

        pA->_Begin = start;
        pA->_Id = -1;
        pA->_Score = m_UnkScore + prevScore;

        // check if the previous arc is also Unknown then merge them
        if (0 < start && -1 ==  pPrevA->_Id) {
            pA->_Begin =  pPrevA->_Begin;
        }
    }
}


template < class Ty >
const int FATokenSegmentationTools_1best_t < Ty >::
    Process (
        const Ty * pIn, 
        const int InSize, 
        __out_ecount(MaxOutSize) int * pOut,
        const int MaxOutSize,
        const int UnkId
    ) const
{
    DebugLogAssert (m_pDfa && m_pMealy && m_pK2I && m_pI2Info);

    if (0 >= InSize) {
        return 0;
    }

    LogAssert (pIn && InSize <= FALimits::MaxArrSize);

    // allocate storage for best arcs for each ending position
    std::vector <_TArc> End2BestArc (InSize);
    DebugLogAssert(InSize == End2BestArc.size ());
    _TArc * pArcs = End2BestArc.data ();

    // get the initial state
    const int InitialState = m_pDfa->GetInitial ();

    // populate the arcs
    for (int start = 0; start < InSize; ++start) {

        int State = InitialState;
        int SumOw = 0;
        int Ow = 0;
        bool TokenUnknown = true;

        // go as deep as we can from the start position
        for (int i = start; i < InSize; ++i) {

            const Ty Iw = pIn [i];
            State = m_pMealy->GetDestOw (State, Iw, &Ow);

            // see if the does not have a transition
            if (-1 == State) {
                break;
            }

            SumOw += Ow;
            DebugLogAssert (0 <= Ow);

            // see if the destination state is a final state
            if (m_pDfa->IsFinal (State)) {

                AddArc (pArcs, start, i, SumOw);
                TokenUnknown = false;
            }

        } // of for(int i = start; i < InSize; ++start) ...

        if (TokenUnknown) {
            AddUnknownArc (pArcs, start);
        }

    } // for(int start = 0; start < InSize; ++start) ...

    int ActualOutSize = 0;
    int end = InSize - 1;

    // now let's go in the reverse order and follow the best path
    while (0 <= end) {

        const _TArc * pA = pArcs + end;
        const int start = pA->_Begin;
        const int ID = pA->_Id;

        // validate the invariant of the algorithm
        DebugLogAssert (0 <= start && start < InSize);
        DebugLogAssert (0 <= end && end < InSize);
        DebugLogAssert (start <= end);
        DebugLogAssert (-1 == ID || ID >= 0);

        if (ActualOutSize + 3 <= MaxOutSize) {

            // write the results in <t, f, id> order so when reversed the order is correct
            pOut[ActualOutSize] = end;
            pOut[ActualOutSize + 1] = start;
            pOut[ActualOutSize + 2] = (-1 != ID) ? ID : UnkId;
        }
        ActualOutSize += 3;

        // skip to the end of the previous segment
        DebugLogAssert (start - 1 < end);
        end = start - 1;
    }

    // see if the pOut contains all the results
    if (MaxOutSize >= ActualOutSize) {
        // reverse the array so the results <id, f, t> and in left to tight order
        const int ActualOutSize_2 = ActualOutSize / 2;
        for(int i = 0; i < ActualOutSize_2; ++i) {
            int tmp = pOut[i];
            pOut[i] = pOut[ActualOutSize - i - 1];
            pOut[ActualOutSize - i - 1] = tmp;
        }
    }

    return ActualOutSize;
}

}

#endif
