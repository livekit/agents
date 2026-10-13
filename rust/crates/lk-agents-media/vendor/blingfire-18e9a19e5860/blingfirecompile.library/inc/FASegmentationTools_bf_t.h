/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_SEGMENTATIONTOOLS_BF_T_H_
#define _FA_SEGMENTATIONTOOLS_BF_T_H_

#include "FAConfig.h"
#include "FARSDfaCA.h"
#include "FAState2OwCA.h"
#include "FAUtf32Utils.h"
#include "FALimits.h"
#include "FASecurity.h"

namespace BlingFire
{

///
/// Splits input sequence into known segments of the maximum length.
///

template < class Ty >
class FASegmentationTools_bf_t {

public:
    FASegmentationTools_bf_t ();

public:
    /// initializes from the valid configuration object
    void SetConf (const FAW2SConfKeeper * pConf);

    /// returns segmentation positions, the ends of subsequent segments
    /// returns -1, if no segment chain were found
    const int Process (
            const Ty * pIn,
            const int InSize,
            __out_ecount(MaxOutSize) int * pOut,
            const int MaxOutSize
        );

private:
    inline void Prepare ();
    inline void AddArcs (const int End);
    inline void AddArc (const int Beg, const int End, const int ArcWeight);
    inline const int GetBestSplit (
            __out_ecount(MaxOutSize) int * pOut,
            const int MaxOutSize
        ) const;

private:
    // algorithm defaults and limits
    enum { 
        DefMinInputLen = 8,
    };

private:
    // input chain
    const Ty * m_pIn;
    int m_InSize;
    // DFA keeping all known segments, stored from right to left
    const FARSDfaCA * m_pDfa;
    const FAState2OwCA * m_pState2Ow;
    // keeps minimum input length
    int m_MinInputLen;
    // keeps minimum valid segment length
    int m_MinSegLen;
    // ignore-case flag
    bool m_ignore_case;
    // direction
    int m_dir;
    // queue of segmentation starts (m_Q size is limited with 2 * MaxWordLen)
    int m_Q [2 * FALimits::MaxWordLen];
    // actual queue size
    int m_QueueSize;
    // maps Begin into the best End, with respect to the entire path
    int m_Begin2BestEnd [FALimits::MaxWordLen];
    // maps Begin into Weight of the best splitting path
    unsigned int m_Begin2BestWeight [FALimits::MaxWordLen];
    // maps Begin into Depth of the best splitting path
    int m_Begin2BestDepth [FALimits::MaxWordLen];
    // a set (not sorted) of considered Begin positions
    int m_Begins [FALimits::MaxWordLen];
    // number of elements in the array m_Begins (not greater MaxWordLen)
    int m_BeginCount;
    // indicates whether splitting was found
    bool m_SplitFound;
};


template < class Ty >
FASegmentationTools_bf_t < Ty >::
    FASegmentationTools_bf_t () :
        m_pIn (NULL),
        m_InSize (0),
        m_pDfa (NULL),
        m_pState2Ow (NULL),
        m_MinInputLen (DefMinInputLen),
        m_MinSegLen (-1),
        m_ignore_case (false),
        m_dir (FAFsmConst::DIR_R2L),
        m_QueueSize (0),
        m_BeginCount (0),
        m_SplitFound (false)
{
    for (int i = 0; i < FALimits::MaxWordLen; ++i) {
        m_Begin2BestEnd [i] = -1;
        m_Begin2BestWeight [i] = 1;
        m_Begin2BestDepth [i] = 0;
    }
}


template < class Ty >
void FASegmentationTools_bf_t < Ty >::
    SetConf (const FAW2SConfKeeper * pConf)
{
    LogAssert (pConf);

    m_pDfa = pConf->GetRsDfa ();
    m_pState2Ow = pConf->GetState2Ow ();
    m_MinInputLen = pConf->GetMinInputLen ();
    m_MinSegLen = pConf->GetMinSegLen ();
    m_ignore_case = pConf->GetIgnoreCase ();
    m_dir = pConf->GetDirection ();

    if (0 >= m_MinInputLen || FALimits::MaxWordLen < m_MinInputLen) {
        m_MinInputLen = DefMinInputLen;
    }
    if (0 >= m_MinSegLen || FALimits::MaxWordLen < m_MinSegLen) {
        m_MinSegLen = -1;
    }
}


template < class Ty >
inline void FASegmentationTools_bf_t < Ty >::Prepare ()
{
    m_SplitFound = false;
    m_QueueSize = 0;

    /// return used values back into the initial state

    DebugLogAssert (0 <= m_BeginCount && m_BeginCount <= FALimits::MaxWordLen);

    for (int i = 0; i < m_BeginCount; ++i) {

        const int Pos = m_Begins [i];
        DebugLogAssert (FALimits::MaxWordLen >= Pos && 0 <= Pos);

        m_Begin2BestEnd [Pos] = -1;
        m_Begin2BestWeight [Pos] = 1;
        m_Begin2BestDepth [Pos] = 0;
    }

    m_BeginCount = 0;
}


template < class Ty >
inline void FASegmentationTools_bf_t < Ty >::
    AddArc (const int Beg, const int End, const int ArcWeight)
{
    DebugLogAssert (0 <= Beg && Beg < m_InSize);
    DebugLogAssert (0 <= End && End < m_InSize);
    DebugLogAssert (End + 1 < FALimits::MaxWordLen - 1);

    const int CurrWeight = (1 + ArcWeight) * (m_Begin2BestWeight [End + 1]);
    const int CurrDepth = 1 + m_Begin2BestDepth [End + 1];

    // see if this Beg position has been used already
    if (-1 == m_Begin2BestEnd [Beg]) {

        m_Begin2BestEnd [Beg] = End;
        m_Begin2BestWeight [Beg] = CurrWeight;
        m_Begin2BestDepth [Beg] = CurrDepth;

        m_Begins [m_BeginCount++] = Beg;
        DebugLogAssert (0 < m_BeginCount && m_BeginCount <= FALimits::MaxWordLen);

        m_Q [m_QueueSize++] = Beg;
        DebugLogAssert (0 < m_QueueSize && m_QueueSize <= 2 * FALimits::MaxWordLen);

    } else {

        const int BestWeight = m_Begin2BestWeight [Beg];
        const int BestDepth = m_Begin2BestDepth [Beg];

        if ((CurrDepth < BestDepth) || \
            (CurrDepth == BestDepth && CurrWeight > BestWeight)) {

            m_Begin2BestEnd [Beg] = End;
            m_Begin2BestWeight [Beg] = CurrWeight;
            m_Begin2BestDepth [Beg] = CurrDepth;
        }
    } // of if (-1 == m_Begin2BestEnd [Beg]) ...
}


template < class Ty >
inline void FASegmentationTools_bf_t < Ty >::
    AddArcs (const int End)
{
    DebugLogAssert (m_pDfa && m_pState2Ow);
    DebugLogAssert (m_pIn && End < m_InSize);

    int State = m_pDfa->GetInitial ();

    for (int i = End; i >= 0; --i) {

        const Ty Iw = m_pIn [i];
        State = m_pDfa->GetDest (State, Iw);

        if (-1 == State) {

            break;

        } else if (m_pDfa->IsFinal (State)) {

            const int Ow = m_pState2Ow->GetOw (State);
            DebugLogAssert (0 <= Ow);
            AddArc (i, End, Ow);

            if (0 == i) {
                m_SplitFound = true;
            }
        }
    } // of for (int i = End; ...
}


template < class Ty >
inline const int FASegmentationTools_bf_t < Ty >::
    GetBestSplit (
        __out_ecount(MaxOutSize) int * pOut,
        const int MaxOutSize
    ) const
{
    if (MaxOutSize < m_InSize) {
        return m_InSize;
    }
    DebugLogAssert (pOut);

    int Count = 0;
    int Pos = m_Begin2BestEnd [0];
    DebugLogAssert (-1 != Pos);

    // build an array of segment boundaries
    while (0 <= Pos && Pos < m_InSize && Count < MaxOutSize) {
        pOut [Count++] = Pos;
        Pos = m_Begin2BestEnd [Pos + 1];
    }

    // reverse the positions, if needed
    //
    // Relation between segment boundaries and reverse segment boundaries:
    //
    // 012345678  (positions)
    // xxxyyyyzz  (compound of three segments)
    //   *   * *  (segment boundaries)
    // zzyyyyxxx  (reversed compound of three segments)
    //  *   *  *  (reversed segment boundaries)
    // 
    if (FAFsmConst::DIR_L2R == m_dir && 1 < Count) {
        int i;
        const int Count_1 = Count - 1;
        for (i = 0; i < Count_1; ++i) {
            pOut [i] = m_InSize - pOut [i] - 2;
        }
        for (i = 0; i < Count_1 / 2; ++i) {
            const int T = pOut [i];
            pOut [i] = pOut [Count_1 - i - 1];
            pOut [Count_1 - i - 1] = T;
        }
    }

    // attach all segments smaller than m_MinSegLen to the left, if needed
    if (0 < m_MinSegLen) {

        int BeginPos = 0;
        int OutCount = 0;
        
        for (int i = 0; i < Count; ++i) {

            const int EndPos = pOut [i];
            const int SegLen = EndPos - BeginPos + 1;

            if (SegLen < m_MinSegLen && 0 < OutCount) {
                pOut [OutCount - 1] = EndPos;
            } else {
                pOut [OutCount++] = EndPos;
            }

            BeginPos = EndPos + 1;
        }

        return OutCount;

    } else {

        return Count;
    }
}


template < class Ty >
const int FASegmentationTools_bf_t < Ty >::
    Process (
        const Ty * pIn, 
        const int InSize, 
        __out_ecount(MaxOutSize) int * pOut,
        const int MaxOutSize
    )
{
    Ty TmpBuff [FALimits::MaxWordLen];
    DebugLogAssert (m_pDfa && m_pState2Ow);

    if (FALimits::MaxWordLen < InSize || m_MinInputLen > InSize) {
        return -1;
    }

    m_InSize = InSize;

    // ignore-case, if needed
    if (false == m_ignore_case) {
        m_pIn = pIn;
    } else {
        m_pIn = TmpBuff;
        for (int i = 0 ; i < InSize; ++i) {
            const int InSymbol = (int) pIn [i] ;
            const int OutSymbol = FAUtf32ToLower (InSymbol) ;
            TmpBuff [i] = (Ty) OutSymbol ;
        }
    }
    // reverse the string, if needed
    if (FAFsmConst::DIR_L2R == m_dir) {
        for (int i = 0 ; i < InSize; ++i) {
            const Ty C = m_pIn [i];
            TmpBuff [InSize - i - 1] = C;
        }
        m_pIn = TmpBuff;
    }

    Prepare ();

    int QP = 0;
    int Depth = 1;

    /// enqueue a depth marker
    m_Q [m_QueueSize++] = -1;
    DebugLogAssert (0 < m_QueueSize && m_QueueSize <= 2 * FALimits::MaxWordLen);

    /// and initial arcs
    AddArcs (InSize - 1);

    while (QP != m_QueueSize) {

        const int b = m_Q [QP++];
        DebugLogAssert ((-1 == b) || (FALimits::MaxWordLen >= b && 0 <= b));

        if (-1 == b) {

            if (m_SplitFound) {

                DebugLogAssert (QP < InSize + 1);
                const int Count = GetBestSplit (pOut, MaxOutSize);
                return Count;

            } else if (QP == m_QueueSize) {

                return -1;

            } else {

                /// enqueue a depth marker
                m_Q [m_QueueSize++] = -1;
                DebugLogAssert (0 < m_QueueSize && m_QueueSize <= 2 * FALimits::MaxWordLen);

                Depth++;
            }
        } else if (0 != b) {

            AddArcs (b - 1);
        }
    } // of while (0 != Size) ...

    DebugLogAssert (QP < InSize + 1);
    return -1;
}

}

#endif
