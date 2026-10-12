/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_Ts2PTABLE_H_
#define _FA_Ts2PTABLE_H_

#include "FAConfig.h"
#include "FAFsmConst.h"
#include "FAArrayCA.h"

namespace BlingFire
{

class FAAllocatorA;
class FATsConfKeeper;

///
/// Keeps tag bigram or trigram tables: P(T2|T1), P(T3|T1,T2)
///

class FATs2PTable {

public:
    FATs2PTable ();
    ~FATs2PTable ();

public:
    // returns natural log of P(T2|T1)
    inline const float GetProb (const int T1, const int T2) const;
    // returns natural log of P(T3|T1,T2)
    inline const float GetProb (const int T1, const int T2, const int T3) const;
    // sets up the configuration
    void SetConf (const FATsConfKeeper * pConf);

private:
    /// dictionary to convert quantized interger probability into 
    /// a floating point number
    float * m_pInt2Prob;
    /// m_pInt2Prob array size
    int m_Int2ProbArrSize;

    /// maximum possible tag value
    int m_MaxTag;
    /// precalculated m_MaxTag * m_MaxTag
    int m_MaxTag2;

    /// compressed array of quantized integer probabilities
    const FAArrayCA * m_pArr;
    /// m_pArr array size
    int m_ArrSize;

    /// uncompressed array of probabilities
    const float * m_pFloatArr;
    /// m_pFloatArr array size
    int m_FloatArrSize;
};


inline const float FATs2PTable::GetProb (const int T1, const int T2) const
{
    DebugLogAssert (0 < T1 && m_MaxTag >= T1);
    DebugLogAssert (0 < T2 && m_MaxTag >= T2);

    const int Idx = (m_MaxTag * (T1 - 1)) + (T2 - 1);

    if (!m_pArr) {

        DebugLogAssert (m_pFloatArr && 0 < m_FloatArrSize);

        if (0 > m_FloatArrSize || Idx >= m_FloatArrSize) {
            return float (FAFsmConst::MIN_LOG_PROB);
        }

        return m_pFloatArr [Idx];

    } else {

        DebugLogAssert (m_pInt2Prob && m_pArr);

        if (0 > Idx || Idx >= m_ArrSize) {
            return float (FAFsmConst::MIN_LOG_PROB);
        }

        const int PrbIdx = m_pArr->GetAt (Idx);
        DebugLogAssert (0 <= PrbIdx && PrbIdx < m_Int2ProbArrSize);

        return m_pInt2Prob [PrbIdx];
    }
}

inline const float FATs2PTable::
    GetProb (const int T1, const int T2, const int T3) const
{
    DebugLogAssert (0 < T1 && m_MaxTag >= T1);
    DebugLogAssert (0 < T2 && m_MaxTag >= T2);
    DebugLogAssert (0 < T3 && m_MaxTag >= T3);
    DebugLogAssert (m_MaxTag2 == m_MaxTag * m_MaxTag);

    const int Idx = (m_MaxTag2 * (T1 - 1)) + (m_MaxTag * (T2 - 1)) + (T3 - 1);

    if (!m_pArr) {

        DebugLogAssert (m_pFloatArr && 0 < m_FloatArrSize);

        if (0 > m_FloatArrSize || Idx >= m_FloatArrSize) {
            return float (FAFsmConst::MIN_LOG_PROB);
        }

        return m_pFloatArr [Idx];

    } else {

        DebugLogAssert (m_pInt2Prob && m_pArr);

        if (0 > Idx || Idx >= m_ArrSize) {
            return float (FAFsmConst::MIN_LOG_PROB);
        }

        const int PrbIdx = m_pArr->GetAt (Idx);
        DebugLogAssert (0 <= PrbIdx && PrbIdx < m_Int2ProbArrSize);

        return m_pInt2Prob [PrbIdx];
    }
}

}

#endif
