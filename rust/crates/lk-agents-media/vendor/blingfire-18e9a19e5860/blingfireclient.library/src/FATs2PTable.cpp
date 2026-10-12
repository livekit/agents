/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FATs2PTable.h"
#include "FATsConfKeeper.h"

namespace BlingFire
{

FATs2PTable::FATs2PTable () :
    m_pInt2Prob (NULL),
    m_Int2ProbArrSize (0),
    m_MaxTag (0),
    m_MaxTag2 (0),
    m_pArr (NULL),
    m_ArrSize (0),
    m_pFloatArr (NULL),
    m_FloatArrSize (0)
{}


FATs2PTable::~FATs2PTable ()
{
    if (NULL != m_pInt2Prob) {
        delete [] m_pInt2Prob;
        m_pInt2Prob = NULL;
    }
}


void FATs2PTable::SetConf (const FATsConfKeeper * pConf)
{
    m_MaxTag = 0;
    m_MaxTag2 = 0;
    m_pArr = NULL;
    m_ArrSize = 0;
    m_pFloatArr = NULL;
    m_FloatArrSize = 0;
    m_Int2ProbArrSize = 0;

    if (NULL != m_pInt2Prob) {
        delete [] m_pInt2Prob;
        m_pInt2Prob = NULL;
    }

    if (!pConf) {
        return;
    }

    const bool IsLog = pConf->GetIsLog ();
    const int MaxProb = pConf->GetMaxProb ();
    m_MaxTag = pConf->GetMaxTag ();
    m_MaxTag2 = m_MaxTag * m_MaxTag;

    /// m_pArr (compressed int-array) is not NULL, it keeps probability data in
    /// the quantized form, quantization parameters are in the float-array and
    /// currently these are just min and max values of stored probabilities
    /// if m_pArr is NULL then all probability values are stored directly as
    /// floats in the float-array

    m_pArr = pConf->GetArr ();
    m_FloatArrSize = pConf->GetArr (&m_pFloatArr);

    LogAssert (0 < m_MaxTag && 0 < MaxProb);
    LogAssert (!m_pFloatArr || 0 < m_FloatArrSize);

    // check if int-array keeps all data
    if (m_pArr) {

        m_ArrSize = m_pArr->GetCount ();
        LogAssert (0 < m_ArrSize);

        // then float-array (if any) keeps min/max float values
        LogAssert (0 == m_FloatArrSize || (2 == m_FloatArrSize && m_pFloatArr));

        float Min = float (FAFsmConst::MIN_LOG_PROB);
        float Max = float (FAFsmConst::MAX_LOG_PROB);

        if (!IsLog) {
            Min = 0.0f;
            Max = 1.0f;
        }

        if (2 == m_FloatArrSize) {
            Min = m_pFloatArr [0];
            Max = m_pFloatArr [1];
        }

        m_Int2ProbArrSize = MaxProb + 1;
        m_pInt2Prob = new float [m_Int2ProbArrSize];
        LogAssert (m_pInt2Prob);

        for (int i = 0; i <= MaxProb; ++i) {

            float LogProb = ((float(i) / float(MaxProb)) * (Max - Min)) + Min;

            if (!IsLog) {
                if (0.0f == LogProb) {
                    LogProb = float (FAFsmConst::MIN_LOG_PROB);
                } else {
                    LogProb = log (LogProb);
                }
            }
            m_pInt2Prob [i] = LogProb;
        }
    }
}

}
