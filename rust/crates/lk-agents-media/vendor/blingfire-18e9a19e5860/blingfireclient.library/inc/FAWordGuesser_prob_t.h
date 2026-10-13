/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_WORDGUESSER_PROB_T_H_
#define _FA_WORDGUESSER_PROB_T_H_

#include "FAConfig.h"
#include "FAWordGuesser_t.h"
#include "FAState2OwsCA.h"

namespace BlingFire
{

///
/// Word -> TAG/PROB guesser; for the given word returns an array of 
/// possible tags and an array of tag probabilities given the word, i.e. P(T|W)
///

template < class Ty >
class FAWordGuesser_prob_t : public FAWordGuesser_t < Ty > {

public:
    FAWordGuesser_prob_t ();
    ~FAWordGuesser_prob_t ();

public:
    /// Note: 1. this method expects initialized configuration object
    /// pInTr can be NULL if no transformation assumed
    /// 2. This method is expensive, it is supposed to be called once per 
    /// session
    void Initialize (
            const FAWgConfKeeper * pConf, 
            const FATransformCA_t < Ty > * pInTr
        );

    /// returns two parallel arrays and their size (natural log probs)
    /// returns 0 if no guess was made
    /// returns -1 if error
    const int Process (
            const Ty * pWordStr,   // input word
            const int WordLen,     // inpur word length
            const int ** ppTags,   // output array of tags
            const float ** ppProbs // output array of natural logs of P(T|W)
        );

    /// returns two parallel arrays and their size (natural log probs)
    /// returns 0 if no guess was made
    /// returns -1 if error
    /// Note:
    /// It always returns the actual necessary count of the full output. 
    /// It is caller responsibility to check if the output buffer was not
    /// large enough to accommodate all the output. If that is the case, 
    /// the content of the output is not defined, the caller should decide
    /// what to do: a) increase the buffer size and call the method again,
    /// b) ignore the output, or c) LogAssert.
    /// The maximum necessary size for the pTags/pProbs is:
    ///   pConf->GetState2Ows()->GetMaxOwsCount();
    const int Process (
            const Ty * pWordStr,   // input word
            const int WordLen,     // inpur word length
            int * pTags,           // output array of tags
            float * pProbs,        // output array of natural logs of P(T|W)
            const int MaxOutSize   // pTags/pProbs, maximum size
        ) const;

    /// returns two parallel arrays and their size (probs in the int scale)
    /// returns 0 if no guess was made
    /// returns -1 if error
    const int Process (
            const Ty * pWordStr, // input word
            const int WordLen,   // inpur word length
            const int ** ppTags, // output array of tags
            const int ** ppProbs // output array of P(T|W) in the int scale
        );

private:
    inline void Clear ();

private:
    int m_MaxCount;
    float * m_pProbs;

    int m_MaxIntProb;
    float * m_pI2P;

    float m_MaxProb;
};


template < class Ty >
FAWordGuesser_prob_t< Ty >::FAWordGuesser_prob_t () :
    m_MaxCount (0),
    m_pProbs (NULL),
    m_MaxIntProb (0),
    m_pI2P (NULL),
    m_MaxProb (FAFsmConst::MAX_LOG_PROB)
{}


template < class Ty >
FAWordGuesser_prob_t< Ty >::~FAWordGuesser_prob_t ()
{
    FAWordGuesser_prob_t< Ty >::Clear ();
}


template < class Ty >
inline void FAWordGuesser_prob_t< Ty >::Clear ()
{
    m_MaxCount = 0;

    if (m_pProbs) {
        delete [] m_pProbs;
        m_pProbs = NULL;
    }

    m_MaxIntProb = 0;

    if (m_pI2P) {
        delete [] m_pI2P;
        m_pI2P = NULL;
    }
}


template < class Ty >
void FAWordGuesser_prob_t< Ty >::Initialize (
        const FAWgConfKeeper * pConf, 
        const FATransformCA_t < Ty > * pInTr
    )
{
    /// setup the base guesser
    FAWordGuesser_t< Ty >::Initialize (pConf, pInTr);

    /// return this object into the initial state
    FAWordGuesser_prob_t< Ty >::Clear ();

    if (!pConf) {
        return;
    }

    m_MaxIntProb = pConf->GetMaxProb ();

    if (0 < m_MaxIntProb) {

        // allocate the dictionary
        LogAssert (NULL == m_pI2P);
        m_pI2P = new float [m_MaxIntProb + 1];
        LogAssert (m_pI2P);

        // get minimum and the maximum of the stored values
        const float Min = pConf->GetMinProbVal ();
        const float Max = pConf->GetMaxProbVal ();
        // fLogScale is used only if Maximum and minimum are specified
        const bool fLogScale = pConf->GetIsLog ();

        // see if the min/max information is stored
        if (Max > Min) {
            // fill in the dictionary
            for (int i = 0; i <= m_MaxIntProb; ++i) {
                // remap the probability value
                float Value = 
                    ((float(i) / float(m_MaxIntProb)) * (Max - Min)) + Min;
                // see if we need to calculate the logarithm
                if (!fLogScale) {
                    if (0 < Value) {
                        Value = log (Value);
                    } else {
                        Value = FAFsmConst::MIN_LOG_PROB;
                    }
                }
                m_pI2P [i] = Value;
            }
        // assume (0..1] if not
        } else {
            // fill in the dictionary
            m_pI2P [0] = FAFsmConst::MIN_LOG_PROB;
            for (int i = 1; i <= m_MaxIntProb; ++i) {
                const float LogProb = log (float (i) / float (m_MaxIntProb));
                m_pI2P [i] = LogProb;
            }
        } // of if if (Max > Min) ...

    } // if (0 < m_MaxIntProb) ...

    const FAState2OwsCA * pOws = pConf->GetState2Ows ();

    if (pOws) {
        m_MaxCount = pOws->GetMaxOwsCount () / 2;
    }

    if (0 < m_MaxCount) {
        LogAssert (NULL == m_pProbs);
        m_pProbs = new float [m_MaxCount];
        LogAssert (m_pProbs);
    }
}


template < class Ty >
const int FAWordGuesser_prob_t< Ty >::
    Process (
            const Ty * pWordStr, 
            const int WordLen, 
            const int ** ppTags, 
            const float ** ppProbs
        )
{
    DebugLogAssert (0 == WordLen || NULL != pWordStr);
    DebugLogAssert (ppTags && ppProbs);

    const int OwCount = FAWordGuesser_t< Ty >::Process (pWordStr, WordLen, ppTags);

    // at least one pair, normally that's the case
    if (2 <= OwCount) {

        DebugLogAssert (*ppTags);
        DebugLogAssert (0 == OwCount % 2);

        const int Count = OwCount >> 1;
        DebugLogAssert (Count <= m_MaxCount && m_pProbs);

        const int * pProbOws = (*ppTags) + Count;

        for (int i = 0; i < Count; ++i) {

            const int IntProb = pProbOws [i];
            DebugLogAssert (0 <= IntProb && IntProb <= m_MaxIntProb);

            m_pProbs [i] = m_pI2P [IntProb];

        } // of for (int i = 0; ...

        *ppProbs = m_pProbs;
        return Count;

    /// only the default tag is returned
    } else if (1 == OwCount) {

        *ppProbs = & m_MaxProb;
        return 1;

    /// nothing
    } else {

        return OwCount;
    }
}


template < class Ty >
const int FAWordGuesser_prob_t< Ty >::
    Process (
            const Ty * pWordStr,
            const int WordLen,
            int * pOutTags,
            float * pOutProbs,
            const int MaxOutSize
        ) const
{
    DebugLogAssert (0 == WordLen || NULL != pWordStr);

    // use pOutTags to store both tag and encoded probabilities
    const int OwCount = FAWordGuesser_t< Ty >::Process (pWordStr, WordLen, pOutTags, MaxOutSize);

    // at least one pair, usually that's the case
    if (2 <= OwCount) {

        DebugLogAssert (pOutTags);
        DebugLogAssert (0 == OwCount % 2);

        const int Count = OwCount >> 1;
        DebugLogAssert (Count <= m_MaxCount);

        // see if only the count should be returned without copying the actual values
        if (MaxOutSize >= OwCount) {

            // get a pointer to the encoded IntProbs
            const int * pProbOws = pOutTags + Count;

            for (int i = 0; i < Count; ++i) {

                const int IntProb = pProbOws [i];
                DebugLogAssert (0 <= IntProb && IntProb <= m_MaxIntProb);

                pOutProbs [i] = m_pI2P [IntProb];

            } // of for (int i = 0; ...
        }

        return Count;

    /// only the default tag is returned
    } else if (1 == OwCount) {

        *pOutProbs = m_MaxProb;
        return 1;

    /// nothing
    } else {

        return OwCount;
    }
}


template < class Ty >
const int FAWordGuesser_prob_t< Ty >::
    Process (
            const Ty * pWordStr,
            const int WordLen,
            const int ** ppTags,
            const int ** ppProbs
        )
{
    DebugLogAssert (0 == WordLen || NULL != pWordStr);
    DebugLogAssert (ppTags && ppProbs);

    const int OwCount = FAWordGuesser_t< Ty >::Process (pWordStr, WordLen, ppTags);

    // at least one pair, normally that's the case
    if (2 <= OwCount) {

        DebugLogAssert (*ppTags);
        DebugLogAssert (0 == OwCount % 2);

        const int Count = OwCount >> 1;
        DebugLogAssert (Count <= m_MaxCount && m_pProbs);

        *ppProbs = (*ppTags) + Count;
        return Count;

    } else if (1 == OwCount) {

        *ppProbs = & m_MaxIntProb;
        return 1;

    } else {

        return OwCount;
    }
}

}

#endif
