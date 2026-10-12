/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FAHmmTagger_l1.h"
#include "FAFsmConst.h"

namespace BlingFire
{

FAHmmTagger_l1::FAHmmTagger_l1 ():
    m_fInitialized (false),
    m_pW2TP (NULL),
    m_pPT (NULL),
    m_pPTT (NULL),
    m_MaxTags (0),
    m_EosTag (0),
    m_WordCount (0)
{
}


void FAHmmTagger_l1::Clear ()
{
    // initialize the arrays
    m_tags.Clear ();
    m_probs.Clear ();
    m_counts.Clear ();
    m_prev_best_idx.Clear ();

    // add the beginning of the sequence word (BOS)
    m_tags.push_back (m_EosTag);
    m_probs.push_back (0);
    m_counts.push_back (1);

    m_WordCount = 1;
}


void FAHmmTagger_l1::
    Initialize (
        const FAWordGuesser_prob_t < int > * pW2TP,
        const FAT2PTable * pPT,
        const FATs2PTable * pPTT,
        const int EosTag,
        const int MaxTagsPerWord,
        FAAllocatorA * pMemMgr)
{
    // invalid parameters
    LogAssert (pW2TP && pPT && pPTT && 0 < EosTag && 0 < MaxTagsPerWord && pMemMgr);

    m_pW2TP = pW2TP;
    m_pPT = pPT;
    m_pPTT = pPTT;
    m_MaxTags = MaxTagsPerWord;
    m_EosTag = EosTag;

    LogAssert (TAGS_PROBS_DELTA >= m_MaxTags);

    // initialize the arrays
    m_tags.SetAllocator (pMemMgr);
    m_tags.Clear ();
    m_probs.SetAllocator(pMemMgr);
    m_probs.Clear();
    m_counts.SetAllocator(pMemMgr);
    m_counts.Clear();
    m_prev_best_idx.SetAllocator(pMemMgr);
    m_prev_best_idx.Clear();

    // add the beginning of the sequence word (BOS)
    m_tags.push_back (EosTag);
    m_probs.push_back (0);
    m_counts.push_back (1);
    
    m_WordCount = 1;

    m_fInitialized = true;
}


void FAHmmTagger_l1::AddWord (const int * pWord, const int WordLen)
{
    LogAssert (m_fInitialized);

    DebugLogAssert (m_tags.size () == m_probs.size());
    DebugLogAssert (0 < m_WordCount);
    DebugLogAssert (m_WordCount <= m_tags.size());
    DebugLogAssert (m_WordCount <= m_counts.size());

    // invalid parameters
    LogAssert (NULL != pWord || 0 == WordLen);

    int * pAllCounts = m_counts.begin ();
    DebugLogAssert (pAllCounts);

    // get current amount of tags so far
    const int TotalTagCount = pAllCounts [m_WordCount - 1];

    // get output pointers
    int * pCount = pAllCounts + m_WordCount;
    int * pTags = m_tags.begin () + TotalTagCount;
    float * pProbs = m_probs.begin () + TotalTagCount;

    const int UnusedElements = m_tags.size () - TotalTagCount;

    // see how much space is left in the m_tags and m_probs
    if (m_MaxTags > UnusedElements)
    {
        // allocate TAGS_PROBS_DELTA more elements
        const int MtagsOldSize = m_tags.size ();
        m_tags.resize (MtagsOldSize + TAGS_PROBS_DELTA);
        pTags = m_tags.begin () + MtagsOldSize;
        const int MprobsOldSize = m_probs.size ();
        m_probs.resize(MprobsOldSize + TAGS_PROBS_DELTA);
        pProbs = m_probs.begin () + MprobsOldSize;
        LogAssert (pTags && pProbs);
        // adjust the pointers to point to the next available element
        pTags -= UnusedElements;
        pProbs -= UnusedElements;
    }
    // see how much space is left in the m_counts
    if (0 >= m_counts.size () - m_WordCount)
    {
        const int McountsOldSize = m_counts.size ();
        m_counts.resize (McountsOldSize + WORD_COUNT_DELTA);
        pCount = m_counts.begin () + McountsOldSize;
        LogAssert (pCount);
    }

    // Look up ln(P(T|W)) probabilities and tags. 
    // Store it at the pTags and pProbs locations.
    const int TagCount = m_pW2TP->Process (pWord, WordLen, pTags, pProbs, m_MaxTags);
    LogAssert (0 < TagCount && TagCount <= m_MaxTags);

    /// for each tag look up ln(P(T)) and substract it from ln(P(T|W))
    /// this is because we have to find the argmax using the P(W|T), not P(T|W)
    for (int i = 0; i < TagCount; ++i)
    {
        const int Tag = pTags [i];
        const float TagProb = m_pPT->GetProb (Tag);
        pProbs [i] -= TagProb;
    }

    // store the amount of tags/probs added so far
    *pCount = TagCount + TotalTagCount;

    // increment the word count
    m_WordCount++;
}


void FAHmmTagger_l1::AddEosWord ()
{
    DebugLogAssert (m_tags.size () == m_probs.size ());
    DebugLogAssert (0 < m_WordCount);
    DebugLogAssert (m_WordCount <= m_tags.size ());
    DebugLogAssert (m_WordCount <= m_counts.size ());

    int * pAllCounts = m_counts.begin ();
    DebugLogAssert (pAllCounts);

    // get current amount of tags so far
    const int TotalTagCount = pAllCounts [m_WordCount - 1];

    // get output pointers
    int * pCount = pAllCounts + m_WordCount;
    int * pTags = m_tags.begin () + TotalTagCount;
    float * pProbs = m_probs.begin () + TotalTagCount;

    // see how much space is left in the m_tags and m_probs
    if (0 >= m_tags.size () - TotalTagCount)
    {
        const int MtagsOldSize = m_tags.size ();
        m_tags.resize (MtagsOldSize + 1);
        pTags = m_tags.begin () + MtagsOldSize;
        const int MprobsOldSize = m_probs.size ();
        m_probs.resize (MprobsOldSize + 1);
        pProbs = m_probs.begin () + MprobsOldSize;
        LogAssert (pTags && pProbs);
    }
    // see how much space is left in the m_counts
    if (0 >= m_counts.size () - m_WordCount)
    {
        const int McountsOldSize = m_counts.size ();
        m_counts.resize (McountsOldSize + 1);
        pCount = m_counts.begin () + McountsOldSize;
        LogAssert (pCount);
    }

    // add EOS tag
    *pTags = m_EosTag;
    *pProbs = 0;
    *pCount = 1 + TotalTagCount;

    // increment the word count
    m_WordCount++;
}


const int FAHmmTagger_l1::Process (int * pOut, const int MaxOutSize)
{
    LogAssert (m_fInitialized);

    // see if the sentence is empty
    if (1 >= m_WordCount)
    {
        return 0;
    }

    // add the end of the sequence: EOS
    AddEosWord ();

    // the output sequence size
    const int OutSize = m_WordCount - 2;

    // the output buffer should be at least m_WordCount long
    LogAssert (MaxOutSize >= OutSize && pOut);

    // get tags of all words
    const int * pTags = m_tags.begin ();
    DebugLogAssert (pTags);

    // all word idx --> cumulative count mapping
    const int * pCounts = m_counts.begin ();
    DebugLogAssert (pCounts);

    // get probabilities of all words, this will become an array of costs
    // during the forward iteration step of the algorithm
    float * pProbs = m_probs.begin ();
    DebugLogAssert (pProbs);

    /// allocate array to keep back references
    const int TotalTagCount = pCounts [m_WordCount - 1];
    const int MprevOldSize = m_prev_best_idx.size ();
    m_prev_best_idx.resize (MprevOldSize + TotalTagCount);
    int * pBackRefIdxs = m_prev_best_idx.begin () + MprevOldSize;
    LogAssert (pBackRefIdxs);
    *pBackRefIdxs = -1;

    // keep track of the tags and probabilities for the previous word
    const int * pPrevTags = pTags;
    const float * pPrevProbs = pProbs;
    DebugLogAssert (1 == *pCounts);
    int PrevCount = 1;
    int PrevTagsSoFar = 1;
    int PrevPrevTagsSoFar = 0;

    ///
    /// forward iteration step
    ///
    for (int i = 1; i < m_WordCount; ++i)
    {
        const int TagsSoFar = pCounts [i];
        const int CurrCount = TagsSoFar - PrevTagsSoFar;

        const int * pCurrTags = pTags + PrevTagsSoFar;
        float * pCurrProbs = pProbs + PrevTagsSoFar;
        int * pCurrBackRefIdxs = pBackRefIdxs + PrevTagsSoFar;

        /// calculate best path for each tag for i-th word
        ///  from the pCurrTags

        // iterate thru all the tags of the current word
        for (int k = 0; k < CurrCount; ++k)
        {
            const int CurrTag = pCurrTags [k];

            // BestJ indicates where the best path is comming from
            // assume it goes from the tag j == 0 initially
            int BestJ = 0;
            // get the previous tag
            const int PrevTag = *pPrevTags;
            // get the transition probability
            const float TrProb = m_pPTT->GetProb (PrevTag, CurrTag);
            // get the previous path probability
            const float PrevTagPathProb = *pPrevProbs;
            // assign the path probability (from the j==0 tag)
            float BestPathProb = PrevTagPathProb + TrProb;

            // iterate thru the rest of the tags of the previous word
            for (int j = 1; j < PrevCount; ++j)
            {
                // get the previous path cost
                const float PreviousTagPathProb = pPrevProbs [j];

                // see if the j-th path is worse without adding the transition
                // probability, (transition probabilities are negative or 0)
                // Note: this condition is optional optimization, it does not 
                // affect the value of the best path, but it allows to not lookup
                // some marginal paths.
                if (BestPathProb < PreviousTagPathProb)
                {
                    // get the previous tag                
                    const int PreviousTag = pPrevTags [j];
                    // get the transition cost
                    const float TransProb = m_pPTT->GetProb (PreviousTag, CurrTag);

                    if (BestPathProb < PreviousTagPathProb + TransProb)
                    {
                        BestPathProb = PreviousTagPathProb + TransProb;
                        BestJ = j;
                    }
                }
            } // for (int j = 1; j < PrevCount; ++j) ... 

            // summ the best path cost for the tag k with the cost of being in k
            pCurrProbs [k] += BestPathProb;

            // for the current tag at k position, save the back reference to 
            // the best previos tag index, this index is an absolute index in pTags
            pCurrBackRefIdxs [k] = BestJ + PrevPrevTagsSoFar;

        } // for (int k = 0; k < CurrCount; ++k) ...

        pPrevTags = pCurrTags;
        pPrevProbs = pCurrProbs;
        PrevCount = CurrCount;
        PrevPrevTagsSoFar = PrevTagsSoFar;
        PrevTagsSoFar = TagsSoFar;

    } // of for (int i = 1; i <= m_WordCount; ++i) ...

    ///
    /// backward iteration step
    ///

    int BestTagIdx = pBackRefIdxs [TotalTagCount - 1];

    for (int j = OutSize - 1; j >= 0; j--)
    {
        const int BestTag = pTags [BestTagIdx];
        pOut [j] = BestTag; 
        BestTagIdx = pBackRefIdxs [BestTagIdx];
    }

    // reset the amount of words
    m_WordCount = 1;

    return OutSize;
}

}
