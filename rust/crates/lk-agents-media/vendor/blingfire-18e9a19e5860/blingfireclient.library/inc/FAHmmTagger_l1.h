/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_HMMTAGGER_L1_H_
#define _FA_HMMTAGGER_L1_H_

#include "FAConfig.h"
#include "FAWordGuesser_prob_t.h"
#include "FAT2PTable.h"
#include "FATs2PTable.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

///
/// 1-best first order HMM POS tagger.
///
/// Expected usage:
///
/// 1. For i=0; i<WordCount do pTagger->AddWord (Wi); done
/// 2. pTagger->Process (pOut, WordCount);
/// [3. Repeat Steps 1-2 for other sequences without freeing the memory.]
/// [4. pTagger->Clear (); // free the memory]
/// [5. Repeat Steps 1-4.]
///

class FAHmmTagger_l1
{
public:
    FAHmmTagger_l1 ();

public:
    /// sets up the data containers, constants, and the memory manager
    void Initialize (
            const FAWordGuesser_prob_t < int > * pW2TP, // P(T|W)
            const FAT2PTable * pPT,                     // P(T)
            const FATs2PTable * pPTT,                   // P(T|T-1)
            const int EosTag,                           // EOS/BOS tag
            const int MaxTagsPerWord,        // maximum amount of tags per word
            FAAllocatorA * pMemMgr                    // memory manager
        );

    /// adds another word to the internal structures
    void AddWord (const int * pWord, const int WordLen);

    /// Makes decoding and returns a mostlikely sequence of tags.
    /// Resset input word count, so that results for the given sequence
    /// are returned only once. The pOut array should be at least WordCount
    /// long.
    const int Process (int * pOut, const int MaxOutSize);

    /// returns the object into the initial state and 
    //   frees memory from the internal arrays
    void Clear ();

private:
    /// adds a special EOS word at the end
    void AddEosWord ();

private:
    /// true if the processor has been initialized
    bool m_fInitialized;

    /// returns for the given word returns ln(P(T|W)) distribution
    const FAWordGuesser_prob_t < int > * m_pW2TP;

    /// for the given tag returns ln(P(T))
    const FAT2PTable * m_pPT;

    /// for the given tag, returns ln(P(T|T-1))
    const FATs2PTable * m_pPTT;

    /// maximum possible amount of tags per word
    int m_MaxTags;

    /// end of sequence, beginning of sequence tag
    int m_EosTag;

    /// all tags of all words
    FAArray_cont_t < int > m_tags;

    /// all lexical probabilities of all words, it is prallel to m_tags
    /// during the forward iteration of the viterbi algorithm the array
    /// contains the best probability for the path ending at this tag
    FAArray_cont_t < float > m_probs;

    /// for each the tag idx (from m_tags) keeps track of the best 
    /// previous tag index (from m_tags), this array is prallel to m_tags
    FAArray_cont_t < int > m_prev_best_idx;

    /// how many tags/probs, up to the i-th word there are in the m_tags
    FAArray_cont_t < int > m_counts;

    // actual word count + 2 for BOS and EOS
    int m_WordCount;

    enum {
        // if there is not enough space in the m_counts array,
        // then WORD_COUNT_DELTA more will be allocated
        WORD_COUNT_DELTA = 1024,

        // if there is not enough space in the m_tags/m_probs arrays,
        // then TAGS_PROBS_DELTA more will be allocated in the arrays
        TAGS_PROBS_DELTA = 1024,
    };
};

}

#endif
