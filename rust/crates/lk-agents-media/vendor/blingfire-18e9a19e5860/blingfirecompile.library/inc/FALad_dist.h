/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef BLING_FIRE_NOAP
#ifndef _FA_LAD_DIST_H_
#define _FA_LAD_DIST_H_

#include "FAConfig.h"
#include "FAWordGuesser_prob_t.h"
#include "FASummTagScores.h"
#include "MemLfpStringMap.h"

namespace BlingFire
{

class FALadLDB;

///
/// Word/N-gram based LAD
///

class FALad_dist {

public:
    FALad_dist ();
    ~FALad_dist ();

public:
    /// initialization should be done once prior to any Process calls
    void Initialize (CMemLfpManager * pMemMgr, const FALadLDB * pLDB);
    /// returns the best language id or -1 in case of an error
    const int Process (const char * pText, size_t TextSize);
    /// returns two arrays one is a mapping Language --> Score and 
    /// the other (parallel) is a mapping Language --> Count
    /// Note: the Process () should be called before this function can be used
    const int GetScores (const float ** ppScores, const int ** ppCounts) const;
    /// returns object into initial state
    void Clear ();

private:
    /// this function breaks the text into tokens (close to words) and 
    /// feeds each of them to the ScoreNextToken for scoring
    void ScoreText (const char * pText, size_t TextSize);
    /// this functions scores the token and updates scores in different accumulators
    void ScoreNextToken ();
    /// returns true if the UTF-32 buffer has junk symbols
    inline const bool HasJunk (const int * pIn, const int Size) const;
    /// returns true if the UTF-32 buffer has upper case letters
    inline const bool HasUpper (const int * pIn, const int Size) const;
    /// resets scores and counts
    inline void Reset ();
    /// finds the best tag given the statistics map and the scorer
    /// returns -1 if no best tag was found
    const int GetBestScore (
            CMemLfpStringMap * pMap, 
            FAWordGuesser_prob_t < int > * pScorer, 
            const int TotalCount, 
            const int MinCount,
            const int * pLangs,
            const int LangCount
        );
    /// returns the most represented script on the page, or -1 if none
    const int GetBestScript () const ;

private:
    enum {
        MaxNgramOrder = 4,
        MaxTokenLen = 1024,
        DefJunkSymbol = 0x21,
    };

    /// maximum tag value
    int m_MaxTag;
    /// Unknown language tag value
    int m_UnkTag;
    // n-gram order
    int m_Order;
    // n-gram min backoff order
    int m_MinOrder;
    // maximum amount of n-grams to use
    int m_MaxCount;
    /// percent of n-grams should match
    int m_MinMatchRatio;
    /// percent of words should match
    int m_MinWordMatchRatio;

    /// temporary token buffer
    int m_Token [MaxTokenLen + 2];
    /// token length
    int m_TokenLen;

    /// maps found ngrams to counts
    CMemLfpManager * m_pMemMgr;
    CMemLfpStringMap m_Str2Count [MaxNgramOrder];
    CMemLfpStringMap m_Word2Count;

    // total counts
    int m_NgramCount [MaxNgramOrder];
    int m_WordCount;

    /// n-gram --> log p(L|ngram) mapping
    FAWordGuesser_prob_t < int > m_NgrScorer;
    /// word --> log p(L|ngram) mapping
    FAWordGuesser_prob_t < int > m_WordScorer;
    /// L --> Score container
    float * m_pScores;
    /// L --> Count container
    int * m_pCounts;

    /// charmap
    const FAMultiMapCA * m_pCharMap;

    /// char --> script map
    const FAMultiMapCA * m_pC2SMap;
    /// script --> lang map
    const FAMultiMapCA * m_pS2LMap;
    /// script tag boundaries
    int m_MinScriptTag;
    int m_MaxScriptTag;
    /// script --> count map
    int * m_pScript2Count;

    /// indicates which scorers can be used
    bool m_fUseNgrams;
    bool m_fUseWords;
};

}

#endif
#endif
