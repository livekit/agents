/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_LAD_H_
#define _FA_LAD_H_

#include "FAConfig.h"
#include "FAWordGuesser_prob_t.h"
#include "FASummTagScores.h"

namespace BlingFire
{

class FALadLDB;

///
/// Word/N-gram based LAD
///

class FALad {

public:
    FALad ();
    ~FALad ();

public:
    /// initialization should be done once prior to any Process calls
    void Initialize (const FALadLDB * pLDB);
    /// returns the best language id or -1 in case of an error
    const int Process (const char * pText, size_t TextSize);
    /// returns two arrays one is a mapping Language --> Score and 
    /// the other (parallel) is a mapping Language --> Count
    /// Note: the Process () should be called before this function can be used
    const int GetScores (const float ** ppScores, const int ** ppCounts) const;
    /// returns object into initial state
    void Clear ();

private:
    /// returns the tag (other than m_UnkTag) with the best score and with
    /// the count no smaller than the MinCount
    /// returns -1 if such tag does not exist
    inline const int GetBestTag (
            const float * pScores,
            const int * pCounts,
            const int MinCount,
            const int * pLangs,
            const int LangCount
        ) const;
    /// this function breaks the text into tokens (close to words) and 
    /// feeds each of them to the ScoreNextToken for scoring
    void ScoreText (const char * pText, size_t TextSize);
    /// this functions scores the token and updates scores in different accumulators
    void ScoreNextToken ();
    /// returns true if the UTF-32 buffer has junk symbols
    inline const bool HasJunk (const int * pIn, const int Size) const;
    /// returns true if the UTF-32 buffer has upper case letters
    inline const bool HasUpper (const int * pIn, const int Size) const;
    /// resets all the scores
    inline void ResetScores ();
    /// finds best scores for each scorer
    inline void FindBestScores (const int * pLangs, const int LangCount);
    /// finds the best script
    const int GetBestScript () const;

private:

    /// compute scores by n-grams of the order Order
    const int ScoreNgrams (const char * pText, size_t TextSize, const int Order);
    /// compute scores by words
    const int ScoreWords (const char * pText, size_t TextSize);

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

    /// word --> log p(L|W) mapping
    FAWordGuesser_prob_t < int > m_WordScorer;
    /// n-gram --> log p(L|ngram) mapping
    FAWordGuesser_prob_t < int > m_NgrScorer;

    /// scorers agregators
    FASummTagScores m_WordsScores;
    int m_WordCount;
    int m_BestWordTag;
    /// scorers agregators
    FASummTagScores m_NgramScores [MaxNgramOrder];
    int m_NgramCount [MaxNgramOrder];
    int m_BestTag [MaxNgramOrder];

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

    /// the solution vector
    const int * m_pCounts;
    const float * m_pScores;

    /// indicates which scorers can be used
    bool m_fUseNgrams;
    bool m_fUseWords;
};

}

#endif
