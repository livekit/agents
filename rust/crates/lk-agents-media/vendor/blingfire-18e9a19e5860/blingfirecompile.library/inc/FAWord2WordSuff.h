/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_WORD2WORDSUFF_H_
#define _FA_WORD2WORDSUFF_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;
class FARSDfaCA;

///
/// From the give automaton of words (L), this class generates all pairs
/// < W1, S > s.t. W1 + S == W2, where W1 \in L and W2 \in L, and '+' is 
/// a concatenation operation.
///

class FAWord2WordSuff {

public:
    FAWord2WordSuff (FAAllocatorA * pAlloc);
    virtual ~FAWord2WordSuff ();

public:
    // keeps all the words
    void SetRsDfa (const FARSDfaCA  * pDfa);
    // sets up splitting separator symbol
    void SetDelim (const int Delim);
    // produce the output
    void Process ();

    // all splits will be output here
    virtual void PutSplit (
            const int * pLeft, 
            const int LeftSize,
            const int * pRight, 
            const int RightSize
        );

private:
    // recursively calculates splitting starting from State
    // (the depth of recursion is limited with the word length)
    void Split_rec (const int State);
    void PutSplits ();

protected:
    // keeps automaton alphabet
    const int * m_pIws;
    int m_IwsCount;
    // automaton
    const FARSDfaCA * m_pDfa;
    // keeps split separator symbol
    int m_Delim;

private:
    // the alphabet
    FAArray_cont_t < int > m_iws;
    // keeps current split, s.t. W1 == Concat (W2, Suff), where W1, W2 \in L
    FAArray_cont_t < int > m_split;
    FAArray_cont_t < int > m_delim_pos;
    FAArray_cont_t < int > m_left;
    FAArray_cont_t < int > m_right;
    // indicates whether splitting point is found
    bool m_PrefFound;
};

}

#endif
