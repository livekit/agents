/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_CALCIWEQCLASSES_H_
#define _FA_CALCIWEQCLASSES_H_

#include "FAConfig.h"
#include "FASplitSets.h"
#include "FAArray_cont_t.h"
#include "FAChain2Num_hash.h"
#include "FAEncoder_pref.h"

namespace BlingFire
{

class FAAllocatorA;
class FARSDfaA;
class FARSNfaA;
class FAMealyDfaA;
class FAMealyNfaA;
class FAMapA;

// 
// This class calculates equivalence classes for Iws for RS or Mealy Nfa or Dfa
//
// Input:
//
// initial: 0
// 0 1 1
// 0 1 2
// 0 1 30
// 1 2 2
// final: 1
// final: 2
//
// Output:
//
// 1  -> 0
// 30 -> 0
// 2  -> 1
//
// Note: 
//  1. Output weights are continuous, [NewIwBase, ... , GetMaxNewIw ()]
//  2. The following usage sequence is assumed:
//     1. Set*
//     2. Process
//     3. Get*
//

class FACalcIwEqClasses {

public:
    FACalcIwEqClasses (FAAllocatorA * pAlloc);

public:
    void SetIwBase (const int IwBase);
    void SetIwMax (const int IwMax);
    void SetNewIwBase (const int NewIwBase);
    void SetRsDfa (const FARSDfaA * pInDfa);
    void SetDfaSigma (const FAMealyDfaA * pDfaSigma);
    void SetRsNfa (const FARSNfaA * pInNfa);
    void SetNfaSigma (const FAMealyNfaA * pNfaSigma);
    void SetIw2NewIw (FAMapA * pIw2NewIw);
    void Process ();
    const int GetMaxNewIw () const;

private:
    // makes processor ready for the input 
    void Prepare ();
    inline const int GetDestId (const int State, const int Iw);
    inline const int GetOwsId (const int State, const int Iw);
    void Clear ();

private:
    // input (RS-NFA, RS-DFA, Mealy-NFA or Mealy-DFA)
    const FARSDfaA * m_pInDfa;
    const FAMealyDfaA * m_pDfaSigma;
    const FARSNfaA * m_pInNfa;
    const FAMealyNfaA * m_pNfaSigma;

    // input range
    int m_IwBase;
    int m_IwMax;
    int m_NewIwBase;
    // output map
    FAMapA * m_pIw2NewIw;
    // output MaxNewIw
    int m_MaxNewIw;
    // list of Iws to be classified
    FAArray_cont_t < int > m_e2iw;
    FAArray_cont_t < int > m_e2info;
    // classifying algorithm
    FASplitSets m_split_sets;
    // mapping from set to id (is used to map NFA Dst and Ow sets)
    FAChain2Num_hash m_set2id;
    FAEncoder_pref m_enc;
    /// NFA's input alphabet
    FAArray_cont_t < int > m_iws;
    /// temporary storage for NFA's output weights
    FAArray_cont_t < int > m_ows;

};

}

#endif
