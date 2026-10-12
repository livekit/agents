/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_DFA2MEALYNFA_H_
#define _FA_DFA2MEALYNFA_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FARSDfaA;
class FARSNfaA;
class FAMealyNfaA;

///
/// This class converts Rabin-Scott Dfa with <Iw, Ow> trantions into 
/// a Mealy NFA.
///

class FADfa2MealyNfa {

public:
    FADfa2MealyNfa (FAAllocatorA * pAlloc);

public:
    /// sets up Dfa on <Iw, Ow> labels
    void SetInDfa (const FARSDfaA * pDfa);
    /// sets up the output Mealy Nfa
    void SetOutNfa (FARSNfaA * pOutNfa, FAMealyNfaA * pOutOws);
    /// makes convertion
    void Process ();

private:
    const FARSDfaA * m_pDfa;
    FARSNfaA * m_pOutNfa;
    FAMealyNfaA * m_pOutOws;

    FAArray_cont_t < int > m_finals;
    FAArray_cont_t < int > m_stack;
    FAArray_cont_t < int > m_state2state;
};

}

#endif
