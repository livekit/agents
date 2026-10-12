/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_TRBRNFA2MEALYNFA_H_
#define _FA_TRBRNFA2MEALYNFA_H_

#include "FAConfig.h"
#include "FAEpsilonRemoval.h"
#include "FAEpsilonGraph_mealy.h"
#include "FARSNfa_wo_ro.h"
#include "FAMealyNfa.h"
#include "FAArray_cont_t.h"
#include "FAMultiMap_judy.h"

namespace BlingFire
{

class FARSNfaA;
class FAMealyNfaA;

///
/// This class converts triangular bracket RS NFA into Mealy NFA, where
/// output weights represent combinations of openning and closing triangular
/// brackets.
///
/// Note: pInNfa and pOutNfa can point to the same object
///

class FATrBrNfa2MealyNfa {

public:
    FATrBrNfa2MealyNfa (FAAllocatorA * pAlloc);

public:
    /// sets up base value for triangular bracket transitions
    void SetTrBrBaseIw (const int TrBrBaseIw);
    /// sets up max value for triangular bracket transitions
    void SetTrBrMaxIw (const int TrBrMaxIw);
    /// specifies which symbol can be used as Epsilon
    void SetEpsilonIw (const int EpsilonIw);
    /// sets up Nfa with bracket transitions
    void SetInNfa (const FARSNfaA * pInNfa);
    /// sets up the output Mealy Nfa
    void SetOutNfa (FARSNfaA * pOutNfa, FAMealyNfaA * pOutOws);

    /// makes convertion
    void Process ();

    /// returns Ow -> [TrBr] map
    const FAMultiMapA * GetOw2TrBrMap () const;

private:
    // returns object into the initial state
    inline void Clear ();
    // builds epsilon Mealy NFA from RS NFA
    inline void BuildEpsilonMealy ();
    // removes epsilon transitions
    inline void RemoveEpsilons ();
    // makes Ows contiguous, based from 0
    inline void RemapOws ();

private:
    const FARSNfaA * m_pInNfa;
    FARSNfaA * m_pOutNfa;
    FAMealyNfaA * m_pOutOws;
    int m_TrBrBaseIw;
    int m_TrBrMaxIw;
    int m_EpsilonIw;

    /// epsilon removal algorithm
    FAEpsilonRemoval m_erm;
    /// epsilon graph making concatenation of Ows in output NFA
    FAEpsilonGraph_mealy m_g;
    /// temporary storage for Mealy E-NFA
    FARSNfa_wo_ro m_tmp_nfa;
    /// Mealy E-NFA sigma function
    FAMealyNfa m_tmp_ows1;
    /// Mealy NFA sigma function
    FAMealyNfa m_tmp_ows2;
    /// temporary array
    FAArray_cont_t < int > m_tmp;
    /// keeps mapping from Ow -> [Ows]
    FAMultiMap_judy m_ow2trbrs;
};

}

#endif
