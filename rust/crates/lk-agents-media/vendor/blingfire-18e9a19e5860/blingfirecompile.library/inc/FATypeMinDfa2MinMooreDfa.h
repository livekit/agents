/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_TYPEMINDFA2MINMOOREDFA_H_
#define _FA_TYPEMINDFA2MINMOOREDFA_H_

#include "FAConfig.h"
#include "FAChain2Num_judy.h"
#include "FAArray_cont_t.h"
#include "FAMultiMap_judy.h"

namespace BlingFire
{

class FAAllocatorA;
class FARSDfaA;
class FAState2OwA;
class FAMultiMapA;


/// 
/// Constructs minimal moore dfa for digitizer.
/// Final states contain Ow, which indicates non-intersecting class over types.
///
/// Note: TypeNums are 1-based
///

class FATypeMinDfa2MinMooreDfa {

public:

    FATypeMinDfa2MinMooreDfa (FAAllocatorA * pAlloc);

public:
    /// sets up minimal type iw, 
    /// all the other type iws are greater or equal to this value
    void SetInitialTypeIw (const int InitialTypeIw);
    /// sets up the initial output weight
    void SetInitialOw (const int Ow);
    /// sets up input RS Dfa
    void SetInDfa (const FARSDfaA * pInDfa);
    /// sets up output Moore Dfa
    void SetOutMooreDfa (FARSDfaA * pOutDfa, FAState2OwA * pState2Ow);
    /// sets up output TypeNum -> { Ow } map
    void SetType2OwsMap (FAMultiMapA * pType2Ows);
    /// makes convertion
    void Process ();
    /// return object into the initial state (automatically called)
    void Clear ();

private:
    /// copies the state and 
    inline void ProcessState (const int State);
    /// maps chain into a new id, or returns an already associated
    inline const int BuildOw ();
    /// updates type to Ows map
    inline void UpdateType2Ows (const int Ow);

private:
    /// initial type iw
    int m_InitialTypeIw;
    /// input dfa
    const FARSDfaA * m_pInDfa;
    /// output Moore Dfa
    FARSDfaA * m_pOutDfa;
    FAState2OwA * m_pState2Ow;
    /// output Type -> Ows map
    FAMultiMapA * m_pType2Ows;
    /// maps all different type sets into id
    FAChain2Num_judy m_set2id;
    /// chain storage
    FAArray_cont_t < int > m_chain;
    /// current type id
    int m_MaxOw;
    int m_InitialOw;
};

}

#endif
