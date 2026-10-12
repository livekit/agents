/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_TYPESNFALIST2TYPENFA_H_
#define _FA_TYPESNFALIST2TYPENFA_H_

#include "FAConfig.h"
#include "FARSNfa_ar_judy.h"
#include "FANfas2CommonENfa.h"
#include "FAEpsilonRemoval.h"
#include "FAChain2Num_judy.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FARSDfaA;
class FAAllocatorA;

///
/// Reads a list of RS automata from stream preceeding by the type information.
/// Converts them into a single epsilon-free Nfa with type sets associated with
/// each final state.
///
/// Note:
///
///   The algorithm makes additional treatment for AnyIw only if 
///   it has been explicitly set up.
///

class FATypesNfaList2TypeNfa {

public:

    FATypesNfaList2TypeNfa (FAAllocatorA * pAlloc);

public:
    /// sets up epsilon input weight
    void SetEpsilonIw (const int EpsilonIw);
    /// sets up spacial AnyOther input weight
    void SetAnyIw (const int AnyIw);
    /// sets up the initial type iw
    void SetInitialTypeIw (const int InitialTypeIw);
    /// adds < TypeName, Nfa > pair
    void AddTyNfa (const char * pTypeStr, const FARSNfaA * pNfa);
    /// adds < TypeName, Dfa > pair
    void AddTyDfa (const char * pTypeStr, const FARSDfaA * pDfa);
    /// builds common Nfa
    void Process ();
    /// returns digitizer Nfa interface
    const FARSNfaA * GetNfa () const;
    /// returns object into initial state
    void Clear ();

private:
    /// adds type info into the current dictionary nfa
    void AddTypeInfo (const int TypeIw);
    /// returns type associated with the string, or creates a new one
    const int GetType (const char * pTypeStr);
    /// frees all intermediate containers
    inline void Clear_a ();

private:
    /// current type Iw
    int m_CurrTypeIw;
    /// initial type Iw
    int m_InitialTypeIw;
    /// dictionary nfa
    FARSNfa_ar_judy m_nfa;
    /// common nfa
    FANfas2CommonENfa m_common;
    /// epsilon removal algorithm
    FAEpsilonRemoval m_eremoval;
    /// mapping from string to type id
    FAChain2Num_judy m_type2id;
    /// temporary states/chain container
    FAArray_cont_t < int > m_arr;
};

}

#endif
