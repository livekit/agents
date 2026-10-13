/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_WRECOMPILER_H_
#define _FA_WRECOMPILER_H_

#include "FAConfig.h"
#include "FAWREConfA.h"
#include "FAWRERules2TokenNfa.h"
#include "FAWRETokens2Digitizers.h"
#include "FATokenNfa2TupleNfa.h"
#include "FADictInterpreter_t.h"
#include "FANfa2Dfa_t.h"
#include "FADfa2MinDfa_hg_t.h"
#include "FAEpsilonRemoval.h"
#include "FAChain2Num_hash.h"
#include "FAMultiMap_judy.h"
#include "FARSNfa_wo_ro.h"
#include "FARSNfa_ar_judy.h"
#include "FARSDfaRenum_remove_gaps.h"
#include "FARSDfa_renum.h"
#include "FARSDfa_ro.h"
#include "FARSDfa2MooreDfa.h"
#include "FAState2Ows.h"
#include "FAMealyNfa.h"
#include "FAMealyDfa.h"
#include "FATrBrNfa2MealyNfa.h"
#include "FAUnAmbiguous.h"
#include "FACmpTrBrOws_greedy.h"
#include "FAMealyNfa2Dfa.h"
#include "FACalcIwEqClasses.h"
#include "FAArray_p2ca.h"

namespace BlingFire
{

class FATagSet;
class FAAllocatorA;

///
/// WRE compiler
///

class FAWRECompiler : public FAWREConfA {

public:
    FAWRECompiler (FAAllocatorA * pAlloc);
    virtual ~FAWRECompiler ();

public:
    /// setup methods, (all input pointers can be NULL)
    void SetType (const int Type);
    void SetEncodingName (const char * pEncName);
    void SetTagSet (const FATagSet * pTagSet);
    void SetTagSet2 (const FATagSet * pTagSet2);
    void SetTagDict (const FADictInterpreter_t < int > * pTagDict); // TODO: change this to LDB
    void SetDictRoot (const char * pDictRoot);

    /// compilation methods
    void AddRule (const char * pWRE, const int Length);
    void Process ();
    void Clear ();

    /// following three methods are needed for input external digitizer
    void SetTokens (FAChain2NumA * pTokens);
    void SetCNF (FAMultiMapA * pTokenNum2CNF);
    void SetType2Ows (const FAMultiMapA * pType2Ows);

/// FAWREConfA implementation
public:
    const int GetType () const;
    const int GetTokenType () const;
    const int GetTagOwBase () const;
    const FARSDfaCA * GetTxtDigDfa () const;
    const FAState2OwCA * GetTxtDigOws () const;
    const FAArrayCA * GetDictDig () const;
    const FARSDfaCA * GetDfa1 () const;
    const FARSDfaCA * GetDfa2 () const;
    const FAState2OwsCA * GetState2Ows () const;
    const FAMealyDfaCA * GetSigma1 () const;
    const FAMealyDfaCA * GetSigma2 () const;
    const FAMultiMapCA * GetTrBrMap () const;
    void GetTxtDigititizer (
            const FARSDfaA ** ppDfa,
            const FAState2OwA ** ppState2Ow
        ) const;
    void GetDictDigitizer (
            const int ** ppId2Ow,
            int * pId2OwSize
        ) const;
    void GetDfa1 (const FARSDfaA ** ppDfa) const;
    void GetDfa2 (const FARSDfaA ** ppDfa) const;
    void GetState2Ows (const FAState2OwsA ** ppState2Ows) const;
    void GetSigma1 (const FAMealyDfaA ** ppSigma) const;
    void GetSigma2 (const FAMealyDfaA ** ppSigma) const;
    void GetTrBrMap (const FAMultiMapA ** ppTrBr) const;

/// from FAWREConfA (write -- not implemented)
private:
    void SetTokenType (const int TokenType);
    void SetTagOwBase (const int TagOwBase);
    void GetTxtDigititizer (FARSDfaA ** ppDfa, FAState2OwA ** ppState2Ow);
    void SetDictDigitizer (const int * pId2Ow, const int Size);
    void GetDfa1 (FARSDfaA ** ppDfa);
    void GetDfa2 (FARSDfaA ** ppDfa);
    void GetState2Ows (FAState2OwsA ** ppState2Ows);
    void GetSigma1 (FAMealyDfaA ** ppSigma);
    void GetSigma2 (FAMealyDfaA ** ppSigma);
    void GetTrBrMap (FAMultiMapA ** ppTrBr);

public:
    /// following three methods are needed for output external digitizer
    const FAChain2NumA * GetTokens () const;
    const FAMultiMapA * GetCNF () const;
    const FAMultiMapA * GetType2Ows () const;

private:
    void BuildRs ();
    void BuildMoore ();
    void BuildMealy ();

private:
    int m_Type;
    const FARSDfaA * m_pDfa1;
    const FARSDfaA * m_pDfa2;
    const FAState2OwsA * m_pMooreOws;
    const FAMealyDfaA * m_pSigma1;
    const FAMealyDfaA * m_pSigma2;
    const FAMultiMapA * m_pTrBrMap;

    /// WRE --> Token Nfa
    FAWRERules2TokenNfa m_wre2tnnfa;
    /// keeps tokens, if not supplied from outside
    FAChain2Num_hash m_token2num;
    /// points to the current container of tokens
    FAChain2NumA * m_pToken2Num;

    /// Tokens --> Digitizers + Token2Ows map
    FAWRETokens2Digitizers m_tns2dgs;
    /// keeps CNF, if not supplied from outside
    FAMultiMap_judy m_cnf;
    /// points to the current container of CNF
    FAMultiMapA * m_pCNF;
    /// points to the current container of Type2Ows map
    const FAMultiMapA * m_pType2Ows;
    /// client-side like array of dictionary digitizer
    FAArray_p2ca m_dct_arr;

    /// Token Nfa + Token2Ows map --> Nfa
    FATokenNfa2TupleNfa m_tnnfa2nfa;
    FARSNfa_wo_ro m_nfa;

    /// eNFA --> NFA, needed for Moore only
    FAEpsilonRemoval m_e_removal;
    FARSNfa_ar_judy m_nfa_rw; // used for both Moore and Mealy

    /// NFA -> DFA
    FANfa2Dfa_t < FARSNfaA, FARSDfa_wo_ro > m_nfa2dfa;
    FARSDfa_wo_ro m_dfa;

    /// DFA -> Min DFA
    FADfa2MinDfa_hg_t < FARSDfa_wo_ro, FARSDfa_ro > m_dfa2mindfa;
    FARSDfa_ro m_min_dfa;

    /// Min DFA --> Min DFA no gaps
    FARSDfaRenum_remove_gaps m_renum;
    FARSDfa_renum m_renum_dfa;

    /// DFA --> Moore DFA, needed for Moore only
    FARSDfa2MooreDfa m_rs2moore;
    FARSDfa_ro m_dfa1;
    FAState2Ows m_moore_ows;

    /// Rs Nfa --> Mealy Nfa, needed for Mealy only
    FATrBrNfa2MealyNfa m_nfa2mealy;
    FAMealyNfa m_mealy_ows;

    /// Mealy Nfa --> functional Mealy Nfa
    FACmpTrBrOws_greedy m_trbr_less;
    FAUnAmbiguous m_unamb;

    /// Mealy Nfa --> Mealy Dfa(s)
    FAMealyNfa2Dfa m_nfa2dfa_mealy;
    FAMealyDfa m_sigma1;
    FARSDfa_ro m_dfa2;
    FAMealyDfa m_sigma2;

    /// Mealy Dfa1 + Mealy Dfa1 -- mutual alphabet minimization
    FACalcIwEqClasses m_iw_classify;
    FAMap_judy m_old2new;
    FARSDfa_wo_ro m_dfa2_min;
    FAMealyDfa m_sigma2_min;
};

}

#endif
