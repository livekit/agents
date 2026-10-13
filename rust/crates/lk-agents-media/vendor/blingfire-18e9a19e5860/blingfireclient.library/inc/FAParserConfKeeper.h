/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_PARSERCONFKEEPER_H_
#define _FA_PARSERCONFKEEPER_H_

#include "FAConfig.h"

namespace BlingFire
{

class FALDB;
class FAWREConfCA;
class FAWREConf_pack;
class FAMultiMapCA;
class FAMultiMap_pack;
class FARSDfaCA;

///
/// Keeps parser's configration
///
/// Note: The pointers can be NULL, if yet not initialized
///

class FAParserConfKeeper {

public:
    FAParserConfKeeper ();
    ~FAParserConfKeeper ();

public:
    /// initialization vector
    void Initialize (const FALDB * pLDB, const int * pValues, const int Size);
    /// returns object into the initial state
    void Clear ();

public:
    /// returns compiled WRE configuration
    const FAWREConfCA * GetWre () const;
    /// returns a map of actions
    const FAMultiMapCA * GetActs () const;
    /// returns an optional map with action data, if this map does not exist returns NULL
    const FAMultiMapCA * GetActData () const;
    /// returns true if the case should be ignored
    const bool GetIgnoreCase() const;
    /// returns maximum allowed function call depth
    const int GetMaxDepth () const;
    /// returns maximum allowed passes over the input
    const int GetMaxPassCount () const;
    /// returns an array of function initial states, the array contains -1
    /// for undefined function ids
    const unsigned int GetFnIniStates (const int ** ppFn2Ini) const;

public:
    void SetWre (const FAWREConfCA * pWre);
    void SetActs (const FAMultiMapCA * pActs);
    void SetActData (const FAMultiMapCA * pActData);
    void SetIgnoreCase (const bool IgnoreCase);
    void SetMaxDepth (const int MaxDepth);

private:
    /// helper method, calculates an initial state for the FnId
    static inline const int GetFnId2State (const int FnId, const FARSDfaCA * pDfa, const int TokenType, const int TagOwBase);
    /// helper method, calculates m_pFn2Ini mapping
    inline void CalcFnInitialStates (const FAWREConfCA * pDfa, const FAMultiMapCA * pActs);

private:
    FAWREConf_pack * m_pWre;
    FAMultiMap_pack * m_pActs;
    FAMultiMap_pack * m_pActData;
    bool m_IgnoreCase;
    int m_MaxDepth;
    int m_MaxPassCount;

    const FAWREConfCA * m_pWreCA;
    const FAMultiMapCA * m_pActsCA;
    const FAMultiMapCA * m_pActDataCA;

    /// maps function id into an initial state, or to -1 if not valid
    int * m_pFn2Ini;
    unsigned int m_Fn2IniSize;

    enum {
        DefMaxDepth = 5,
        DefMaxPassCount = 1,
        MinActSize = 3,
    };
};

}

#endif
