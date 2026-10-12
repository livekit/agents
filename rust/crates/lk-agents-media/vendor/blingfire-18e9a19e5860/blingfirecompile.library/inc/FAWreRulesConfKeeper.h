/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_WRERULESCONFKEEPER_H_
#define _FA_WRERULESCONFKEEPER_H_

#include "FAConfig.h"

namespace BlingFire
{

class FALDB;
class FAWREConfCA;
class FAMultiMapCA;
class FAWREConf_pack;
class FAMultiMap_pack;

///
/// Keeps compiled left and right sides of WRE rules and related execution
/// parameters.
///
/// Note: The pointers can be NULL, if yet not initialized
///

class FAWreRulesConfKeeper {

public:
    FAWreRulesConfKeeper ();
    ~FAWreRulesConfKeeper ();

public:
    /// this LDB will be used to get the data from
    void SetLDB (const FALDB * pLDB);
    /// initialization vector
    void Init (const int * pValues, const int Size);
    /// returns object into the initial state
    void Clear ();

public:
    /// returns compiled WRE configuration
    const FAWREConfCA * GetWre () const;
    /// returns actions associated with WRE rules
    const FAMultiMapCA * GetActions () const;
    /// returns true if case should be normalized
    const bool GetIgnoreCase() const;
    /// returns a minimal acceptable value for the unigram tag probaility
    /// the value is a log prob
    const float GetMinUniProb () const;

private:
    // input LDB
    const FALDB * m_pLDB;
    // data
    FAWREConf_pack * m_pWre;
    FAMultiMap_pack * m_pActs;
    bool m_IgnoreCase;
    float m_MinUniProb;
};

}

#endif
