/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_GLOBALCONFKEEPER_H_
#define _FA_GLOBALCONFKEEPER_H_

#include "FAConfig.h"

namespace BlingFire
{

class FALDB;
class FARSDfaCA;
class FARSDfa_pack_triv;

///
/// Keeps global client configuration options
///

class FAGlobalConfKeeper {

public:
    FAGlobalConfKeeper ();
    ~FAGlobalConfKeeper ();

public:
    /// this LDB will be used to get the data from
    void SetLDB (const FALDB * pLDB);
    /// initialization vector
    void Init (const int * pValues, const int Size);
    /// returns object into the initial state
    void Clear ();

public:
    /// true if reductive stemming is expected
    const bool GetDoW2B () const;
    /// returns a prefix automaton
    const FARSDfaCA * GetPrefixes () const;
    /// returns a SUFFIX automaton
    const FARSDfaCA * GetSuffixes () const;

private:
    // input LDB
    const FALDB * m_pLDB;
    bool m_DoW2B;
    FARSDfa_pack_triv * m_pDfa;
    FARSDfa_pack_triv * m_pSuffDfa;
};

}

#endif
