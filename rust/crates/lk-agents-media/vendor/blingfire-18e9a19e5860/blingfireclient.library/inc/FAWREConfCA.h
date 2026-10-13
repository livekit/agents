/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_WRECONFCA_H_
#define _FA_WRECONFCA_H_

#include "FAConfig.h"

namespace BlingFire
{

class FAArrayCA;
class FAState2OwCA;
class FAState2OwsCA;
class FAMealyDfaCA;
class FAMultiMapCA;
class FARSDfaCA;

///
/// An run-time interface to the compiled WRE data.
///

class FAWREConfCA {

public:
    /// returns WRE rule type
    virtual const int GetType () const = 0;
    /// returns token type
    virtual const int GetTokenType () const = 0;
    /// returns base tag Ow value
    virtual const int GetTagOwBase () const = 0;
    /// returns Txt digitizer's DFA, or NULL
    virtual const FARSDfaCA * GetTxtDigDfa () const = 0;
    /// returns Txt digitizer's Ows map, or NULL
    virtual const FAState2OwCA * GetTxtDigOws () const = 0;
    /// returns Dict digitizer, true if exists
    virtual const FAArrayCA * GetDictDig () const = 0;
    /// returns rules Dfa1, NULL if does not exist
    virtual const FARSDfaCA * GetDfa1 () const = 0;
    /// returns rules Dfa2, NULL if does not exist
    virtual const FARSDfaCA * GetDfa2 () const = 0;
    /// returns Moore automaton reaction, NULL if does not exist
    virtual const FAState2OwsCA * GetState2Ows () const = 0;
    /// returns Sigma1, NULL if does not exist
    virtual const FAMealyDfaCA * GetSigma1 () const = 0;
    /// returns Sigma2, NULL if does not exist
    virtual const FAMealyDfaCA * GetSigma2 () const = 0;
    /// returns TrBr map, NULL if does not exist
    virtual const FAMultiMapCA * GetTrBrMap () const = 0;
};

}

#endif
