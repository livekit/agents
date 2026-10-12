/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_WRECONFA_H_
#define _FA_WRECONFA_H_

#include "FAConfig.h"
#include "FAWREConfCA.h"

namespace BlingFire
{

class FARSDfaA;
class FAState2OwA;
class FAState2OwsA;
class FAMealyDfaA;
class FAMultiMapA;

///
/// An commpile-time interface to the compiled WRE data.
///

class FAWREConfA : public FAWREConfCA {

/// inherited from FAWREConfCA and needs to be repeated
public:
    virtual const FARSDfaCA * GetDfa1 () const = 0;
    virtual const FARSDfaCA * GetDfa2 () const = 0;
    virtual const FAState2OwsCA * GetState2Ows () const = 0;
    virtual const FAMealyDfaCA * GetSigma1 () const = 0;
    virtual const FAMealyDfaCA * GetSigma2 () const = 0;
    virtual const FAMultiMapCA * GetTrBrMap () const = 0;

/// read interface
public:
    /// returns Txt digitizer, NULL if does not exist
    virtual void GetTxtDigititizer (
            const FARSDfaA ** ppDfa,
            const FAState2OwA ** ppState2Ow
        ) const = 0;
    /// returns Dict digitizer, NULL if does not exist
    virtual void GetDictDigitizer (
            const int ** ppId2Ow,
            int * pId2OwSize
        ) const = 0;
    /// returns rules Dfa1, NULL if does not exist
    virtual void GetDfa1 (const FARSDfaA ** ppDfa) const = 0;
    /// returns rules Dfa2, NULL if does not exist
    virtual void GetDfa2 (const FARSDfaA ** ppDfa) const = 0;
    /// returns Moore automaton reaction, NULL if does not exist
    virtual void GetState2Ows (const FAState2OwsA ** ppState2Ows) const = 0;
    /// returns Sigma1, NULL if does not exist
    virtual void GetSigma1 (const FAMealyDfaA ** ppSigma) const = 0;
    /// returns Sigma2, NULL if does not exist
    virtual void GetSigma2 (const FAMealyDfaA ** ppSigma) const = 0;
    /// returns TrBr map, NULL if does not exist
    virtual void GetTrBrMap (const FAMultiMapA ** ppTrBr) const = 0;

/// write interface
public:
    /// sets up WRE rule type
    virtual void SetType (const int Type) = 0;
    /// sets up token type
    virtual void SetTokenType (const int TokenType) = 0;
    /// sets up Tag-Ow base
    virtual void SetTagOwBase (const int TagOwBase) = 0;
    /// returns writable text digitizer containers
    virtual void GetTxtDigititizer (
            FARSDfaA ** ppDfa, 
            FAState2OwA ** ppState2Ow
        ) = 0;
    /// sets up dict-digitizer
    virtual void SetDictDigitizer (const int * pId2Ow, const int Size) = 0;
    /// returns writable rules automaton
    virtual void GetDfa1 (FARSDfaA ** ppDfa) = 0;
    /// returns writable rules2 automaton
    virtual void GetDfa2 (FARSDfaA ** ppDfa) = 0;
    /// returns writable State -> Ows map
    virtual void GetState2Ows (FAState2OwsA ** ppState2Ows) = 0;
    /// returns writable Sigma1 map
    virtual void GetSigma1 (FAMealyDfaA ** ppSigma) = 0;
    /// returns writable Sigma2 map
    virtual void GetSigma2 (FAMealyDfaA ** ppSigma) = 0;
    /// returns writable TrBr map
    virtual void GetTrBrMap (FAMultiMapA ** ppTrBr) = 0;
    /// returns object into initial state
    virtual void Clear () = 0;

};

}

#endif
