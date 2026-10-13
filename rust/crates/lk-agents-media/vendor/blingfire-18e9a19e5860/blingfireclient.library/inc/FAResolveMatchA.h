/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_RESOLVEMATCHA_H_
#define _FA_RESOLVEMATCHA_H_

#include "FAConfig.h"

namespace BlingFire
{

///
/// This class describes the interface for match results disambiguation.
/// For example, in some cases overlapped results may be prohibited.
///
/// Usage:
///   1.1. AddReslut
///   1.2. AddReslut
///   1.N. ...
///   2. Process
///   3.1. GetResult
///   3.1. GetResult
///   3.M. ...
///   4. Clear
///
/// Note: "Rule" not necessarally refers to the rule number, it is the label
///   that identifies the action of the rule.
///

class FAResolveMatchA {

public:
    /// adds match result
    virtual void AddResult (
            const int Rule, 
            const int From, 
            const int To
        ) = 0;

    /// resolves conflicts
    virtual void Process () = 0;

    /// returns the number of results, after disambiguation
    virtual const int GetCount () const = 0;

    /// returns i-th result
    virtual void GetResult (
            const int i,
            int * pRule,
            int * pFrom,
            int * pTo
        ) const = 0;

    /// returns object into the initial state
    virtual void Clear () = 0;

};

}

#endif
