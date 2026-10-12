/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_PARSETREEA_H_
#define _FA_PARSETREEA_H_

#include "FAConfig.h"

namespace BlingFire
{

///
/// Parsing tree interface.
///

class FAParseTreeA {

public:
    /// returns upper level nodes
    virtual const int GetUpperNodes (const int ** ppNodes) const = 0;
    /// returns upper level labels
    virtual const int GetUpperLabels (const int ** ppLabels) const = 0;
    /// returns right neighbour, -1 if no right neighbour
    virtual const int GetNext (const int Node) const = 0;
    /// returns left child, -1 if no left child
    virtual const int GetChild (const int Node) const = 0;
    /// returns node's label
    virtual const int GetLabel (const int Node) const = 0;

public:
    /// sets up initial upper level as a 0-based array of contiguous values
    virtual void Init (const int Count) = 0;
    /// add a constituent at upper level
    virtual void AddNode (const int Label, const int From, const int To) = 0;
    /// should be called before GetUpper* call if there were AddNode calls
    virtual void Update () = 0;

};

}

#endif
