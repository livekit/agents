/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_RSNFA_A_H_
#define _FA_RSNFA_A_H_

#include "FAConfig.h"
#include "FARSNfaCA.h"

namespace BlingFire
{

///
/// Common interface for Rabin-Scott Nfa.
///
/// General agreement for this interface:
///   All arrays are expected to be sorted from smaller to bigger numbers,
///   if for some implementation it is not possible, this should be stated
///   explicitly in the comments.
///

class FARSNfaA : public FARSNfaCA {

/// read interface
public:

    /// inherited from FARSNfaCA
    virtual const int GetDest (
            const int State,
            const int Iw,
            int * pDstStates,
            const int MaxCount
        ) const = 0;

    /// returns the maximum state number, -1 if any
    virtual const int GetMaxState () const = 0;
    /// returns the maximum Iw, -1 if any
    virtual const int GetMaxIw () const = 0;
    /// returns the final states
    virtual const int GetFinals (const int ** ppStates) const = 0;
    /// returns true if there is at least one final within the pStates
    virtual const bool IsFinal (const int * pStates, const int Size) const = 0;
    /// returns State's input weights
    virtual const int GetIWs (const int State,
                              const int ** ppIws) const = 0;
    /// for the given State and the Input weight returns Destination states 
    /// returns -1, if there is no mapping
    /// returns  0, if the set of destination states is empty
    virtual const int GetDest (const int State, 
                               const int Iw,
                               const int ** ppIwDstStates) const = 0;

/// write interface
public:

    /// sets the maximum state number, q \in [0, MaxState]
    virtual void SetMaxState (const int MaxState) = 0;
    /// sets the maximum possible state, iw \in [0, MaxIw]
    virtual void SetMaxIw (const int MaxIw) = 0;
    /// makes Nfa to be ready
    virtual void Create () = 0;
    /// Adds initial state
    virtual void SetInitials (const int * pStates, const int StateCount) = 0;
    /// Adds finals state
    virtual void SetFinals (const int * pStates, const int StateCount) = 0;
    /// Adds or alters a transitions
    virtual void SetTransition (const int FromState,
                                const int Iw,
                                const int * pDstStates,
                                const int DstStatesCount) = 0;
    /// adds a single transition
    virtual void SetTransition (
            const int FromState, 
            const int Iw, 
            const int DstState
        ) = 0;
    // have to be called after all transitions have been added
    virtual void Prepare () = 0;
    // return Nfa into the state as if constructor has just been called
    virtual void Clear () = 0;
};

}

#endif
