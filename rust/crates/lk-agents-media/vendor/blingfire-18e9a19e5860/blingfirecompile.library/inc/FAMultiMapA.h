/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_MULTI_MAP_A_H_
#define _FA_MULTI_MAP_A_H_

#include "FAConfig.h"
#include "FAMultiMapCA.h"

namespace BlingFire
{

/// General interface for Key -> { Val_1, ... ,Val_N } multi-map.
/// Holds:
/// \forall Key_1, Key_2 \in M, Key_1 != Key_2.

class FAMultiMapA : public FAMultiMapCA {

public:

    /// adds Key -> { Val_1, ... ,Val_N } pair
    /// if the ValuesCount == 0 the pointer pValues is not taken into account
    virtual void Set (
            const int Key, 
            const int * pValues, 
            const int ValuesCount
        ) = 0;

    /// before: Key -> NewVal, Vals = { Val_1, ... , Val_N }, Vals can be empty
    /// after : Key -> { Val_1, ... , Val_N, NewVal }
    virtual void Add (const int Key, const int Value) = 0;

    /// search for the next Key present that is greater than the passed Key
    /// returns the number of elements associated with this Key and pointer
    /// to its Values, -1 if not found (pointer value is not defined)
    /// (if the first Key is unknown, specify 0 to find it)
    virtual const int Next (int * pKey, const int ** ppValues) const = 0;

    /// same as Next
    /// (if the last Key is unknown, specify -1 to find it)
    virtual const int Prev (int * pKey, const int ** ppValues) const = 0;

};

}

#endif
