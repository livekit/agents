/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_MULTI_MAP_CA_H_
#define _FA_MULTI_MAP_CA_H_

#include "FAConfig.h"
#include "FASecurity.h"

namespace BlingFire
{

///
/// Multi-Map client-side interface
///
/// Note:
/// 1. The interface allows (but does not specifies) mixed explicit/implicit
/// data representation.
/// 2. For mixed implementations int Get(int, int **) is a subset of 
/// int Get(int, int*, int). It may return -1 while there will be data 
/// associated with the input Key.
///

class FAMultiMapCA {

/// Interface methods to explicitly stored data
public:
    /// gets the Key -> { Val_1, ... , Val_N } pair
    /// returns the number of Values associated with the Key,
    /// returns -1 if there is no such Key in the map
    virtual const int Get (
            const int Key,
            const int ** ppValues
        ) const = 0;

/// Interface methods to implicitly stored data
public:
    /// Gets the Key -> { Val_1, ... , Val_N } pair. Returns the number of
    /// Values associated with the Key, or returns -1 if there is no such Key
    /// in the map. if pValues is NULL then the method does not use and just
    /// returns the count.
    virtual const int Get (
            const int Key,
            __out_ecount_opt(MaxCount) int * pValues,
            const int MaxCount
        ) const = 0;

    /// Returns the maximum possible size of values array, e.g.
    /// the maximum number of elements Get method can return.
    virtual const int GetMaxCount () const = 0;

};

}

#endif
