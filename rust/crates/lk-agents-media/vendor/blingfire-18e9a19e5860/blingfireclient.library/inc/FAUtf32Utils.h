/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_UTF32UTILS_H_
#define _FA_UTF32UTILS_H_

#include "FAConfig.h"
#include "FASecurity.h"

namespace BlingFire
{

///
/// Note: 
/// 1. All symbols are assumed to be UTF-32LE
/// 2. Transformation is made according to the common case folding from 
/// Unicode 4.1.0. Some of the transformations are irreversible, e.g. 
/// ANGSTREM SIGN, MICRO SIGN and so on, consider Unicode 4.1.0 for details.
///

/// returns upper case symbol, if inappropriate or not possible
/// returns unchanged Symbol
const int FAUtf32ToUpper (const int Symbol);

/// the same as above but converts the entire sequence
void FAUtf32StrUpper (__out_ecount (Size) int * pChain, const int Size);

/// returns lower case symbol, if inappropriate or not possible 
/// returns unchanged Symbol
const int FAUtf32ToLower (const int Symbol);

/// the same as above but converts the entire sequence
void FAUtf32StrLower (__out_ecount (Size) int * pChain, const int Size);

/// returns true if Symbol is in upper case
const bool FAUtf32IsUpper (const int Symbol);

/// returns true if Symbol is in upper case
const bool FAUtf32IsLower (const int Symbol);

}

#endif
