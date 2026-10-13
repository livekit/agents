/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_MAP_A_H_
#define _FA_MAP_A_H_

#include "FAConfig.h"

namespace BlingFire
{

///
/// General interface for Key -> Val map M.
///
/// Note: No duplicate Keys are not allowed.
///

class FAMapA {

public:

  /// returns pointer to the Value by the Key, NULL if not found
  virtual const int * Get (const int Key) const = 0;
  /// makes an association Key -> Value
  virtual void Set (const int Key, const int Value) = 0;
  /// search for the next Key present that is greater than the passed Key
  /// and returns the pointer to its Value, NULL if not found
  virtual const int * Next (int * pKey) const = 0;
  /// similar to the Next
  virtual const int * Prev (int * pKey) const = 0;
  /// removes Key -> Value pair
  virtual void Remove (const int Key) = 0;

};

}

#endif
