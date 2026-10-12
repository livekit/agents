/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_MAP_JUDY_H_
#define _FA_MAP_JUDY_H_

#include "FAConfig.h"

/// #define HAVE_JUDY_LIB

#ifndef HAVE_JUDY_LIB

#include "FAMap_std.h"

namespace BlingFire
{

typedef class FAMap_std FAMap_judy;

}

// of ifndef HAVE_JUDY_LIB
#else

#include "FAMapA.h"

#include <Judy.h>

namespace BlingFire
{

class FAMap_judy : public FAMapA {

public:

  FAMap_judy ();
  virtual ~FAMap_judy ();

public:

  const int* Get (const int Key) const;
  void Set (const int Key, const int Value);
  /// if the min Key is unknown, specify 0
  const int * Next (int * pKey) const;
  /// if the max Key is unknown, specify -1
  const int * Prev (int * pKey) const;
  /// removes Key -> Value pair,
  /// restructures the map so pointers returned becomes invalid!
  void Remove (const int Key);
  /// makes map as if it was just constructed
  void Clear ();

private:

  Pvoid_t m_map;
};

}

// of ifndef HAVE_JUDY_LIB
#endif

#endif
