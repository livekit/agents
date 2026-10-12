/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_MAP_STD_H_
#define _FA_MAP_STD_H_

#include "FAConfig.h"
#include "FAMapA.h"

#include <map>

namespace BlingFire
{

class FAMap_std : public FAMapA {

public:

  FAMap_std ();
  virtual ~FAMap_std ();

public:

  const int * Get (const int Key) const;
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

    std::map < int, int > m_map;

};

}

#endif
