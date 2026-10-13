/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_TOKEN_H_
#define _FA_TOKEN_H_

#include "FAConfig.h"

namespace BlingFire
{

class FAToken {

public:

  FAToken ();
  FAToken (const int Type, const int Offset, const int Length);

  const int GetType () const;
  void SetType (const int Type);

  const int GetOffset () const;
  void SetOffset (const int Offset);

  const int GetLength () const;
  void SetLength (const int Length);

private:

  int m_type;
  int m_offset;
  int m_length;

};

}

#endif
