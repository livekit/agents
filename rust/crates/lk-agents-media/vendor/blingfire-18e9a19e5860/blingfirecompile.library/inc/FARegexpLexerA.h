/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_REGEXPLEXERA_H_
#define _FA_REGEXPLEXERA_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"
#include "FAToken.h"

namespace BlingFire
{

///
/// A common interface for regular-expression lexical analysers.
///

class FARegexpLexerA {

public:
  virtual void SetRegexp (const char * pReStr, const int Length) = 0;
  virtual void SetTokens (FAArray_cont_t < FAToken > * pTokens) = 0;
  virtual void Process () = 0;

};

}

#endif
