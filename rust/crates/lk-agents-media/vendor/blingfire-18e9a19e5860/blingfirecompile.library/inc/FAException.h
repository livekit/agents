/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_EXCEPTION_H_
#define _FA_EXCEPTION_H_

#include "FAConfig.h"
#include "FALimits.h"
#include "FAMsg.h"

namespace BlingFire
{

///
/// Base class for all FA* exceptions
///

class FAException {

public:
    /// creates exception with FAMsg::InternalError message
    FAException (
            const char * pSourceName,
            const int SourceLine
        );
    /// creates exception with custom error message
    FAException (
            const char * pErrMsg, 
            const char * pSourceName, 
            const int SourceLine
        );
    /// returns error message
    const char * GetErrMsg () const;
    /// returns error source file name
    const char * GetSourceName () const;
    /// returns error source file line
    const int GetSourceLine () const;

private:
    inline void CopyMsg (const char * pErrMsg);

private:
    const char * m_pSourceFile;
    int m_SourceLine;
    char m_ErrMsg [FALimits::MaxWordSize + 1];
};

}

/// throws an exception if (X) is false
#define FAAssert(X, M) \
  if(!(X)) { \
    throw FAException (M, __FILE__, __LINE__); \
  }

/// always throws an exception
#define FAError(M) \
    throw FAException (M, __FILE__, __LINE__);

/// throws an exception if S is a bad stream
#define FAAssertStream(S, F) \
    if (!(S) || (S)->bad() || (S)->fail() || (S)->eof()) { \
        std::string ErrMsg = "Bad file name or stream: "; \
        if(F != NULL) ErrMsg += F; \
        throw FAException (ErrMsg.c_str (), __FILE__, __LINE__); \
    }

#endif
