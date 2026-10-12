/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_STRING_TOKENIZER_H_
#define _FA_STRING_TOKENIZER_H_

#include "FAConfig.h"
#include "FASecurity.h"

namespace BlingFire
{

///
/// This class splits input string into tokens by specified delimiters
///
/// Note: 
///   1. Methods Get* returns true if they finished successfully, 
///      false otherwise
///

class FAStringTokenizer {

public:
    FAStringTokenizer ();

public:

    /// sets up string of symbols that should be interpreted as space
    /// by default uses DefSpaces
    void SetSpaces (const char * pSpaces);
    /// sets up a new string for tokenization
    void SetString (const char * pStr, const int StrLen);

    /// gets next string value parsed from the input string
    const bool GetNextStr (const char ** ppStrToken, int * pTokenLen);
    /// gets next integer value parsed from the input string
    const bool GetNextInt (int * pIntToken, const int Base = 10);
    /// gets an array from the current position of the input string
    /// of the predefined size
    const bool GetArray (
            __out_ecount(ArraySize) int * pArray, 
            const int ArraySize
        );
    /// gets an array from the current position of the input string
    /// of the unknown size upto ArrayMaxSize
    const int GetArray2 (
            __out_ecount(ArrayMaxSize) int * pArray, 
            const int ArrayMaxSize
        );

private:
    inline const bool IsSpace (const unsigned char Symbol) const;

private:

    const char * m_pStr;
    int m_StrLen;
    int m_CurrPos;
    const char * m_pSpaces;

    static const char * const DefSpaces;
};

}

#endif
