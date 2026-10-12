/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_PRINT_UTILS_H_
#define _FA_PRINT_UTILS_H_

#include "FAConfig.h"

#include <iostream>

namespace BlingFire
{

class FARegexpTree;
class FARegexpTree2Funcs;
class FAToken;
class FAWREToken;
class FATagSet;


/// template function for printing of arrays
/// array elements must support << operator
template <class Ty>
void FAPrintArray (std::ostream& os, const Ty * pA, const int Size)
{
  if (NULL != pA) {

    os << "[";
    for (int i = 0; i < Size; ++i) {

      os << " " << (int) pA [i];
    }
    os << " ]";

  } else {

    os << "NULL";
  }
}

/// template function for reversed printing of arrays
/// array elements must support << operator
template <class Ty>
void FAPrintArray_rev (std::ostream& os, const Ty * pA, const int Size)
{
  if (NULL != pA) {

    os << "[";
    for (int i = Size - 1; i >= 0; --i) {

      os << " " << (int) pA [i];
    }
    os << " ]";

  } else {

    os << "NULL";
  }
}

template<>
void FAPrintArray (std::ostream& os, const FAToken * pA, const int Size);

template<>
void FAPrintArray (std::ostream& os, const FAWREToken * pA, const int Size);

template<>
void FAPrintArray (std::ostream& os, const float * pA, const int Size);


/// prints out RegexpTree in dotty format
void FAPrintRegexpTree (
        std::ostream& os,
        const FARegexpTree * pTree,
        const char * pRegexp
    );

/// prints regexp functions in dotty format
void FAPrintRegexpFuncs (
        std::ostream& os,
        const FARegexpTree2Funcs * pFuncs,
        const FARegexpTree * pTree
    );

/// FAToken << operator
std::ostream& operator <<(std::ostream& os, const FAToken& Token);

/// FAWREToken << operator
std::ostream& operator <<(std::ostream& os, const FAWREToken& Token);


/// prints 0-separated list of words (in UTF-8)
void FAPrintWordList (
        std::ostream& os, 
        const int * pWordList, 
        const int Count,
        const char Delim = ' '
    );

/// prints splitted chain (in UTF-8)
void FAPrintSplitting (
        std::ostream& os,
        const int * pChain,
        const int ChainSize,
        const int * pEnds,
        const int EndsCount,
        const char Delim = ' '
    );

/// print a list of tags (in ASCII)
void FAPrintTagList (
        std::ostream& os,
        const FATagSet * pTagSet,
        const int * pTags,
        const int Count,
        const char Delim = ' '
    );


/// prints word (in UTF-8)
void FAPrintWord (
        std::ostream& os,
        const int * pWord,
        const int WordLen
    );

/// prints tagged word (in UTF-8)
void FAPrintTaggedWord (
        std::ostream& os,
        const FATagSet * pTagSet,
        const int * pWord,
        const int WordLen,
        const int Tag
    );


/// prints a single value (in ASCII)
void FAPrintValue (
        std::ostream& os,
        const int Value,
        const int Width,
        const bool Hex
    );

/// prints a chain of digitis (in ASCII)
void FAPrintChain (
        std::ostream& os,
        const int * pChain,
        const int ChainSize,
        const int Dir,
        const int Width,
        const bool Hex
    );


/// prints hyphenated word in UTF-8
void FAPrintHyphWord (
        std::ostream& os,
        const int * pWord,
        const int * pHyphs,
        const int WordLen
    );

}

#endif
