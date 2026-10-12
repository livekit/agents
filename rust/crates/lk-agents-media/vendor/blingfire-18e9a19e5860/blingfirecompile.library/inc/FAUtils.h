/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_UTILS_H_
#define _FA_UTILS_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"
#include "FAUtils_cl.h"
#include "FAUtf32Utils.h"
#include "FAFsmConst.h"
#include "FAMultiMapCA.h"
#include "FAStringTokenizer.h"
#include "FASecurity.h"

namespace BlingFire
{

class FARSNfaA;
class FARSDfaA;
class FAMealyDfaA;
class FAMealyNfaA;
class FAMapA;
class FAMultiMapA;
class FATaggedTextCA;
class FATaggedTextA;

///
/// General purpose utility
///

/// makes output in binary mode
void FAIOSetup ();

/// makes input in binary mode
void FAInputIOSetup ();

/// Raises syntax error
///
///   pBuffer can be NULL if unknown
///   pMsg can be NULL if unknown
///   Offset can be -1 if unknown
///
void FASyntaxError (const char * pBuffer, // processing buffer pointer
                    const int BuffLength, // processing buffer length
                    const int Offset,     // error offset
                    const char * pMsg);   // error message


/// returns WIN32 code page number by the given encoding name
/// returns 0 if encoding name is unknown
const unsigned int FAEncoding2CodePage (const char * pEncStr);


/// implements comparison of indices by array values (for big to small sorting)
class FAIdxCmp_b2s {
public:
    FAIdxCmp_b2s (const int * pValues);

public:
    const bool operator () (const int Idx1, const int Idx2) const;

private:
    const int * m_pValues;
};


/// implements comparison of indices by array values (for small to big sorting)
class FAIdxCmp_s2b {
public:
    FAIdxCmp_s2b (const int * pValues);

public:
    const bool operator () (const int Idx1, const int Idx2) const;

private:
    const int * m_pValues;
};


/// returns true if the position in text is escaped
const bool FAIsEscaped (const int Pos, const char * pStr, const int StrLen);


/// returns true if FARSNfaA is in fact a DFA
const bool FAIsDfa (const FARSNfaA * pNfa);

/// builds alphabet for FARSNfaA
void FAGetAlphabet (const FARSNfaA * pNfa, FAArray_cont_t <int> * pA);

/// makes an NFA -> NFA copy
void FACopyNfa (FARSNfaA * pDstNfa, const FARSNfaA * pSrcNfa);

/// makes an DFA -> NFA copy
void FACopyDfa2Nfa (FARSNfaA * pDstNfa, const FARSDfaA * pSrcDfa);

/// makes an NFA -> DFA copy, possible only if FAIsDfa(pNfa) is true
void FACopyNfa2Dfa (FARSDfaA * pDstDfa, const FARSNfaA * pSrcNfa);


/// returns true if FAMultiMapA is empty
const bool FAIsEmpty (const FAMultiMapA * pMMap);


/// remaps OldDfa into NewDfa using Old2New Iw map,
///  in-place modification is not allowed
void FARemapRsFsmIws (
        const FARSDfaA * pOldDfa,
        FARSDfaA * pNewDfa,
        const FAMapA * pOld2New
    );

/// remaps OldNfa into NewNfa using Old2New Iw map,
///  in-place modification is not allowed
void FARemapRsFsmIws (
        const FARSNfaA * pOldNfa,
        FARSNfaA * pNewNfa,
        const FAMapA * pOld2New
    );

/// remaps OldSigma into NewSigma changing its Ows,
///  in-place modification is allowed
void FARemapMealySigma1 (
        const FARSDfaA * pOldDfa,
        const FAMealyDfaA * pOldSigma,
        FAMealyDfaA * pNewSigma,
        const FAMapA * pOld2New
    );

/// remaps OldSigma into NewSigma changing its Iws,
///  in-place modification is not allowed
void FARemapMealySigma2 (
        const FARSDfaA * pOldDfa,
        const FAMealyDfaA * pOldSigma,
        FAMealyDfaA * pNewSigma,
        const FAMapA * pOld2New
    );

/// remaps OldSigma into NewSigma changing its Iws,
///  in-place modification is not allowed
void FARemapMealySigma2 (
        const FARSNfaA * pOldNfa,
        const FAMealyNfaA * pOldSigma,
        FAMealyNfaA * pNewSigma,
        const FAMapA * pOld2New
    );


/// reverse N:1 map
void FAReverseMap (FAMultiMapA * pRevMap, const FAMapA * pMap);

/// copies tagged text
void FACopyTaggedText (FATaggedTextA * pOut, const FATaggedTextCA * pIn);


/// 
/// Converts a string of space-separated integers into an array.
///
/// Base specifies the number base to use (see the help for strtol() for more details)
///
/// Returns the actual array size, if the return value is bigger than 
/// MaxChainSize then array size should be increased to at least the
/// return value elements.
///
template < class Ty >
const int FAReadIntegerChain (
        const char * pLine, 
        const int LineLen, 
        const int Base, 
        __out_ecount(MaxOutSize) Ty * pChain, 
        const int MaxChainSize
    )
{
    FAStringTokenizer tokenizer;

    tokenizer.SetSpaces (" ");
    tokenizer.SetString (pLine, LineLen);

    int OutSize = 0;
    const char * pStr = NULL;
    int Len = 0;

    while (tokenizer.GetNextStr (&pStr, &Len)) {

        std::string buff (pStr, Len);
        const long C = strtol (buff.c_str (), NULL, Base);

        if (OutSize + 1 < MaxChainSize) {
            pChain [OutSize] = (Ty) C ;
        }

        OutSize++;
    }

    return OutSize;
}


/// 
/// Converts a string of space separated hexadecial numbers into an array.
///
/// Calling this function is equivalent to calling
///   FAReadIntegerChain(pLine, LineLen, 16, pChain, MaxChainSize)
///
/// Returns the actual array size, if the return value is bigger than 
/// MaxChainSize then array size should be increased to at least the
/// return value elements.
///
const int FAReadHexChain (
        const char * pLine, 
        const int LineLen, 
        __out_ecount(MaxOutSize) int * pChain, 
        const int MaxChainSize
    );

}

#endif
