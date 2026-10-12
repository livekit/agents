/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_UTILS_CL_H_
#define _FA_UTILS_CL_H_

#include "FAConfig.h"
#include "FAUtf32Utils.h"
#include "FAFsmConst.h"
#include "FAMultiMapCA.h"
#include "FASecurity.h"
#include "FALimits.h"

namespace BlingFire
{

class FARSNfaCA;
class FARSDfaCA;


///
/// General purpose utility
///

// returns new size of the array
const int FASortUniq (int * pBegin, int * pEnd);

/// returns true if the array is sorted
template < class Ty >
inline const bool FAIsSorted (const Ty * pArray, const int Size)
{
  for (int i = 1; i < Size; ++i) {

    DebugLogAssert (pArray);

    if (pArray [i - 1] > pArray [i]) {
      return false;
    }
  }

  return true;
}

/// returns true if the array is sorted and the elements are uniq
template < class Ty >
inline const bool FAIsSortUniqed (const Ty * pArray, const int Size)
{
  for (int i = 1; i < Size; ++i) {

    DebugLogAssert (pArray);

    if (pArray [i - 1] >= pArray [i]) {
      return false;
    }
  }

  return true;
}


/// returns index of the element or -1 if not found
template < class Ty >
inline const int FAFind_linear (const Ty * pBegin, const int Size, const Ty Val)
{
    DebugLogAssert (0 <= Size);

    for (int i = 0; i < Size; ++i) {

        DebugLogAssert (pBegin);

        if (Val == pBegin [i]) {
            return i;
        }
    }

    return -1;
}



/// returns index of the element or -1 if not found
template < class Ty >
inline const int FAFind_log (const Ty * pBegin, const int Size, const Ty Val)
{
    DebugLogAssert (0 <= Size);

    // this optimization is helpful for automata arc lookup.
    if (Val >= 0 && (int)Val < Size && pBegin[Val] == Val)
    {
        return Val;
    }

    /// logarithmic search plus linear at the end

    int From = 0;
    int To = Size - 1;

    while (8 < To - From) {

        DebugLogAssert (pBegin);

        const int Pos = ((unsigned int)(To + From)) >> 1;
        const Ty CurrVal = pBegin [Pos];

        if (Val == CurrVal) {

            return Pos;

        } else if (Val < CurrVal) {

            To = Pos - 1;

        } else {

            From = Pos + 1;
        }
    } // of while (8 < To - From) ...

    while (From <= To) {

        DebugLogAssert (pBegin);

        const Ty CurrVal = pBegin [From];

        if (Val > CurrVal)
            From++;
        else if (Val == CurrVal)
            return From;
        else
            return -1;

    } // of while (From <= To)

    return -1;
}


/// returns a position of the first element which equal to the given or less
template < class Ty > 
inline const int FAFindEqualOrLess_log (
        const Ty * pBegin,
        const int Size,
        const Ty Val
    )
{
    DebugLogAssert (0 <= Size);
    DebugLogAssert (FAIsSortUniqed (pBegin, Size));

    // this optimization is helpful for automata arc lookup.

    if (Val >= 0 && (int)Val < Size && pBegin[Val] == Val)
    {
        return Val;
    }

    /// logarithmic search plus linear at the end

    int From = 0;
    int To = Size - 1;

    while (8 < To - From) {

        DebugLogAssert (pBegin);

        const int Pos = ((unsigned int)(To + From)) >> 1;
        const Ty CurrVal = pBegin [Pos];

        if (Val == CurrVal) {

            return Pos;

        } else if (Val < CurrVal) {

            To = Pos - 1;

        } else {

            From = Pos + 1;
        }
    } // of while (8 < To - From) ...

    while (From <= To) {

        DebugLogAssert (pBegin);

        const Ty CurrVal = pBegin [From];

        if (Val == CurrVal)
            return From;
        else if (Val < CurrVal)
            return From - 1;

        From++;

    } // of while (From <= To)

    return From - 1;
}


/// If pWord belong to the 0-separated pWordList then
/// function returns pWord's position, otherwise it returns -1.
/// O(N) - with very small constant.
template < class Ty >
inline const int FAFindWordPos (
        const Ty * pWordList,
        const int WordListSize,
        const Ty * pWord,
        const int WordSize
    )
{
    DebugLogAssert (0 < WordSize && pWord);
    DebugLogAssert (0 < WordListSize && pWordList);

    int Pos = 0;
    int CurrWordPos = 0;

    while (Pos < WordListSize) {

        if (0 == pWordList [Pos]) {

            const int CurrWordSize = Pos - CurrWordPos;
            DebugLogAssert (0 < CurrWordSize);

            if (CurrWordSize == WordSize &&
                0 == memcmp (pWord, pWordList + CurrWordPos, CurrWordSize * sizeof (Ty))) {
                return CurrWordPos;
            }

            CurrWordPos = Pos + 1;

        } // of if (0 == pWordList [Pos]) ...

        Pos++;
    }

    return -1;
}




/// returns true if FARSNfaCA is invalid or empty
const bool FAIsValidNfa (const FARSNfaCA * pNfa);

/// returns true if FARSDfaCA is invalid or empty
const bool FAIsValidDfa (const FARSDfaCA * pDfa);


/// returns sequence case type, e.g. one of the FAFsmConst::CASE_* constants
template < class Ty >
const int FAGetCaseType (const Ty * pChain, const int Size)
{
    if (0 < Size) {

        bool fCap = true;
        bool fAllLo = true;
        bool fAllUp = true;

        const Ty Symbol = pChain [0];

        const bool Lo = FAUtf32IsLower (Symbol);
        const bool Up = FAUtf32IsUpper (Symbol);

        if (Lo) {
            fAllUp = false;
            fCap = false;
        } else if (Up) {
            fAllLo = false;
        }

        int i = 0;

        while (++i < Size && (fCap || fAllLo || fAllUp)) {

            const Ty Symbol = pChain [i];

            const bool Lo = FAUtf32IsLower (Symbol);
            const bool Up = FAUtf32IsUpper (Symbol);

            if (Lo) {
                fAllUp = false;
            } else if (Up) {
                fAllLo = false;
                fCap = false;
            }

        } // of while (++i < Size && (fCap || fAllLo || fAllUp)) ...

        if (fAllLo) {
            return FAFsmConst::CASE_ALL_LOWER;
        } else if (fCap) {
            return FAFsmConst::CASE_CAPITALIZED;
        } else if (fAllUp) {
            return FAFsmConst::CASE_ALL_UPPER;
        }
    }

    /// that is something else
    return FAFsmConst::CASE_OTHER;
}



/// normalizes pIn string with respect to the given map
/// function returns the size of the output string
/// !!! normalization cannot be done in-place !!!
template < class Ty >
inline const int FANormalize (
        const Ty * pIn,
        const int InCount,
        __out_ecount(MaxOutSize) Ty * pOut,
        const int MaxOutSize,
        const FAMultiMapCA * pMap
    )
{
    DebugLogAssert (pIn != pOut);
    DebugLogAssert (pMap);
    DebugLogAssert (0 == MaxOutSize || NULL != pOut);

    const int MaxNormCount = 10;
    int Norm [MaxNormCount];

    int OutSize = 0;

    for (int i = 0; i < InCount; ++i) {

        const Ty Ci = pIn [i];
        const int NormCount = pMap->Get (Ci, Norm, MaxNormCount);

        if (-1 == NormCount) {

            if (OutSize < MaxOutSize) {
                pOut [OutSize] = Ci;
            }
            OutSize++;

        } else if (1 == NormCount) {

            if (OutSize < MaxOutSize) {
                pOut [OutSize] = (Ty) Norm [0];
            }
            OutSize++;

        } else if (1 < NormCount && NormCount <= MaxNormCount) {

            // see how much of the buffer left, can be 0 or less 
            int CopyCount = MaxOutSize - OutSize;

            // CopyCount = MIN {left buffer size, NormCount}
            if (NormCount < CopyCount) {
                CopyCount = NormCount;
            }

            for (int j = 0; j < CopyCount; ++j) {
                const Ty Co = (Ty) Norm [j];
                pOut [OutSize + j] = Co;
            }

            OutSize += NormCount;

        } // of if (-1 == NormCount) ...
    } // of for (int i = 0; ...

    return OutSize;
}

/// normalizes pIn string with respect to the given map
/// function returns the size of the output string
/// !!! normalization cannot be done in-place !!!
template < class Ty >
inline const int FANormalize (
        const Ty * pIn,
        const int InCount,
        __out_ecount(MaxOutSize) Ty * pOut,
        __out_ecount(MaxOutSize) int * pOffsets,
        const int MaxOutSize,
        const FAMultiMapCA * pMap
    )
{
    DebugLogAssert (pIn != pOut);
    DebugLogAssert (pMap);
    DebugLogAssert (0 == MaxOutSize || (NULL != pOut && NULL != pOffsets));

    const int MaxNormCount = 10;
    int Norm [MaxNormCount];

    int OutSize = 0;

    for (int i = 0; i < InCount; ++i) {

        const Ty Ci = pIn [i];
        const int NormCount = pMap->Get (Ci, Norm, MaxNormCount);

        if (-1 == NormCount) {

            if (OutSize < MaxOutSize) {
                pOut [OutSize] = Ci;
                pOffsets [OutSize] = i;
            }
            OutSize++;

        } else if (1 == NormCount) {

            if (OutSize < MaxOutSize) {
                pOut [OutSize] = (Ty) Norm [0];
                pOffsets [OutSize] = i;
            }
            OutSize++;

        } else if (1 < NormCount && NormCount <= MaxNormCount) {

            // see how much of the buffer left, can be 0 or less 
            int CopyCount = MaxOutSize - OutSize;

            // CopyCount = MIN {left buffer size, NormCount}
            if (NormCount < CopyCount) {
                CopyCount = NormCount;
            }

            for (int j = 0; j < CopyCount; ++j) {
                const Ty Co = (Ty) Norm [j];
                pOut [OutSize + j] = Co;
                pOffsets [OutSize + j] = i;
            }

            OutSize += NormCount;

        } // of if (-1 == NormCount) ...
    } // of for (int i = 0; ...

    return OutSize;
}


/// normalizes input word (IN-PLACE is allowed)
/// a new word length is returned
template < class Ty >
inline const int FANormalizeWord (
        const Ty * pIn,
        const int InCount,
        __out_ecount(MaxOutSize) Ty * pOut,
        const int MaxOutSize,
        const FAMultiMapCA * pMap
    )
{
    if (0 < InCount && InCount <= FALimits::MaxWordLen) {

        DebugLogAssert (pIn);

        // allocate the buffer of the maximum word length
        Ty Tmp [FALimits::MaxWordLen];

        // assume not in-place
        Ty * pDst = pOut;
        int DstMaxSize = MaxOutSize;

        // switch to Tmp, if in-place
        if (pIn == pOut) {
            pDst = Tmp;
            DstMaxSize = FALimits::MaxWordLen;
        }

        // make normalization, into a destination buffer
        const int OutCount = FANormalize (pIn, InCount, \
            pDst, DstMaxSize, pMap);

        // see if the output completelly fits the destination buffer
        if (0 <= OutCount && OutCount <= DstMaxSize) {
            // if not in-place then all done
            if (pDst != Tmp) {
                return OutCount;
            // it's in-place, see if the output buffer is large enough
            } else if (OutCount <= MaxOutSize) {
                memcpy (pOut, Tmp, sizeof (Ty) * OutCount);
                return OutCount;
            }
        }
    }

    // the input or resulting words are out of word length limits
    // or the output does not fit the output buffer size
    return 0;
}


/// returns a CRC32 hash key for the buffer of data
///  crc can be used to get a combined value with the hash of previous buffer(s).
const unsigned int FAGetCrc32 (const unsigned char *buf, size_t size, unsigned int crc = 0);

}

#endif
