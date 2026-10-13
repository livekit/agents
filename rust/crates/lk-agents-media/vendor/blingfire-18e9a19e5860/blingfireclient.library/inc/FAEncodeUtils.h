/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_ENCODE_UTILS_H_
#define _FA_ENCODE_UTILS_H_

#include "FAConfig.h"
#include "FAFsmConst.h"


// Encodes int with variable-length prefix code
#define FAEncodeIntPrefix(V, pOut)                          \
                                                            \
    if (0x0000007F >= V) {                                  \
                                                            \
      *pOut = (unsigned char) V;                            \
      pOut++;                                               \
                                                            \
    } else if (0x00003FFF >= V) {                           \
                                                            \
      *pOut = (unsigned char) (0x80 | (V >> 8));            \
      pOut++;                                               \
      *pOut = (unsigned char) (0xFF & V);                   \
      pOut++;                                               \
                                                            \
    } else if (0x001FFFFF >= V) {                           \
                                                            \
      *pOut = (unsigned char) (0xC0 | (V >> 16));           \
      pOut++;                                               \
      *pOut = (unsigned char) ((0xFF00 & V) >> 8);          \
      pOut++;                                               \
      *pOut = (unsigned char) (0xFF & V);                   \
      pOut++;                                               \
                                                            \
    } else if (0x0FFFFFFF >= V) {                           \
                                                            \
      *pOut = (unsigned char) (0xE0 | (V >> 24));           \
      pOut++;                                               \
      *pOut = (unsigned char) ((0xFF0000 & V) >> 16);       \
      pOut++;                                               \
      *pOut = (unsigned char) ((0xFF00 & V) >> 8);          \
      pOut++;                                               \
      *pOut = (unsigned char) (0xFF & V);                   \
      pOut++;                                               \
                                                            \
    } else {                                                \
                                                            \
      *pOut = 0xF0;                                         \
      pOut++;                                               \
      *((unsigned int *)pOut) = V;                          \
      pOut += sizeof (unsigned int);                        \
    }


// decodes int from a variable-length prefix code
#define FADecodeIntPrefix(pIn, V)                           \
{                                                           \
    unsigned char S = *(unsigned char *)pIn;                \
                                                            \
    if (0 == 0x80 & S) {                                    \
                                                            \
        V = S;                                              \
        pIn++;                                              \
                                                            \
    } else if (0 == 0x40 & S) {                             \
                                                            \
        V = ((unsigned int)(S & 0x3F)) << 8;                \
        pIn++;                                              \
        V |= *(unsigned char *)pIn;                         \
        pIn++;                                              \
                                                            \
    } else if (0 == 0x20 & S) {                             \
                                                            \
        V = ((unsigned int)(S & 0x1F)) << 16;               \
        pIn++;                                              \
        V |= ((unsigned int)*(unsigned char *)pIn) << 8;    \
        pIn++;                                              \
        V |= *(unsigned char *)pIn;                         \
        pIn++;                                              \
                                                            \
    } else if (0 == 0x10 & S) {                             \
                                                            \
        V = ((unsigned int)(S & 0x0F)) << 24;               \
        pIn++;                                              \
        V |= ((unsigned int)*(unsigned char *)pIn) << 16;   \
        pIn++;                                              \
        V |= ((unsigned int)*(unsigned char *)pIn) << 8;    \
        pIn++;                                              \
        V |= *(unsigned char *)pIn;                         \
        pIn++;                                              \
                                                            \
    } else {                                                \
                                                            \
        pIn++;                                              \
        V = *(unsigned int*)pIn;                            \
        pIn += sizeof (unsigned int);                       \
    }                                                       \
}


// returns the number of bytes necessary to encode V
#define FASizeOfIntPrefix(V, Size)                          \
                                                            \
    if (0x0000007F >= V) {                                  \
        Size = 1;                                           \
    } else if (0x00003FFF >= V) {                           \
        Size = 2;                                           \
    } else if (0x001FFFFF >= V) {                           \
        Size = 3;                                           \
    } else if (0x0FFFFFFF >= V) {                           \
        Size = 4;                                           \
    } else {                                                \
        Size = 5;                                           \
    }


// Recodes 0-byte with two bytes sequence.
//
// Makes the following substitutions:
// 0x00 -> 0x7E
// 0x7E -> 0x7F 0x01
// 0x7F -> 0x7F 0x02
#define FAMaskZeroByte(pOut)                                \
                                                            \
  if (0x00 == *pOut) {                                      \
                                                            \
    *pOut = 0x7E;                                           \
                                                            \
  } else if (0x7E == *pOut) {                               \
                                                            \
    *pOut = 0x7F;                                           \
    pOut++;                                                 \
    *pOut = 0x01;                                           \
                                                            \
  } else if (0x7F == *pOut) {                               \
                                                            \
    *pOut = 0x7F;                                           \
    pOut++;                                                 \
    *pOut = 0x02;                                           \
  }



// 1. Encodes int with variable-length prefix code
// 2. Masks zero-bytes with FAMaskZeroByte(pOut) algorithm
#define FAEncodeIntPrefixMaskZero(V, pOut)                  \
                                                            \
    if (0x0000007F >= V) {                                  \
                                                            \
      *pOut = (unsigned char) V;                            \
      FAMaskZeroByte (pOut);                                \
      pOut++;                                               \
                                                            \
    } else if (0x00003FFF >= V) {                           \
                                                            \
      *pOut = (unsigned char) (0x80 | (V >> 8));            \
      pOut++;                                               \
      *pOut = (unsigned char) (0xFF & V);                   \
      FAMaskZeroByte (pOut);                                \
      pOut++;                                               \
                                                            \
    } else if (0x001FFFFF >= V) {                           \
                                                            \
      *pOut = (unsigned char) (0xC0 | (V >> 16));           \
      pOut++;                                               \
      *pOut = (unsigned char) ((0xFF00 & V) >> 8);          \
      FAMaskZeroByte (pOut);                                \
      pOut++;                                               \
      *pOut = (unsigned char) (0xFF & V);                   \
      FAMaskZeroByte (pOut);                                \
      pOut++;                                               \
                                                            \
    } else if (0x0FFFFFFF >= V) {                           \
                                                            \
      *pOut = (unsigned char) (0xE0 | (V >> 24));           \
      pOut++;                                               \
      *pOut = (unsigned char) ((0xFF0000 & V) >> 16);       \
      FAMaskZeroByte (pOut);                                \
      pOut++;                                               \
      *pOut = (unsigned char) ((0xFF00 & V) >> 8);          \
      FAMaskZeroByte (pOut);                                \
      pOut++;                                               \
      *pOut = (unsigned char) (0xFF & V);                   \
      FAMaskZeroByte (pOut);                                \
      pOut++;                                               \
                                                            \
    } else {                                                \
                                                            \
      *pOut = 0xF0;                                         \
      pOut++;                                               \
      *pOut = (unsigned char) (0xFF & V);                   \
      FAMaskZeroByte (pOut);                                \
      pOut++;                                               \
      *pOut = (unsigned char) ((0xFF00 & V) >> 8);          \
      FAMaskZeroByte (pOut);                                \
      pOut++;                                               \
      *pOut = (unsigned char) ((0xFF0000 & V) >> 16);       \
      FAMaskZeroByte (pOut);                                \
      pOut++;                                               \
      *pOut = (unsigned char) (V >> 24);                    \
      FAMaskZeroByte (pOut);                                \
      pOut++;                                               \
    }


//
// Encodes Value as Char, Short or Int
// Warning: May cause un-aligned access.
//
#define FAEncode_C_S_I(pDump, Offset, Value, SizeOfValue)   \
{                                                           \
    DebugLogAssert (pDump);                                         \
                                                            \
    if (sizeof (char) == SizeOfValue) {                     \
                                                            \
        *(char *)(pDump + Offset) = (char) Value;           \
        Offset++;                                           \
                                                            \
    } else if (sizeof (short) == SizeOfValue) {             \
                                                            \
        *(short *)(pDump + Offset) = (short) Value;         \
        Offset += sizeof (short);                           \
                                                            \
    } else {                                                \
        DebugLogAssert (sizeof (int) == SizeOfValue);               \
                                                            \
        *(int *)(pDump + Offset) = Value;                   \
        Offset += sizeof (int);                             \
    }                                                       \
}

//
// Decodes Char, Short or Int value
// Warning: May cause un-aligned access.
//
#define FADecode_C_S_I(pDump, Offset, Value, SizeOfValue)   \
{                                                           \
    DebugLogAssert (pDump);                                         \
                                                            \
    if (sizeof (char) == SizeOfValue) {                     \
                                                            \
        Value = *(const char *)(pDump + Offset);            \
                                                            \
    } else if (sizeof (short) == SizeOfValue) {             \
                                                            \
        Value = *(const short *)(pDump + Offset);           \
                                                            \
    } else {                                                \
        DebugLogAssert (sizeof (int) == SizeOfValue);               \
                                                            \
        Value = *(const int *)(pDump + Offset);             \
    }                                                       \
}

//
// Encodes Value as Unsigned Char, Short or Int
// Warning: May cause un-aligned access.
//
#define FAEncode_UC_US_UI(pDump, Offset, Value, SizeOfValue) \
{                                                            \
    DebugLogAssert (pDump);                                          \
                                                             \
    if (sizeof (char) == SizeOfValue) {                      \
                                                             \
        *(unsigned char *)(pDump + Offset) =                 \
            (unsigned char) Value;                           \
        Offset += sizeof (char);                             \
                                                             \
    } else if (sizeof (short) == SizeOfValue) {              \
                                                             \
        *(unsigned short *)(pDump + Offset) =                \
            (unsigned short) Value;                          \
        Offset += sizeof (short);                            \
                                                             \
    } else {                                                 \
                                                             \
        DebugLogAssert (sizeof (int) == SizeOfValue);                \
                                                             \
        *(unsigned int *)(pDump + Offset) =                  \
            (unsigned int) Value;                            \
        Offset += sizeof (int);                              \
    }                                                        \
}

//
// Decodes Unsigned Char, Short or Int value
// Warning: May cause un-aligned access.
//
#define FADecode_UC_US_UI(pDump, Offset, Value, SizeOfValue) \
{                                                            \
    DebugLogAssert (pDump);                                          \
                                                             \
    if (sizeof (char) == SizeOfValue) {                      \
                                                             \
        Value = *(const unsigned char *)(pDump + Offset);    \
                                                             \
    } else if (sizeof (short) == SizeOfValue) {              \
                                                             \
        Value = *(const unsigned short *)(pDump + Offset);   \
                                                             \
    } else {                                                 \
                                                             \
        DebugLogAssert (sizeof (int) == SizeOfValue);                \
                                                             \
        Value = *(const unsigned int *)(pDump + Offset);     \
    }                                                        \
}

//
// Encodes Value as three bytes
//
#define FAEncode_3(pDump, Offset, Value)                     \
{                                                            \
    DebugLogAssert (pDump);                                          \
    DebugLogAssert (Value == (0xffffff & Value));                    \
                                                             \
    pDump [Offset++] =                                       \
        (unsigned char) (Value >> 16);                       \
    pDump [Offset++] =                                       \
        (unsigned char) ((0xff00 & Value) >> 8);             \
    pDump [Offset++] =                                       \
        (unsigned char) (0xff & Value);                      \
}

//
// Decodes Value from three bytes
//
#define FADecode_3(pDump, Offset, Value)                     \
{                                                            \
    DebugLogAssert (pDump);                                          \
                                                             \
    const unsigned char hi = pDump [Offset];                 \
    const unsigned char me = pDump [Offset + 1];             \
    const unsigned char lo = pDump [Offset + 2];             \
                                                             \
    Value = (hi << 16) | (me << 8) | lo;                     \
}

//
// Encodes Value as 1, 2, 3 or 4 bytes.
//
#define FAEncode_1_2_3_4(pDump, Offset, Value, SizeOfValue)  \
{                                                            \
    DebugLogAssert (pDump);                                          \
                                                             \
    if (1 == SizeOfValue) {                                  \
                                                             \
        pDump [Offset++] =                                   \
            (unsigned char) (0xff & Value);                  \
                                                             \
    } else if (2 == SizeOfValue) {                           \
                                                             \
        pDump [Offset++] =                                   \
            (unsigned char) ((0xff00 & Value) >> 8);         \
        pDump [Offset++] =                                   \
            (unsigned char) (0xff & Value);                  \
                                                             \
    } else if (3 == SizeOfValue) {                           \
                                                             \
        pDump [Offset++] =                                   \
            (unsigned char) ((0xff0000 & Value) >> 16);      \
        pDump [Offset++] =                                   \
            (unsigned char) ((0xff00 & Value) >> 8);         \
        pDump [Offset++] =                                   \
            (unsigned char) (0xff & Value);                  \
                                                             \
    } else {                                                 \
        DebugLogAssert (4 == SizeOfValue);                           \
                                                             \
        pDump [Offset++] =                                   \
            (unsigned char) (Value >> 24);                   \
        pDump [Offset++] =                                   \
            (unsigned char) ((0xff0000 & Value) >> 16);      \
        pDump [Offset++] =                                   \
            (unsigned char) ((0xff00 & Value) >> 8);         \
        pDump [Offset++] =                                   \
            (unsigned char) (0xff & Value);                  \
    }                                                        \
}

//
// Decodes Value from 1, 2, 3 or 4 bytes.
//
#define FADecode_1_2_3_4(pDump, Offset, Value, SizeOfValue)  \
{                                                            \
    DebugLogAssert (pDump);                                          \
                                                             \
    if (1 == SizeOfValue) {                                  \
                                                             \
        Value = pDump [Offset];                              \
                                                             \
    } else if (2 == SizeOfValue) {                           \
                                                             \
        Value = (pDump [Offset] << 8) |                      \
                (pDump [Offset + 1]);                        \
                                                             \
    } else if (3 == SizeOfValue) {                           \
                                                             \
        Value = (pDump [Offset] << 16) |                     \
                (pDump [Offset + 1] << 8) |                  \
                (pDump [Offset + 2]);                        \
    } else {                                                 \
        DebugLogAssert (4 == SizeOfValue);                           \
                                                             \
        Value = (pDump [Offset] << 24) |                     \
                (pDump [Offset + 1] << 16) |                 \
                (pDump [Offset + 2] << 8) |                  \
                (pDump [Offset + 3]);                        \
    }                                                        \
}

//
// Decodes Value by index from the array of 1, 2, 3 or 4 bytes.
//
#define FADecode_1_2_3_4_idx(pDump, Idx, Value, SizeOfValue) \
{                                                            \
    DebugLogAssert (pDump);                                          \
                                                             \
    if (1 == SizeOfValue) {                                  \
                                                             \
        const int Offset = Idx;                              \
        Value = pDump [Offset];                              \
                                                             \
    } else if (2 == SizeOfValue) {                           \
                                                             \
        const int Offset = 2 * Idx;                          \
        Value = (pDump [Offset] << 8) |                      \
                (pDump [Offset + 1]);                        \
                                                             \
    } else if (3 == SizeOfValue) {                           \
                                                             \
        const int Offset = 3 * Idx;                          \
        Value = (pDump [Offset] << 16) |                     \
                (pDump [Offset + 1] << 8) |                  \
                (pDump [Offset + 2]);                        \
    } else {                                                 \
        DebugLogAssert (4 == SizeOfValue);                           \
                                                             \
        const int Offset = 4 * Idx;                          \
        Value = (pDump [Offset] << 24) |                     \
                (pDump [Offset + 1] << 16) |                 \
                (pDump [Offset + 2] << 8) |                  \
                (pDump [Offset + 3]);                        \
    }                                                        \
}


//
// Decodes destination state by its index.
// Similar to FADecode_1_2_3_4_idx except the check size order is
// 3,4,2,1 and it also takes care of dead states.
//
#define FADecodeDst_idx(pDump, Idx, Value, SizeOfValue)      \
{                                                            \
    DebugLogAssert (pDump);                                          \
                                                             \
    if (3 == SizeOfValue) {                                  \
                                                             \
        const int Offset = 3 * Idx;                          \
        Value = (pDump [Offset] << 16) |                     \
                (pDump [Offset + 1] << 8) |                  \
                (pDump [Offset + 2]);                        \
                                                             \
        if (0x00ffffff == Value) {                           \
            Value = FAFsmConst::DFA_DEAD_STATE;              \
        }                                                    \
    } else if (4 == SizeOfValue) {                           \
                                                             \
        const int Offset = 4 * Idx;                          \
        Value = (pDump [Offset] << 24) |                     \
                (pDump [Offset + 1] << 16) |                 \
                (pDump [Offset + 2] << 8) |                  \
                (pDump [Offset + 3]);                        \
                                                             \
        if (0xffffffff == (unsigned int) Value) {            \
            Value = FAFsmConst::DFA_DEAD_STATE;              \
        }                                                    \
    } else if (2 == SizeOfValue) {                           \
                                                             \
        const int Offset = 2 * Idx;                          \
        Value = (pDump [Offset] << 8) |                      \
                (pDump [Offset + 1]);                        \
                                                             \
        if (0x0000ffff == Value) {                           \
            Value = FAFsmConst::DFA_DEAD_STATE;              \
        }                                                    \
    } else {                                                 \
                                                             \
        DebugLogAssert (1 == SizeOfValue);                           \
                                                             \
        const int Offset = Idx;                              \
        Value = pDump [Offset];                              \
                                                             \
        if (0x000000ff == Value) {                           \
            Value = FAFsmConst::DFA_DEAD_STATE;              \
        }                                                    \
    }                                                        \
}

#endif
