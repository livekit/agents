/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FAUtf32Utils.h"


namespace BlingFire
{

extern const int ** toupper_root [];
extern const int ** tolower_root [];


#define FAUtf32ToUpper_core(Symbol)                                \
    if (0xff >= Symbol) {                                          \
                                                                   \
        if ((Symbol >= 0x61 && Symbol <= 0x7A) ||                  \
            (Symbol >= 0xE0 && Symbol <= 0xFE)) {                  \
                                                                   \
            Symbol -= 0x20;                                        \
                                                                   \
        } else if (Symbol == 0xB5) {                               \
                                                                   \
            Symbol = 0x39C;                                        \
       }                                                           \
    } else if (0x1ffff >= Symbol) {                                \
                                                                   \
        DebugLogAssert (toupper_root);                             \
                                                                   \
        const int hi = (Symbol & 0xff0000) >> 16;                  \
        const int me = (Symbol & 0xff00) >> 8;                     \
        const int lo = Symbol & 0xff;                              \
                                                                   \
        const int OutSymbol = ((toupper_root [hi])[me])[lo];       \
                                                                   \
        if (-1 != OutSymbol)                                       \
            Symbol = OutSymbol;                                    \
    }


#define FAUtf32ToLower_core(Symbol)                                \
    if (0xff >= Symbol) {                                          \
                                                                   \
        if ((Symbol >= 0x41 && Symbol <= 0x5A) ||                  \
            (Symbol >= 0xC0 && Symbol <= 0xDE)) {                  \
                                                                   \
            Symbol += 0x20;                                        \
        }                                                          \
    } else if (0x1ffff >= Symbol) {                                \
                                                                   \
        DebugLogAssert (tolower_root);                             \
                                                                   \
        const int hi = (Symbol & 0xff0000) >> 16;                  \
        const int me = (Symbol & 0xff00) >> 8;                     \
        const int lo = Symbol & 0xff;                              \
                                                                   \
        const int OutSymbol = ((tolower_root [hi])[me])[lo];       \
                                                                   \
        if (-1 != OutSymbol)                                       \
            Symbol = OutSymbol;                                    \
    }


const int FAUtf32ToUpper (const int Symbol)
{
    int ResSymbol = Symbol;
    FAUtf32ToUpper_core (ResSymbol);
    return ResSymbol;
}


const int FAUtf32ToLower (const int Symbol)
{
    int ResSymbol = Symbol;
    FAUtf32ToLower_core (ResSymbol);
    return ResSymbol;
}


void FAUtf32StrUpper (__out_ecount (Size) int * pChain, const int Size)
{
    if (pChain) {

        for (int i = 0; i < Size; ++i) {

            int Symbol = pChain [i];
            FAUtf32ToUpper_core (Symbol);
            pChain [i] = Symbol;
        }
    }
}


void FAUtf32StrLower (__out_ecount (Size) int * pChain, const int Size)
{
    if (pChain) {

        for (int i = 0; i < Size; ++i) {

            int Symbol = pChain [i];
            FAUtf32ToLower_core (Symbol);
            pChain [i] = Symbol;
        }
    }
}


const bool FAUtf32IsLower (const int Symbol)
{
    DebugLogAssert (toupper_root);

    // faster processing for the first 0xFF symbols
    if (0xff >= Symbol) {

        if ((Symbol >= 0x61 && Symbol <= 0x7A) || \
            (Symbol >= 0xE0 && Symbol <= 0xFE))
            return true;

        // MICRO SIGN
        if (Symbol == 0xB5)
            return true;

        return false;

    // processing the rest of 0x1FFFF symbols
    } else if (0x1ffff >= Symbol) {

        const int hi = (Symbol & 0xff0000) >> 16;
        const int me = (Symbol & 0xff00) >> 8;
        const int lo = Symbol & 0xff;

        const int OutSymbol = ((toupper_root [hi])[me])[lo];
        return -1 != OutSymbol;

    // no case-sensitive symbols here
    } else {

        return false;
    }
}


const bool FAUtf32IsUpper (const int Symbol)
{
    DebugLogAssert (tolower_root);

    // faster processing for the first 0xFF symbols
    if (0xff >= Symbol) {

        if ((Symbol >= 0x41 && Symbol <= 0x5A) || \
            (Symbol >= 0xC0 && Symbol <= 0xDE))
            return true;

        return false;

    // processing the rest of 0x1FFFF symbols
    } else if (0x1ffff >= Symbol) {

        const int hi = (Symbol & 0xff0000) >> 16;
        const int me = (Symbol & 0xff00) >> 8;
        const int lo = Symbol & 0xff;

        const int OutSymbol = ((tolower_root [hi])[me])[lo];
        return -1 != OutSymbol;

    // no case-sensitive symbols here
    } else {

        return false;
    }
}

}
