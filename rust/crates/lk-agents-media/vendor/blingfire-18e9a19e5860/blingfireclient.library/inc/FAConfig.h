/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_CONFIG_H_
#define _FA_CONFIG_H_

#ifndef NEW
#define NEW new
#endif

#if _MSC_VER < 1400
# ifndef fopen_s
#   define fopen_s(ppF,pN,pA) int(0); *ppF = fopen(pN,pA);
# endif
# ifndef _set_fmode
#   define _set_fmode(X) _fmode = X;
# endif
#endif

#include <ctype.h>
#include <fcntl.h>

#ifdef BLING_FIRE_MAC
  #include <string>
#else
  #include <malloc.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <limits.h>
#include <stdint.h>
#include <stddef.h>

#ifndef BLING_FIRE_NOWINDOWS

/// Windows specific
#include <io.h>
#include <windows.h>
#include <process.h>

#else

/// non-Windows specific
#include <unistd.h>

#ifndef _getpid
#define _getpid getpid
#endif

#ifndef _unlink
#define _unlink unlink
#endif

#ifndef __cdecl
#define __cdecl __attribute__((__cdecl__))
#endif

#ifndef ULONG
#define ULONG unsigned long
#endif

#ifndef LONG
#define LONG long
#endif

#ifndef __stdcall
#define __stdcall
#endif

#ifndef HANDLE
#define HANDLE void*
#endif

#include <stdint.h>

/* because we have stdint.h 
#ifndef __int32
#define __int32 int32_t
#endif

#ifndef __int16
#define __int16 int16_t
#endif

#ifndef uint16_t
#define uint16_t u_int16_t
#endif

#ifndef uint32_t
#define uint32_t u_int32_t
#endif

#ifndef uint8_t
#define uint8_t u_int8_t
#endif
*/


#ifndef __fallthrough
#define __fallthrough
#endif

#endif


#ifndef PATH_MAX
#define PATH_MAX 32000
#endif


// enable this to track the memory allocations and leaks in BlingFire
/// #define _DEBUG_MEMORY


// no APSDK or Memlfp dependecies
#ifdef BLING_FIRE_NOAP

#ifndef UInt8
#define UInt8 uint8_t
#endif

#undef LogAssert
#undef DebugLogAssert

// Disable conditional expr constant warning
#pragma warning(disable : 4127)

// LogAssert
#include <stdexcept>

#define LogAssert(exp, ...) \
    do { \
        if (!(exp)) { \
            char szMessage[1024]; \
            snprintf(szMessage, sizeof(szMessage), \
                "%s, %d: assertion failed: %s\n", __FILE__, __LINE__, #exp); \
            throw std::runtime_error(szMessage); \
        } \
    } while (0)


// DebugLogAssert
#include <assert.h>
#define DebugLogAssert assert

#else

#include "LogAssert.h"

#endif // of #ifdef BLING_FIRE_NOAP


#endif
