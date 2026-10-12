/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_WREIO_H_
#define _FA_WREIO_H_

#include "FAConfig.h"

#include <iostream>

namespace BlingFire
{

class FAWREConfA;

///
/// Saves compiled WRE in a textual format
///
void FAPrintWre (std::ostream & os, const FAWREConfA * pWRE);

///
/// Reads compiled WRE from a textual format
///
void FAReadWre (std::istream & is, FAWREConfA * pWRE);

///
/// Builds and saves WRE memory dump
///
void FASaveWre (std::ostream & os, const FAWREConfA * pWRE);

}

#endif
