/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FAMsg.h"

namespace BlingFire
{

const char * const FAMsg::InternalError = \
    "Internal error.";

const char * const FAMsg::InitializationError = \
    "Object cannot be initialized.";

const char * const FAMsg::InvalidParameters = \
    "Invalid parameters.";

const char * const FAMsg::IndexIsOutOfRange = \
    "Index is out of range.";

const char * const FAMsg::LimitIsExceeded = \
    "Limit size has been exceeded.";

const char * const FAMsg::OutOfMemory = \
    "Out of memory.";

const char * const FAMsg::IOError = \
    "Input/Output error.";

const char * const FAMsg::ReadError = \
    "Can't read from file.";

const char * const FAMsg::WriteError = \
    "Can't write to file.";

const char * const FAMsg::CorruptFile = \
    "Invalid or corrupt file.";

const char * const FAMsg::ResourceIsNotFound = \
    "Resource is not found.";

const char * const FAMsg::ObjectIsNotReady = \
    "Object is not ready.";

const char * const FAMsg::SyntaxError = \
    "Syntax error.";

}
