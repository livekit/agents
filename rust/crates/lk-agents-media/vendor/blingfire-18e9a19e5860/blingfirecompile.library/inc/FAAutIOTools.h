/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_AUT_IO_TOOLS_H_
#define _FA_AUT_IO_TOOLS_H_

#include "FAConfig.h"

#include <iostream>

namespace BlingFire
{

class FAAllocatorA;
class FARSNfaA;
class FARSDfaA;
class FAState2OwA;
class FAState2OwsA;
class FAMealyNfaA;
class FAMealyDfaA;

///
/// A set of methods for printing out and reading in different kinds of 
/// automata in ascii representation.
///
/// Notes:
///   Read (...) - reads from textual representation
///   Print (...) - saves in textual representation
///

class FAAutIOTools {

public:
    FAAutIOTools (FAAllocatorA * pAlloc);

/// tools for FARSNfaA
public:
  void Read (std::istream& is, FARSNfaA * pNFA);
  void Print (std::ostream& os, const FARSNfaA * pNFA);

/// tools for Mealy NFA
public:
  void Read (std::istream& is, FARSNfaA * pNFA, FAMealyNfaA * pOws);
  void Print (std::ostream& os, const FARSNfaA * pNFA, const FAMealyNfaA * pOws);

/// tools for FARSDfaA
public:
  void Read (std::istream& is, FARSDfaA * pDFA);
  void Print (std::ostream& os, const FARSDfaA * pDFA);

/// tools for Mealy DFA
public:
  void Read (std::istream& is, FARSDfaA * pDFA, FAMealyDfaA * pOws);
  void Print (std::ostream& os, const FARSDfaA * pDFA, const FAMealyDfaA * pOws);

/// tools for deterministic Moore machines with single output
public:
  void Read (std::istream& is, FARSDfaA * pDFA, FAState2OwA * pOwMap);
  void Print (std::ostream& os, const FARSDfaA * pDFA, const FAState2OwA * pOwMap);

/// tools for deterministic Moore machines with multiple outputs
public:
  void Read (std::istream& is, FARSDfaA * pDFA, FAState2OwsA * pOwsMap);
  void Print (std::ostream& os, const FARSDfaA * pDFA, const FAState2OwsA * pOwsMap);

private:

  void PrintHeader (std::ostream& os, const FARSNfaA * pNFA);
  void PrintHeader (std::ostream& os, const FARSDfaA * pDFA);

  void PrintNfaCommon (std::ostream& os, const FARSNfaA * pNFA, const FAMealyNfaA * pOws);
  void ReadNfaCommon (std::istream& is, FARSNfaA * pNFA, FAMealyNfaA * pOws);

  void PrintDfaCommon (std::ostream& os, const FARSDfaA * pDFA, const FAMealyDfaA * pOws);
  void ReadDfaCommon (std::istream& is, FARSDfaA * pDFA, FAMealyDfaA * pOws);

private:
    FAAllocatorA * m_pAlloc;
};

}

#endif
