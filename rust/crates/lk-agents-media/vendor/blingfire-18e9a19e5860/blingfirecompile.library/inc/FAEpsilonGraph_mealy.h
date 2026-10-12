/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_EPSILONGRAPH_MEALY_H_
#define _FA_EPSILONGRAPH_MEALY_H_

#include "FAConfig.h"
#include "FAEpsilonGraph.h"
#include "FAChain2Num_hash.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAMealyNfaA;

///
/// This class is used at epsilon removal stage for Mealy NFA.
///
/// Note: 
/// 1. if epsilon path contain Ows along it then the concatenation of Ows
///    will be assigned as an Ow corresponding to that path.
/// 2. epsilon graph must be acyclic
///

class FAEpsilonGraph_mealy : public FAEpsilonGraph {

public:
    FAEpsilonGraph_mealy (FAAllocatorA * pAlloc);

public:
    // will read take Arc -> Ow information from here
    void SetInMealy (const FAMealyNfaA * pInMealy);
    // will store Arc -> [ Ows ]_i information here
    void SetOutMealy (FAMealyNfaA * pOutMealy);
    // overloaded from FAEpsilonGraph in order to update Node -> [ Ows ] mapping
    void SetDstNodes (
            const int Node, 
            const int * pDstNodes, 
            const int Size
        );
    // returns object into initial state
    void Clear ();
    // returns [Ows] <-> Ow mapping
    const FAChain2NumA * GetOwsMap () const;

private:
    inline void UpdateOws (
            const int Src,
            const int Iw,
            const int Dst,
            const int Ow,
            const int RightOws
        );

private:
    /// input/output
    const FAMealyNfaA * m_pInMealy;
    FAMealyNfaA * m_pOutMealy;
    /// keeps mapping for multiple output weights
    FAChain2Num_hash m_Ows2Id;
    /// temporary chain container
    FAArray_cont_t < int > m_chain;

};

}

#endif
