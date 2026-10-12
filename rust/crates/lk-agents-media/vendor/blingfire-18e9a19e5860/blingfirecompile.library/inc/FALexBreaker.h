/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_FALEXBREAKER_H_
#define _FA_FALEXBREAKER_H_

#include "FALexTools_t.h"
#include "FAArray_cont_t.h"
#include "FATagSet.h"
#include "FALimits.h"

namespace BlingFire
{

///
/// fa_lex based word-breaker
///
/// Notes: 
///
/// 1. The word breaker allocates from the pool a buffer of size 20*cbUTF8Text 
///    bytes, if the cbUTF8Text < MaxBuffSize else it uses a constant size 
///    buffer of 20*MaxBuffSize bytes.
/// 2. Output words are limited to have no more than MaxWordLen symbols.
///

class FALexBreaker
{
public:
    FALexBreaker ();
    ~FALexBreaker ();

    /// Initialize the wordbreaker before using it. 
    /// Provide an initialized configuration object.
    /// Extra initializations are ignored, returns ERROR_SUCCESS upon success
    bool Initialize (const FAWbdConfKeeper * pConfig);

    // De-Initializes the wordbreaker
    // De-initializing an un-initialized object is ignored
    // Call this method if you want to reinitialize the word-breaker
    void DeInitialize ();

    /// specifies the input should be treated as a sequence of bytes,
    /// by default the input is a UTF-8 string
    void SetUseBytes (const bool fUseBytes);
    /// if set, the word tags are returned
    void SetTagSet (const FATagSet * pTagSet);
    /// if set, the surface word-forms are returned
    void SetOutText (FAArray_cont_t < char > * pOutText);
    /// if set, the normalized word-forms are returned
    void SetOutNormText (FAArray_cont_t < char > * pOutNormText);

    /// Breaks up the given buffer of text into words according to the compiled
    /// fa_lex grammar. Each discovered word is copied to the output object, 
    /// together with offsets in the input buffer. The output object will be 
    /// initialized inside the method, old entries will be retained returns 
    /// ERROR_SUCCESS upon success
    bool BreakText (const UInt8 *pbUTF8Text, size_t cbUTF8Text);

private:
    // internal
    bool BreakText_int (const UInt8 *pbUTF8Text, size_t cbUTF8Text) ;
    // returns object into the initial state
    void Clear ();
    // m_Txt buffer filled in, the function fills in m_Tokens buffer
    void ProcessTxt ();
    // m_Tokens buffer filled in the function iterates thru the tokens
    // in the right order and feeds them to the PutToken
    void ProcessTokens () const;
    // helper
    inline void AddWord (
            FAArray_cont_t < char > * pOutText, 
            const UInt8 * pWord, 
            const size_t Length, 
            const int Tag
        ) const;
    // adds a token with a proper normalization into the FAArray_cont_t < char >
    inline void PutToken (
            const int Tag,            // token tag
            const int TxtFrom,        // UTF-32LE token offset from
            int TxtTo,                // UTF-32LE token offset to
            const int * pTxt = NULL,  // UTF-32LE normalized token text
            int TxtLen = 0            // UTF-32LE normalized token length
        ) const;

private:
    /// predefined constants
    enum {
        MaxWordLen = FALimits::MaxWordLen,
        MaxBuffSize = 0x100000,
    };

    /// indicates whether this object was initialized
    bool m_fInitialized;
    bool m_fUseBytes;
    int m_iXWordTag;
    int m_iSegTag;
    int m_iIgnoreTag;
    /// input
    const UInt8 * m_pbUTF8Text;
    size_t m_cbUTF8Text;
    const FATagSet * m_pTagSet;
    /// output
    FAArray_cont_t < char > * m_pOutText;
    FAArray_cont_t < char > * m_pOutNormText;
    /// the tokenizer
    FALexTools_t < int > m_wbd;
    /// UTF-32LE normalization map, changes length for some characters
    const FAMultiMapCA * m_pNorm;
    /// current Offset in the input string
    size_t m_uUtf8Offset;
    /// current offset in the m_Txt array
    unsigned int m_uTxtOffset;
    /// temporary fixed buffer: one piece of text in UTF-32
    int m_pTxt [MaxBuffSize];
    int m_pOffsets [MaxBuffSize];
    unsigned int m_uTxtLen;
    /// temporary fixed buffer: a set of tokens extracted from one line
    int m_pTokens [MaxBuffSize * 3];
    int m_iTokensSize;

    // amount of characters which can be processed at once
    unsigned int m_uMaxCharsAtOnce;
    // amount of tokens which can be processed at once
    unsigned int m_uMaxTokensSize;
};

}

#endif
