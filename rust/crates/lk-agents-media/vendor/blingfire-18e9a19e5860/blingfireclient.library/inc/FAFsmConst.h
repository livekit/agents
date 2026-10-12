/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_FSM_CONST_H_
#define _FA_FSM_CONST_H_

#include "FAConfig.h"

namespace BlingFire
{

class FAFsmConst {

public:

    // type of data structure
    enum {
        TYPE_RS_NFA = 0,
        TYPE_POS_RS_NFA,
        TYPE_RS_DFA,
        TYPE_MOORE_DFA,
        TYPE_MOORE_MULTI_DFA,
        TYPE_MULTI_MAP,
        TYPE_MEALY_NFA,
        TYPE_MEALY_DFA,
        TYPE_ARRAY,
        TYPE_FLOAT_ARRAY,
        TYPE_STRING_ARRAY,
        TYPE_COUNT,
    };

    // automaton or mmap container type
    enum {
        MODE_INT = 0,      // general int-based container
        MODE_PACK_TRIV,    // trivial packed container
        MODE_PACK_MPH,     // MPH-based packed container
        MODE_PACK_FIXED,   // fixed size array based representation
        MODE_COUNT,
    };

    // modes of reaction map in-memory representations
    enum {
        OWS_MODE_TRIV = 0,
        OWS_MODE_UNIQ,
        OWS_MODE_DUMP,
        OWS_MODE_COUNT,
    };

    // types of on-disk representations
    enum {
        FORMAT_TXT = 0,    // ASCII text
        FORMAT_DUMP,       // contiguous memory dump
        FORMAT_COUNT,
    };

    // specifies different types of labels of regular expressions
    enum {
        LABEL_DIGIT = 0,   // uint-based regular expressions
        LABEL_CHAR,        // character based regular expressions
        LABEL_WRE,         // WRE-token based regular expressions
        LABEL_COUNT,
    };

    // default values for some common input weights
    enum {
        IW_ANY = 0,         // /./
        IW_L_ANCHOR = 1,    // /^/
        IW_R_ANCHOR = 2,    // /$/
        IW_EPSILON = 3,     // epsilon
        IW_EOS = 4,         // end-of-sequence
        IW_COUNT,
    };

    // one for all Dead-State value
    enum {
        NFA_DEAD_STATE = -1,
        DFA_DEAD_STATE = -2, // as -1 indicates the absence of transition
    };

    /// types of digitizers
    enum {
        DIGITIZER_TEXT = 0,
        DIGITIZER_TAGS,
        DIGITIZER_DCTS,
        DIGITIZER_COUNT,
    };

    // WRE token type bit-masks
    enum {
        WRE_TT_TEXT = 1,
        WRE_TT_TAGS = 2,
        WRE_TT_DCTS = 4,
        WRE_TT_DEFAULT = WRE_TT_TEXT | WRE_TT_TAGS | WRE_TT_DCTS,
    };

    // types of WRE rules
    enum {
        WRE_TYPE_RS = 0,   // accepting rules
        WRE_TYPE_MOORE,    // classifying rules
        WRE_TYPE_MEALY,    // extracting rules
        WRE_TYPE_COUNT,
    };

    // packed WRE configuration constants
    enum {
        WRE_CONF_WRE_TYPE = 0, // WRE-type index
        WRE_CONF_TOKEN_TYPE,   // token-type index
        WRE_CONF_TAG_OW_BASE,  // ...
        WRE_CONF_TXT_DIG,      // Txt digitizer offset index
        WRE_CONF_DCT_DIG,      // Dxt digitizer offset index
        WRE_CONF_FSM1,         // ...
        WRE_CONF_FSM2,
        WRE_CONF_TRBR,
        WRE_CONF_COUNT,        // configuration size
    };

    // types of input chain procesing directions
    enum {
        DIR_L2R = 0,       // left to right
        DIR_R2L,           // right to left
        DIR_AFF,           // affix: last, first, last - 1, first + 1, ...
        DIR_COUNT,
    };

    // transformation types
    enum {
        TR_HYPH_REDUP = 0, // hyphenated reduplication
        TR_HYPH_REDUP_REV, // reversed hyphenated reduplication
        TR_PREFIX,         // puts recognized prefixes to the end of the word
        TR_PREFIX_REV,     // reverse prefix transformation
        TR_UCF,            // upper-case-first transformation
        TR_UCF_REV,        // reverse upper-case-first transformation
        TR_COUNT,
    };

    // types of automaton interpreters
    enum {
        INT_TRIV = 0,      // trivial
        INT_FNFA,          // factor NFA based
        INT_POS,           // match positions reconstruction
        INT_TRBR,          // triangular bracket extraction
        INT_WG,            // word-guesser
        INT_SUBST,         // substituter
        INT_SUFFIX,        // suffix rules interpreter
        INT_COUNT,
    };

    // kinds of functions
    enum {
        FUNC_W2T = 0,  // returns tags for the given word-form
        FUNC_W2B,      // returns base-forms from a word-form
        FUNC_B2W,      // returns word-forms from a base-form
        FUNC_W2W,      // returns word-forms from a word-form
        FUNC_TRS,      // transformation of FATransformCA_t < Ty > type
        FUNC_W2S,      // makes segmentation of the input word
        FUNC_WRE,      // WRE syntax rules
        FUNC_WT2B,     // returns base-forms for the word with the given tag
        FUNC_B2WT,     // returns word-forms with the given tag from the base
        FUNC_WTT2W,    // a superposition of FUNC_B2WT (FUNC_WT2B (w))
        FUNC_TAG_DICT, // tag-dictionary function
        FUNC_W2H,      // word hyphenation function
        FUNC_POS_DICT, // POS tagging (Tag/Prob) dictionary function
        FUNC_B2T,      // returns all word-form tags for the given base-form
        FUNC_T2TB,     // for the given tag returns the base-form tag(s)
        FUNC_TB2T,     // for the given base-form tag returns word-form tags
        FUNC_W2TP,     // by the given word returns tags and p(t|w) values
        FUNC_W2TPL,    // by the following word returns tags and p(t|w-1) values
        FUNC_W2TPR,    // by the next word returns tags and p(t|w+1) values
        FUNC_WBD,      // builds rules and actions for word boundary detection
        FUNC_GLOBAL,   // global client specific configuration
        FUNC_W2H_ALT,  // additional alternative word hyphenation function
        FUNC_T2P,      // returns a probability of a tag, P(T)
        FUNC_TT2P,     // returns a probability of a tag, P(T|T-1)
        FUNC_TTT2P,    // returns a probability of a tag, P(T|T-2,T-1)
        FUNC_NORM_RULES,// concatenation rules for NE normalization
        FUNC_NORM_DICT,// normalization dictionary
        FUNC_EMIT,     // NE emission rules
        FUNC_OIC_RULES,// offensive in context WRE rules
        FUNC_CSS_RULES,// CSS suggestion WRE rules
        FUNC_W2V,      // returns spelling variants for a word-form
        FUNC_W2P,      // returns word's probability p(w)
        FUNC_N2TP,     // by the given ngram returns tags and p(t|ng) values
        FUNC_LAD,      // Language Auto-Detection (LAD)
        FUNC_U2L,      // returns a set of languages and possibly scores for the given url
        FUNC_I2W,      // index to word conversion data
        FUNC_COUNT,
    };

    // parameter names for run-time containers
    enum {
        PARAM_IN_TR = 0,   // Input transformation type
        PARAM_OUT_TR,      // Output transformation type
        PARAM_FSM,         // input automaton (type is clear from context)
        PARAM_RSDFA,       // RS DFA
        PARAM_STATE2OW,    // Reaction
        PARAM_STATE2OWS,   // Multiple reaction
        PARAM_ACTS,        // Rule -> action map
        PARAM_FOLLOW,      // Following positions map
        PARAM_POS2BEGINBR, // Position -> Begining Bracket map
        PARAM_POS2ENDBR,   // Position -> Ending Bracket map
        PARAM_REVERSE,     // Reverse input
        PARAM_DIRECTION,   // Input direction
        PARAM_TRIM,        // Trimming value for word-guesser
        PARAM_REDUP_DELIM, // Reduplication delimiter
        PARAM_PREF_DELIM,  // Prefix delimiter
        PARAM_PREF_FSM,    // Prefix automaton
        PARAM_MAP_MODE,    // Container type for the action or some other map
        PARAM_MIN_LEN,     // smallest length, e.g. compound, pattern, etc.
        PARAM_NO_TR,       // Forbid transformation, even if it is specified
        PARAM_UCF_DELIM,   // upper-case-first transformation delimiter
        PARAM_TOKEN_TYPE,  // WRE token type
        PARAM_FSM_COUNT,   // FSM count per rule
        PARAM_IGNORE_CASE, // process input in the lower case
        PARAM_DEFAULT_TAG, // default tag value
        PARAM_ARRAY,       // Packed Array, for example: word id -> set id
        PARAM_MULTI_MAP,   // Multi-Map, for example: set id -> set
        PARAM_FSM_TYPE,    // specifies the type of FSM (if needed)
        PARAM_LEFT_ANCHOR, // special left Iw
        PARAM_RIGHT_ANCHOR,// special right Iw
        PARAM_TYPE,        // structure type (WRE type for example)
        PARAM_TAG_OW_BASE, // base value for the tag Ow (WRE specific)
        PARAM_DICT_MODE,   // classifier is working as a dictionary
        PARAM_MIN_LEN2,    // other than PARAM_MIN_LEN minimal length
        PARAM_MIN_LEN3,    // other than PARAM_MIN_LEN(2) minimal length
        PARAM_HYPH_TYPE,   // defines hyphenation algorithm to be used
        PARAM_NORMALIZE,   // inidicates whether normalization should be used
        PARAM_MAX_PROB,    // the numrical value corresponding to 1 of the prob
        PARAM_DO_W2B,      // do reductive stemming, e.g. at word-breaking
        PARAM_DEPTH,       // depth, e.g. recursion depth ...
        PARAM_MAX_TAG,     // maximum tag value
        PARAM_LOG_SCALE,   // the log scale is being used
        PARAM_FLOAT_ARRAY, // array of floating point values
        PARAM_WORD,        // word token tag
        PARAM_PUNKT,       // punktuation tags
        PARAM_EOS,         // end of sequence tag
        PARAM_EOP,         // end of paragraph tag
        PARAM_USE_NFST,    // indicates that NFST should be used
        PARAM_CHARMAP,     // character map
        PARAM_WRE_CONF,    // compiled WRE configuration dump
        PARAM_SUFFIX_FSM,  // suffix automaton 
        PARAM_MIN_UNI_PROB,// minimal unigram probability in %
        PARAM_XWORD,       // complex token tag
        PARAM_SEG,         // segment tag
        PARAM_IGNORE,      // ignore tag
        PARAM_ORDER,       // n-grams order
        PARAM_MIN_ORDER,   // n-grams min backoff order
        PARAM_UNKNOWN,     // UNKNOWN tag
        PARAM_MAX_COUNT,   // maximum count value
        PARAM_RATIO,       // ratio in %
        PARAM_RATIO2,      // ratio in %
        PARAM_C2S_MAP,     // character --> script map
        PARAM_S2L_MAP,     // script --> language map
        PARAM_SCRIPT_MIN,  // smallest tag value for scripts
        PARAM_SCRIPT_MAX,  // biggest tag value for scripts
        PARAM_MAX_DISTANCE,// maximum distance value
        PARAM_MAX_PASS_COUNT,// maximum pass count
        PARAM_MAX_SCORE,   // override score
        PARAM_THRESHOLD,   // any threshold
        PARAM_ACT_DATA,    // action data map
        PARAM_MAX_LENGTH,  // maximum length, e.g. maximum token length
        PARAM_VERIFY_LDB_BIN, // if specified, requires a CRC32-like check for the LDB file to pass
        PARAM_TOKENIZATION_TYPE, // specifies which tokenization runtime should be used
        PARAM_ID_OFFSET,   // specifies the integer value to be added to all output IDs (used in Bling Fire tokenizer)
        PARAM_USE_BYTE_ENCODING, // specifies if input characters are UTF-8 bytes, not Unicode symbols
        PARAM_NO_DUMMY_PREFIX,  // for unigram-lm and BPE, if specified then no dummy whitespace is added in the beginning of text
        PARAM_STRING_ARRAY,// string array dump index
        PARAM_TOKENID_MIN, // if specified provides smalles regular (non special) token id value
        PARAM_TOKENID_MAX, // if specified provides biggest regular (non special) token id value
        PARAM_COUNT,
    };

    // parser type
    enum {
        PARSER_TRIV = 0,   // refers to FAParser_triv_t interpreter
        PARSER_NEST = 1,   // refers to FAParser_nest_t interpreter
        PARSER_WRE_LEX = 2,// fa_lex-like WRE rules
    };

    // parser type
    enum {
        RESOLVE_MATCH_ALL  = 0,   // accepts all match results
        RESOLVE_MATCH_NEST = 1,   // removes overlappings and same rule nested
    };

    // length 2 trbr mapping
    enum {
        TRBR_LEFT = -1,    // indicates left triangular bracket
        TRBR_RIGHT = -2,   // indicates right triangular bracket
    };

    // special characters for corpus IO
    enum {
        CHAR_WORD_DELIM = ' ',   // words delimiter for corpus tools
        CHAR_TAG_DELIM = '/',    // word/tag delimiter for tagged word
        CHAR_MWE_DELIM = '_',    // simple words delimiter within MWE
        CHAR_SPACE = ' ',        // normal space for MWEs
    };

    // dictionary data mode type
    enum {
        DM_RAW = 0,      // KEY -> RAW DATA  ; no duplicate words
        DM_TAGS,         // KEY -> TAGS      ; input sorted by KEY
        DM_TAG_PROB,     // KEY -> TAG, PROB ; input sorted by KEY
        DM_HYPH,         // KEY -> FREQ, OWS ; input sorted by KEY
        DM_COUNT,
    };

    // statistics type masks
    enum {
        STAT_TYPE_NONE = 0,   // nothing
        STAT_TYPE_W = 1,      // the word
        STAT_TYPE_WT = 2,     // the word and tag
        STAT_TYPE_WTT = 4,    // the word, tag and the following tag
        STAT_TYPE_TWT = 8,    // the word, tag and the preceding tag
        STAT_TYPE_WTWT = 16,  // the word, tag and the following word, tag
        STAT_TYPE_T = 32,     // the tag
        STAT_TYPE_TT = 64,    // the tag bigram
        STAT_TYPE_TTT = 128,  // the tag trigram
        STAT_TYPE_TTTT = 256, // the tag fourgram
        STAT_TYPE_WW = 512,   // the word bigram
        STAT_TYPE_WWW = 1024, // the word trigram
        STAT_TYPE_W_T = 2048, // the word and the following tag
        STAT_TYPE_TW = 4096,  // the word and the preceding tag
        STAT_TYPE_DEFAULT = STAT_TYPE_WT,
    };

    // default POS tag value
    enum {
        POS_TAG_DEFAULT = 1,
    };

    // describes types of hyphneation
    enum {
        HYPH_TYPE_CORE = 0, // just a pattern based hyphneation
        HYPH_TYPE_W2H_W2S,  // independent W2H and W2S
        HYPH_TYPE_W2S_W2H,  // W2S and W2H for each segment
        HYPH_TYPE_COUNT,
        HYPH_TYPE_DEFAULT = HYPH_TYPE_CORE,
    };

    // hyphenation procedfures names (other modifications are not allowed)
    enum {
        HYPH_CONFLICT = -2,   // conflict between two or more patterns
        HYPH_UNKNOWN = -1,    // uncovered position of text
        HYPH_NO_HYPH = 0,     // no-hyphenation point
        HYPH_SIMPLE_HYPH = 1, // a simple hyphenation
        HYPH_ADD_BEFORE,      // add letter before hyphen
        HYPH_CHANGE_BEFORE,   // change letter before hyphen
        HYPH_DELETE_BEFORE,   // delete letter before hyphen
        HYPH_CHANGE_AFTER,    // change letter after hyphen
        HYPH_DEL_AND_CHANGE,  // delete letter before and change after hyphen
        HYPH_DONT_CARE,
        HYPH_COUNT,
    };

    enum {
        MIN_LOG_PROB = -80,
        MAX_LOG_PROB = 0,
    };

    /// bit masks for state's transitions representation
    enum {
        TRS_NONE = 0x00, // no Dsts
        TRS_IMPL = 0x02, // Dst = Src + |Src|
        TRS_PARA = 0x04, // two parallel arrays, Iws is sorted
        TRS_IWIA = 0x06, // Iws indexed array of Dsts, Dst == Dsts[Iw - Base]
        TRS_RANGE = 0x01, // Iws ranges, each of which corresponds to one Dst
    };

    /// word/sequence case types
    enum {
        CASE_ALL_LOWER = 0, // all letters are lower case only
        CASE_CAPITALIZED,   // first is capital the rest if any are lower only
        CASE_ALL_UPPER,     // all more than one letters are upper case only
        CASE_OTHER,         // other cases (mixed case or words with numbers)
    };

    // defaults in packed representation
    enum {
        TRIV_PACK_DEF_DST_SIZE = 3, // default dst size for triv packed
    };

    // LDB bin validation
    enum {
        VALIDATION_VERSION = 0,
        VALIDATION_SIZE,
        VALIDATION_HASH,
        VALIDATION_COUNT,
    };

    // character normalization method
    enum {
        NORMALIZE_DEFAULT = 0,
        NORMALIZE_PRESERVE_DIACRITICS = 1,
        NORMALIZE_REMOVE_DIACRITICS = 2,
        NORMALIZE_COUNT,
    };

    // segmentation model types
    enum {
        TOKENIZE_DEFAULT = 0,
        TOKENIZE_WORDPIECE = 1,
        TOKENIZE_UNIGRAM_LM = 2,
        TOKENIZE_BPE = 3,
        TOKENIZE_BPE_OPT = 4,     // optimized version of the BPE, prefers a single token match over
                                  //  subtoken, assumes tokens are delimited with U+x2581 

        TOKENIZE_BPE_OPT_WITH_MERGES = 5, // optimized version of the BPE, prefers a single token match over
                                          // subtoken, assumes tokens are delimited with U+x2581 uses scores 
                                          // as merge ranks
        TOKENIZE_COUNT,
    };

};

}

#endif
