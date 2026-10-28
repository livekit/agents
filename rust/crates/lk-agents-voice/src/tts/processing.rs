//! The speech transforms supplied to native provider clients and sentence adapters.

use lk_agents_abi::tts::stream_adapter::{SentenceSplitter, SentenceTextProcessor};
use lk_agents_abi::tts::text::{TranscriptProcessor, TtsTextProcessorFactory};

use crate::tokenize::blingfire::{SentenceBuffer, SentenceBufferOptions};

use super::markup::{sentence_tokenizer_of, text_processor};
use super::transcript::AudioTranscriptNormalizer;

#[derive(Clone)]
pub struct SpeechText;

impl TtsTextProcessorFactory for SpeechText {
    fn sentence_splitter(provider: Option<&str>, expressive: bool) -> Box<dyn SentenceSplitter> {
        let opts = match provider {
            Some(provider) => {
                let (max_token_len, min_token_len, xml_aware) =
                    sentence_tokenizer_of(provider, expressive);
                SentenceBufferOptions {
                    retain_format: false,
                    ..SentenceBufferOptions::from_config(
                        lk_agents_abi::tts::markup::SentenceTokenizerConfig {
                            max_token_len,
                            min_token_len,
                            xml_aware,
                        },
                    )
                }
            }
            None => SentenceBufferOptions { xml_aware: expressive, ..Default::default() },
        };
        Box::new(SentenceBuffer::with_options(opts))
    }

    fn markup(provider: &str) -> Option<Box<dyn SentenceTextProcessor>> {
        text_processor(provider)
    }

    fn transcript(expressive: bool) -> Box<dyn TranscriptProcessor> {
        Box::new(AudioTranscriptNormalizer::new(expressive))
    }
}
