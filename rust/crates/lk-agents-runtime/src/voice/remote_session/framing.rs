const HEADER_SIZE: usize = 4;
pub(in crate::voice) const MAX_MESSAGE_SIZE: usize = 1 << 20;

pub fn frame(payload: &[u8]) -> Result<Vec<u8>, FrameError> {
    let len = payload.len();
    if len > MAX_MESSAGE_SIZE {
        return Err(FrameError::Oversize(len));
    }
    let mut out = Vec::with_capacity(HEADER_SIZE + len);
    out.extend_from_slice(&(len as u32).to_be_bytes());
    out.extend_from_slice(payload);
    Ok(out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum FrameError {
    #[error("frame of {0} bytes exceeds the {MAX_MESSAGE_SIZE} byte limit")]
    Oversize(usize),
}

#[derive(Debug, Default)]
pub struct FrameDecoder {
    buf: Vec<u8>,
}

impl FrameDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, chunk: &[u8]) {
        self.buf.extend_from_slice(chunk);
    }

    pub fn next_frame(&mut self) -> Result<Option<Vec<u8>>, FrameError> {
        if self.buf.len() < HEADER_SIZE {
            return Ok(None);
        }
        let len = u32::from_be_bytes([self.buf[0], self.buf[1], self.buf[2], self.buf[3]]) as usize;
        if len > MAX_MESSAGE_SIZE {
            return Err(FrameError::Oversize(len));
        }
        let total = HEADER_SIZE + len;
        if self.buf.len() < total {
            return Ok(None);
        }
        let body = self.buf[HEADER_SIZE..total].to_vec();
        self.buf.drain(..total);
        Ok(Some(body))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_frames_wait_and_coalesced_frames_are_separate() {
        let mut decoder = FrameDecoder::new();
        let wire = frame(b"hello").unwrap();
        assert_eq!(frame(b"ab").unwrap(), [0, 0, 0, 2, b'a', b'b']);
        decoder.push(&wire[..2]);
        assert_eq!(decoder.next_frame().unwrap(), None);
        decoder.push(&wire[2..]);
        decoder.push(&frame(b"world").unwrap());
        assert_eq!(decoder.next_frame().unwrap(), Some(b"hello".to_vec()));
        assert_eq!(decoder.next_frame().unwrap(), Some(b"world".to_vec()));
        assert_eq!(decoder.next_frame().unwrap(), None);
    }

    #[test]
    fn oversize_is_refused_before_reading_the_body() {
        let mut decoder = FrameDecoder::new();
        decoder.push(&((MAX_MESSAGE_SIZE as u32) + 1).to_be_bytes());
        assert!(matches!(decoder.next_frame(), Err(FrameError::Oversize(_))));
    }

    #[test]
    fn oversize_is_refused_on_encode() {
        assert_eq!(
            frame(&vec![0; MAX_MESSAGE_SIZE]).unwrap().len(),
            HEADER_SIZE + MAX_MESSAGE_SIZE
        );
        assert_eq!(
            frame(&vec![0; MAX_MESSAGE_SIZE + 1]),
            Err(FrameError::Oversize(MAX_MESSAGE_SIZE + 1))
        );
    }
}
