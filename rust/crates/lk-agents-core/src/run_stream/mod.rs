//! Admission for one request's presentation and final-result consumer.

use lk_agents_abi::agent::stream::StreamRefused;

pub mod projection;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    Events,
    Text { delta: bool },
    Result,
}

#[derive(Default)]
pub struct Stream {
    entered: bool,
    attached: bool,
    ready: bool,
    viewed: bool,
    next: u64,
    consumer: Option<(u64, View)>,
    reading: bool,
    closing: bool,
    done: bool,
}

impl Stream {
    pub fn enter(&mut self) -> Result<(), StreamRefused> {
        if self.entered {
            return Err(StreamRefused::Reentered);
        }
        self.entered = true;
        Ok(())
    }
    pub fn attach(&mut self) -> Result<(), StreamRefused> {
        self.require_entered()?;
        if self.attached {
            return Err(StreamRefused::AlreadyBound);
        }
        self.attached = true;
        Ok(())
    }
    pub fn ready(&mut self) {
        self.ready = true;
    }
    fn require_entered(&self) -> Result<(), StreamRefused> {
        if !self.entered { Err(StreamRefused::NotEntered) } else { Ok(()) }
    }
    pub fn claim(&mut self, view: View) -> Result<u64, StreamRefused> {
        self.require_entered()?;
        if !self.ready {
            return Err(StreamRefused::NotStarted);
        }
        if view != View::Result && self.viewed {
            return Err(StreamRefused::IteratorClaimed);
        }
        if let Some((_, active)) = self.consumer {
            return Err(if active != View::Result {
                StreamRefused::IteratorActive
            } else {
                StreamRefused::ConsumerActive
            });
        }
        if view != View::Result {
            self.viewed = true;
        }
        self.next += 1;
        self.consumer = Some((self.next, view));
        Ok(self.next)
    }
    pub fn view(&self, id: u64) -> Result<View, StreamRefused> {
        self.consumer
            .filter(|(active, _)| *active == id)
            .map(|(_, view)| view)
            .ok_or(StreamRefused::CursorClosed)
    }
    pub fn read_view(&self, id: u64) -> Result<View, StreamRefused> {
        let view = self.view(id)?;
        if !self.reading {
            return Err(StreamRefused::ConsumerActive);
        }
        Ok(view)
    }
    pub fn begin_read(&mut self, id: u64) -> Result<View, StreamRefused> {
        let view = self.view(id)?;
        if self.reading {
            return Err(StreamRefused::ConsumerActive);
        }
        self.reading = true;
        Ok(view)
    }
    pub fn end_read(&mut self, id: u64) {
        if self.consumer.is_some_and(|(active, _)| active == id) {
            self.reading = false;
        }
    }
    pub fn release(&mut self, id: u64) {
        if self.consumer.is_some_and(|(active, _)| active == id) {
            self.consumer = None;
            self.reading = false;
        }
    }
    pub fn close_cursor(&mut self, id: u64) -> Result<(), StreamRefused> {
        if self.consumer.is_some_and(|(active, _)| active == id) && self.reading {
            return Err(StreamRefused::ConsumerActive);
        }
        self.release(id);
        Ok(())
    }
    /// Cancellation and normal completion both admit exactly one owned finalization.
    pub fn close(&mut self) -> bool {
        if !self.attached || self.closing || self.done {
            return false;
        }
        self.closing = true;
        true
    }
    pub fn settled(&mut self) {
        self.done = true;
    }
    pub fn done(&self) -> bool {
        self.done
    }
    pub fn attached(&self) -> bool {
        self.attached
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_iterator_and_one_reader_even_when_the_consumer_is_result() {
        let mut stream = Stream::default();
        assert_eq!(stream.claim(View::Result), Err(StreamRefused::NotEntered));
        stream.enter().unwrap();
        assert_eq!(stream.enter(), Err(StreamRefused::Reentered));
        stream.attach().unwrap();
        stream.ready();
        let iterator = stream.claim(View::Events).unwrap();
        assert_eq!(stream.claim(View::Result), Err(StreamRefused::IteratorActive));
        stream.begin_read(iterator).unwrap();
        assert_eq!(stream.begin_read(iterator), Err(StreamRefused::ConsumerActive));
        stream.release(iterator);
        let result = stream.claim(View::Result).unwrap();
        assert_eq!(stream.claim(View::Result), Err(StreamRefused::ConsumerActive));
        assert_eq!(stream.claim(View::Text { delta: true }), Err(StreamRefused::IteratorClaimed));
        // A cancelled older read cannot release the new consumer's admission.
        stream.end_read(iterator);
        stream.release(iterator);
        assert_eq!(stream.begin_read(result), Ok(View::Result));
    }

    #[test]
    fn close_is_admitted_once_and_late_cancellation_cannot_replace_completion() {
        let mut stream = Stream::default();
        assert!(!stream.close());
        stream.enter().unwrap();
        stream.attach().unwrap();
        assert!(stream.close());
        assert!(!stream.close());
        stream.settled();
        assert!(stream.done());
        assert!(!stream.close());
    }
}
