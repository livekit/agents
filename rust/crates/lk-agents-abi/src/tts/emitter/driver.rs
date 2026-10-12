//! The host-effect driver over [`AudioEmitter`]. No host call is made with the machine lock held;
//! a re-entrant call finds `draining` set and leaves its command for the outer drain. Divergence:
//! frames pushed before `aclose()` have already reached the consumer, where the reference drops them.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use super::{AudioEmitter, Effects, EmitterError, SegmentDump};
use crate::{SynthesizedAudio, TimedString, Ts};

pub enum Cmd {
    StartSegment(String),
    EndSegment,
    /// A decoding host decodes before queueing.
    Push(Vec<u8>),
    Words(Vec<TimedString>),
    Flush,
    EndInput,
    Drift,
}

pub enum HostEffect {
    Send(Box<SynthesizedAudio>),
    /// Relative, so the clocks need not agree.
    Arm(Duration),
    Cancel,
    Dump(Box<SegmentDump>),
    Dropped(usize),
    ProviderIds(Vec<String>),
}

/// Called with no lock held, so it may re-enter the driver.
pub trait EmitterHost {
    type Error;

    fn deliver(&mut self, effect: HostEffect) -> Result<(), Self::Error>;
}

/// Effects produced before a machine error are delivered first.
#[derive(Debug)]
pub enum PumpError<E> {
    Machine(EmitterError),
    Host(E),
}

pub struct EmitterDriver {
    machine: Mutex<AudioEmitter>,
    intake: Mutex<VecDeque<Cmd>>,
    /// A re-entrant call leaves its command in `intake` and returns.
    draining: AtomicBool,
    /// So the machine's `Ts` is a virtualisable number rather than a wall clock.
    origin: Instant,
}

impl EmitterDriver {
    pub fn new(machine: AudioEmitter) -> Self {
        EmitterDriver {
            machine: Mutex::new(machine),
            intake: Mutex::new(VecDeque::new()),
            draining: AtomicBool::new(false),
            origin: Instant::now(),
        }
    }

    pub fn now(&self) -> Ts {
        Ts::from_nanos(self.origin.elapsed().as_nanos().min(u64::MAX as u128) as u64)
    }

    /// The machine, for reads and effect-free calls. **Never hold this across a host call.**
    pub fn machine(&self) -> MutexGuard<'_, AudioEmitter> {
        self.machine.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn submit<H: EmitterHost>(
        &self,
        cmd: Cmd,
        host: &mut H,
    ) -> Result<(), PumpError<H::Error>> {
        self.intake.lock().unwrap_or_else(|e| e.into_inner()).push_back(cmd);
        self.pump(host)
    }

    /// Effects replay after releasing the lock. A re-entrant call returns at once.
    pub fn pump<H: EmitterHost>(&self, host: &mut H) -> Result<(), PumpError<H::Error>> {
        if self.draining.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        let r = self.pump_loop(host);
        self.draining.store(false, Ordering::Release);
        r
    }

    fn pump_loop<H: EmitterHost>(&self, host: &mut H) -> Result<(), PumpError<H::Error>> {
        loop {
            let (effects, err) = {
                let mut m = self.machine();
                let now = self.now();
                let mut effects = Effects::new();
                let mut err = None;
                loop {
                    let next = self.intake.lock().unwrap_or_else(|e| e.into_inner()).pop_front();
                    let Some(cmd) = next else { break };
                    // An early `break` keeps what the failing command produced.
                    if let Err(e) = apply(&mut m, cmd, now, &mut effects) {
                        err = Some(e);
                        break;
                    }
                }
                if let Some(ids) = m.take_provider_ids_update() {
                    effects.push(HostEffect::ProviderIds(ids));
                }
                for d in m.take_dumps() {
                    effects.push(HostEffect::Dump(Box::new(d)));
                }
                let dropped = m.take_dropped_bytes();
                if dropped > 0 {
                    effects.push(HostEffect::Dropped(dropped));
                }
                (effects, err)
            };

            if effects.is_empty() {
                return err.map_or(Ok(()), |e| Err(PumpError::Machine(e)));
            }
            for e in effects {
                host.deliver(e).map_err(PumpError::Host)?;
            }
            if let Some(e) = err {
                return Err(PumpError::Machine(e));
            }
            if self.intake.lock().unwrap_or_else(|e| e.into_inner()).is_empty() {
                return Ok(());
            }
        }
    }

    /// Forgets whatever was queued; the host cancels its timer.
    pub fn stop(&self) {
        {
            let mut m = self.machine();
            m.stop(&mut Effects::new());
        }
        self.intake.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
}

fn apply(m: &mut AudioEmitter, cmd: Cmd, now: Ts, eff: &mut Effects) -> Result<(), EmitterError> {
    match cmd {
        // The provider id is pulled after the batch, so it is written even if the segment fails.
        Cmd::StartSegment(id) => m.start_segment(&id).map(|_| ()),
        Cmd::EndSegment => m.end_segment(now, eff),
        Cmd::Push(data) => m.push(now, &data, eff),
        Cmd::Words(w) => m.push_timed_transcript(w),
        Cmd::Flush => m.flush(now, eff),
        Cmd::EndInput => m.end_input(now, eff),
        Cmd::Drift => m.on_drift(now, eff),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A consumer that calls straight back into the driver from inside `Send`.
    struct Reentrant<'a> {
        driver: &'a EmitterDriver,
        sent: Vec<SynthesizedAudio>,
    }

    impl EmitterHost for Reentrant<'_> {
        type Error = ();

        fn deliver(&mut self, effect: HostEffect) -> Result<(), ()> {
            if let HostEffect::Send(audio) = effect {
                self.sent.push(*audio);
                let driver = self.driver;
                // The re-entrant flush must be queued, not run: the outer drain holds the queue.
                driver.submit(Cmd::Flush, self).map_err(|_| ())?;
            }
            Ok(())
        }
    }

    // Catches: a missing single-drainer latch, or a re-entrant command dropped instead of queued.
    #[test]
    fn a_re_entrant_command_is_queued_and_run_by_the_outer_drain_exactly_once() {
        let mut machine = AudioEmitter::new("test");
        machine.initialize("req", 1000, 1, 100, true).unwrap();
        let driver = EmitterDriver::new(machine);
        let mut host = Reentrant { driver: &driver, sent: Vec::new() };

        driver.submit(Cmd::StartSegment("s1".into()), &mut host).unwrap();
        driver.submit(Cmd::Push(vec![0xff; 600]), &mut host).unwrap();
        driver.submit(Cmd::EndSegment, &mut host).unwrap();

        let real: u32 = host
            .sent
            .iter()
            .filter(|a| !a.is_final)
            .map(|a| a.frame.samples_per_channel)
            .sum();
        let finals: Vec<u32> = host
            .sent
            .iter()
            .filter(|a| a.is_final)
            .map(|a| a.frame.samples_per_channel)
            .collect();
        assert_eq!(real, 300, "every pushed sample came out exactly once");
        // The flushes drained the tail, so the segment closes on the synthetic 10 ms marker.
        assert_eq!(finals, vec![10]);
        assert!(!driver.draining.load(Ordering::Acquire), "the latch is released after a drain");
    }

    // Catches: a machine error discarding the effects its batch already produced, or preceding them.
    #[test]
    fn effects_before_a_machine_error_are_delivered_and_the_error_follows() {
        struct Log(Vec<&'static str>);
        impl EmitterHost for Log {
            type Error = ();
            fn deliver(&mut self, effect: HostEffect) -> Result<(), ()> {
                self.0.push(match effect {
                    HostEffect::Send(_) => "send",
                    HostEffect::ProviderIds(_) => "ids",
                    _ => "other",
                });
                Ok(())
            }
        }
        let mut machine = AudioEmitter::new("test");
        machine.initialize("req", 1000, 1, 100, true).unwrap();
        let driver = EmitterDriver::new(machine);
        let mut host = Log(Vec::new());
        driver.submit(Cmd::StartSegment("s1".into()), &mut host).unwrap();
        driver.submit(Cmd::Push(vec![0xff; 600]), &mut host).unwrap();
        // A second `start_segment` while one is open is the reference's `RuntimeError`.
        let err = driver.submit(Cmd::StartSegment("s2".into()), &mut host).unwrap_err();
        assert!(matches!(err, PumpError::Machine(EmitterError::SegmentAlreadyOpen)));
        assert!(host.0.contains(&"send"), "the frames from the push reached the host");
        assert_eq!(host.0.last(), Some(&"ids"), "the id noted before the rejection is written");
    }
}
