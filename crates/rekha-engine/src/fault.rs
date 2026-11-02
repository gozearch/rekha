//! Deterministic fault injection for durability tests.
//!
//! Publication and recovery have failure windows that are impossible to reach
//! from a test by ordinary means: the code is correct on the happy path and the
//! interesting states only exist between two syscalls. Rather than leaving those
//! windows unexercised, the writer calls [`check`] at each boundary.
//!
//! The hook is a **thread-local**, so tests running in parallel cannot observe
//! each other's injected failures. It is inert unless a hook is installed.
//!
//! This is a test seam, not a configuration knob: it is `#[doc(hidden)]`, takes
//! no input from the network, and injection only ever makes an operation fail.

use std::cell::RefCell;

use crate::{EngineError, EngineResult};

/// Called with a fault-point name. `Err` aborts the operation at that point.
pub type Hook = fn(&str) -> Result<(), String>;

thread_local! {
    static HOOK: RefCell<Option<Hook>> = const { RefCell::new(None) };
}

/// Install (or clear) the hook for the current thread.
///
/// Returns nothing: a test either installs a hook for its own thread and clears
/// it on drop, or runs without one.
#[doc(hidden)]
pub fn set_hook(hook: Option<Hook>) {
    HOOK.with(|slot| *slot.borrow_mut() = hook);
}

/// Run the installed hook, if any, for `point`.
pub(crate) fn check(point: &str) -> EngineResult<()> {
    let result = HOOK.with(|slot| match *slot.borrow() {
        Some(hook) => hook(point),
        None => Ok(()),
    });
    result.map_err(|reason| {
        EngineError::Io(std::io::Error::other(format!(
            "fault injected at `{point}`: {reason}"
        )))
    })
}

/// Run the hook for `point`, treating a requested failure as a *successful*
/// publication with deferred cleanup. Used after the point where the checkpoint
/// becomes durable: from there on, an error must not be reported as "nothing
/// happened".
pub(crate) fn check_deferred(point: &str, stage: &'static str) -> Option<&'static str> {
    let injected = HOOK.with(|slot| match *slot.borrow() {
        Some(hook) => hook(point).is_err(),
        None => false,
    });
    injected.then_some(stage)
}
