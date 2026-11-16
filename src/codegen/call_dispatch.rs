//! EXPERIMENT, not for merging: counts what parrot's run-time call dispatch resolves, and can
//! answer repeated lookups from a small cache instead of the hash maps.
//!
//! In SBPF v0 to v2 a `call imm` names its target by a hash, for syscalls and for the program's
//! own functions alike. Parrot resolves that hash every time the instruction executes: first in
//! the loader's syscall registry, then in the executable's function registry. (In v3 only
//! syscalls are resolved this way.) The cache here stands in for resolving at compile time, to
//! measure what the lookups cost on real programs.
//!
//! The cache is keyed by the address of the registry and never invalidated, so it is only sound
//! while no executable is dropped. That holds in the benchmark it exists for.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};

static CACHE_ENABLED: AtomicBool = AtomicBool::new(false);

/// What the current thread's dispatches resolved to, since it started.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CallDispatchCounts {
    /// `call imm` of a v0 to v2 program that was one of the program's own functions.
    pub internal_calls: u64,
    /// `call imm` of a v0 to v2 program that was a syscall.
    pub syscalls: u64,
    /// Syscall of a v3 program.
    pub static_syscalls: u64,
}

impl CallDispatchCounts {
    pub fn since(&self, earlier: &CallDispatchCounts) -> CallDispatchCounts {
        CallDispatchCounts {
            internal_calls: self.internal_calls - earlier.internal_calls,
            syscalls: self.syscalls - earlier.syscalls,
            static_syscalls: self.static_syscalls - earlier.static_syscalls,
        }
    }
}

/// What a key resolved to.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Resolved {
    /// The address of a built-in function.
    Syscall(usize),
    /// The instruction index of a function of the program.
    Internal(usize),
}

#[derive(Clone, Copy)]
struct Entry {
    /// The address of the registry the key was resolved in. 0 for an empty entry.
    registry: usize,
    key: u32,
    resolved: Resolved,
}

const CACHE_ENTRIES: usize = 1024;
const EMPTY: Entry = Entry { registry: 0, key: 0, resolved: Resolved::Internal(0) };

thread_local! {
    static COUNTS: Cell<CallDispatchCounts> = const {
        Cell::new(CallDispatchCounts { internal_calls: 0, syscalls: 0, static_syscalls: 0 })
    };
    static CACHE: [Cell<Entry>; CACHE_ENTRIES] = const { [const { Cell::new(EMPTY) }; CACHE_ENTRIES] };
}

/// Turns the cache on or off for every thread. It starts off.
pub fn set_call_dispatch_cache(enabled: bool) {
    CACHE_ENABLED.store(enabled, Ordering::Relaxed);
}

/// The current thread's counts.
pub fn call_dispatch_counts() -> CallDispatchCounts {
    COUNTS.get()
}

pub(crate) fn count(update: impl FnOnce(&mut CallDispatchCounts)) {
    let mut counts = COUNTS.get();
    update(&mut counts);
    COUNTS.set(counts);
}

/// What `key` resolves to in the registry at address `registry`: from the cache when it is on
/// and has the answer, otherwise from `lookup`.
pub(crate) fn resolve(
    registry: usize,
    key: u32,
    lookup: impl FnOnce() -> Option<Resolved>,
) -> Option<Resolved> {
    if !CACHE_ENABLED.load(Ordering::Relaxed) {
        return lookup();
    }
    CACHE.with(|cache| {
        let slot = &cache[(key as usize ^ (registry >> 4)) % CACHE_ENTRIES];
        let entry = slot.get();
        if entry.registry == registry && entry.key == key {
            return Some(entry.resolved);
        }
        let resolved = lookup()?;
        slot.set(Entry { registry, key, resolved });
        Some(resolved)
    })
}
