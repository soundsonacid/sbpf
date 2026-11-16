//! Where the wall time of program execution goes, per thread (feature `execution-timing`).
//!
//! Two clocks run: one around every [`EbpfVm::execute_program`](crate::vm::EbpfVm), one around
//! every built-in function that a running program calls (a syscall). A program run that is
//! started from inside a syscall (a cross-program invocation) is a run of its own, and it is
//! taken out of the time of the syscall that started it. So every nanosecond inside a top-level
//! run is in exactly one of two places:
//!
//! ```text
//! time inside top-level runs = instructions_ns + syscalls_ns
//! ```
//!
//! A built-in function that is not called by a running program is not a syscall and is not
//! timed. That is how the host invokes built-in programs, the loader that runs a program among
//! them.
//!
//! Each syscall costs two clock reads and some bookkeeping. Part of that lands in the syscall's
//! time and the rest in `instructions_ns`. [`syscall_timer_overhead`] measures both parts.

use std::cell::{Cell, RefCell};
use std::time::Instant;

/// Totals for the current thread since it started.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExecutionTimes {
    /// Time executing program instructions, in any execution mode, in every run.
    pub instructions_ns: u64,
    /// Time inside syscalls, without the program runs they start.
    pub syscalls_ns: u64,
    /// Number of syscalls.
    pub syscalls: u64,
    /// Number of program runs, the ones syscalls start included.
    pub program_runs: u64,
}

impl ExecutionTimes {
    /// What was added since `earlier`.
    pub fn since(&self, earlier: &ExecutionTimes) -> ExecutionTimes {
        ExecutionTimes {
            instructions_ns: self.instructions_ns - earlier.instructions_ns,
            syscalls_ns: self.syscalls_ns - earlier.syscalls_ns,
            syscalls: self.syscalls - earlier.syscalls,
            program_runs: self.program_runs - earlier.program_runs,
        }
    }
}

/// Totals of one syscall for the current thread since it started.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SyscallTimes {
    /// The type name of the built-in function.
    pub name: &'static str,
    pub calls: u64,
    /// Time inside it, without the program runs it started.
    pub ns: u64,
}

thread_local! {
    static TOTALS: Cell<ExecutionTimes> = const {
        Cell::new(ExecutionTimes {
            instructions_ns: 0,
            syscalls_ns: 0,
            syscalls: 0,
            program_runs: 0,
        })
    };
    static BY_SYSCALL: RefCell<Vec<SyscallTimes>> = const { RefCell::new(Vec::new()) };
    /// Whether the thread is executing program instructions right now, as opposed to host code
    /// (outside any run, or inside a syscall).
    static IN_PROGRAM: Cell<bool> = const { Cell::new(false) };
    /// Time that belongs to something nested in whatever is being timed right now: for a run,
    /// its syscalls; for a syscall, the runs it started.
    static NESTED_NS: Cell<u64> = const { Cell::new(0) };
}

/// The current thread's totals.
pub fn execution_times() -> ExecutionTimes {
    TOTALS.get()
}

/// Number of SBPF versions, `SBPFVersion::V0` being index 0.
pub const VERSIONS: usize = 6;

/// Totals of the programs of one SBPF version for the current thread since it started.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VersionTimes {
    /// Time executing the instructions of programs of this version.
    pub instructions_ns: u64,
    /// Runs of programs of this version.
    pub program_runs: u64,
    /// Syscalls made by programs of this version.
    pub syscalls: u64,
}

thread_local! {
    static BY_VERSION: [Cell<VersionTimes>; VERSIONS] = const {
        [const { Cell::new(VersionTimes { instructions_ns: 0, program_runs: 0, syscalls: 0 }) };
            VERSIONS]
    };
    /// The SBPF version of the program that is running, or that made the syscall in progress.
    static CURRENT_VERSION: Cell<usize> = const { Cell::new(0) };
}

thread_local! {
    static BY_EXECUTABLE: RefCell<std::collections::HashMap<usize, VersionTimes>> =
        RefCell::new(std::collections::HashMap::new());
    /// Syscalls the running program has made so far in its current run.
    static SYSCALLS_IN_RUN: Cell<u64> = const { Cell::new(0) };
}

/// The current thread's totals per executable, by the executable's address.
pub fn executable_times() -> std::collections::HashMap<usize, VersionTimes> {
    BY_EXECUTABLE.with_borrow(|by_executable| by_executable.clone())
}

/// The current thread's totals per SBPF version of the executing program, indexed by version.
pub fn version_times() -> [VersionTimes; VERSIONS] {
    BY_VERSION.with(|by_version| std::array::from_fn(|version| by_version[version].get()))
}

/// The current thread's totals per syscall, in order of first use.
pub fn syscall_times() -> Vec<SyscallTimes> {
    BY_SYSCALL.with_borrow(|by_syscall| by_syscall.clone())
}

/// Lives for one `execute_program`.
pub(crate) struct ProgramRunTimer {
    start: Instant,
    was_in_program: bool,
    /// The nested time of the syscall this run was started from.
    outer_nested_ns: u64,
    version: usize,
    /// The version of the program whose syscall started this run.
    outer_version: usize,
    /// The address of the executable.
    executable: usize,
    /// The syscalls so far of the run whose syscall started this run.
    outer_syscalls_in_run: u64,
}

impl ProgramRunTimer {
    pub(crate) fn start(version: crate::program::SBPFVersion, executable: usize) -> Self {
        let version = version as usize;
        Self {
            was_in_program: IN_PROGRAM.replace(true),
            outer_nested_ns: NESTED_NS.replace(0),
            version,
            outer_version: CURRENT_VERSION.replace(version),
            executable,
            outer_syscalls_in_run: SYSCALLS_IN_RUN.replace(0),
            start: Instant::now(),
        }
    }
}

impl Drop for ProgramRunTimer {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed().as_nanos() as u64;
        IN_PROGRAM.set(self.was_in_program);
        // What is left of the run after its syscalls is instructions. All of the run is nested
        // time of the syscall that started it.
        let in_syscalls = NESTED_NS.replace(self.outer_nested_ns.saturating_add(elapsed));
        let instructions_ns = elapsed.saturating_sub(in_syscalls);
        let mut totals = TOTALS.get();
        totals.instructions_ns += instructions_ns;
        totals.program_runs += 1;
        TOTALS.set(totals);
        CURRENT_VERSION.set(self.outer_version);
        BY_VERSION.with(|by_version| {
            let mut times = by_version[self.version].get();
            times.instructions_ns += instructions_ns;
            times.program_runs += 1;
            by_version[self.version].set(times);
        });
        let syscalls = SYSCALLS_IN_RUN.replace(self.outer_syscalls_in_run);
        BY_EXECUTABLE.with_borrow_mut(|by_executable| {
            let times = by_executable.entry(self.executable).or_default();
            times.instructions_ns += instructions_ns;
            times.program_runs += 1;
            times.syscalls += syscalls;
        });
    }
}

/// Lives for one built-in function call. It times the call when the call is a syscall.
pub(crate) struct BuiltinFunctionTimer(Option<SyscallTimer>);

struct SyscallTimer {
    name: &'static str,
    start: Instant,
    /// The nested time of the run this syscall was made from.
    outer_nested_ns: u64,
}

impl BuiltinFunctionTimer {
    pub(crate) fn start(name: &'static str) -> Self {
        Self(IN_PROGRAM.replace(false).then(|| SyscallTimer {
            name,
            outer_nested_ns: NESTED_NS.replace(0),
            start: Instant::now(),
        }))
    }
}

impl Drop for BuiltinFunctionTimer {
    fn drop(&mut self) {
        let Some(timer) = &self.0 else {
            return;
        };
        let elapsed = timer.start.elapsed().as_nanos() as u64;
        IN_PROGRAM.set(true);
        // What is left of the syscall after the runs it started is the syscall itself. All of
        // the syscall is nested time of the run that made it.
        let in_runs = NESTED_NS.replace(timer.outer_nested_ns.saturating_add(elapsed));
        let own = elapsed.saturating_sub(in_runs);
        let mut totals = TOTALS.get();
        totals.syscalls_ns += own;
        totals.syscalls += 1;
        TOTALS.set(totals);
        BY_VERSION.with(|by_version| {
            let caller = &by_version[CURRENT_VERSION.get()];
            let mut times = caller.get();
            times.syscalls += 1;
            caller.set(times);
        });
        SYSCALLS_IN_RUN.set(SYSCALLS_IN_RUN.get() + 1);
        BY_SYSCALL.with_borrow_mut(|by_syscall| {
            // Names are compared by address: one built-in function, one name.
            match by_syscall.iter_mut().find(|times| std::ptr::eq(times.name, timer.name)) {
                Some(times) => {
                    times.calls += 1;
                    times.ns += own;
                }
                None => by_syscall.push(SyscallTimes { name: timer.name, calls: 1, ns: own }),
            }
        });
    }
}

/// What timing one syscall costs, in nanoseconds: the part that ends up in the syscall's time,
/// and the part that ends up in `instructions_ns` of the run that made it.
///
/// Measured here and now by timing empty syscalls in a loop, which is the cheapest the timer
/// gets. The thread's totals are left as they were.
pub fn syscall_timer_overhead() -> (f64, f64) {
    const CALLS: u64 = 2_000_000;
    let totals = TOTALS.get();
    let by_syscall = syscall_times();
    let by_version = version_times();
    let syscalls_in_run = SYSCALLS_IN_RUN.get();
    let was_in_program = IN_PROGRAM.replace(true);
    let nested_ns = NESTED_NS.get();

    let start = Instant::now();
    for _ in 0..CALLS {
        drop(std::hint::black_box(BuiltinFunctionTimer::start("calibration")));
    }
    let all = start.elapsed().as_nanos() as f64 / CALLS as f64;
    let inside = (TOTALS.get().syscalls_ns - totals.syscalls_ns) as f64 / CALLS as f64;

    TOTALS.set(totals);
    BY_SYSCALL.set(by_syscall);
    BY_VERSION.with(|cells| {
        for (cell, times) in cells.iter().zip(by_version) {
            cell.set(times);
        }
    });
    SYSCALLS_IN_RUN.set(syscalls_in_run);
    IN_PROGRAM.set(was_in_program);
    NESTED_NS.set(nested_ns);
    (inside, all - inside)
}
