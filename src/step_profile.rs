//! EXPERIMENT, not for merging (feature `execution-timing`): which interpreter steps a program
//! executes, and how often.
//!
//! Parrot's interpreter has one step per first 16 bits of an instruction: the opcode and the
//! two registers. While the profile is on, the legacy interpreter counts the instructions it
//! executes by those 16 bits, per executable. That says which of parrot's steps a program
//! needs hot, without running parrot.
//!
//! The counting runs in its own loop, so the interpreter's normal loop is unchanged and costs
//! nothing more than one check per program run.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

/// One counter per value of the first 16 bits of an instruction.
pub const STEPS: usize = 1 << 16;

static ENABLED: AtomicBool = AtomicBool::new(false);

thread_local! {
    /// Executions per step, by the address of the executable.
    static PROFILES: RefCell<HashMap<usize, Box<[u64]>>> = RefCell::new(HashMap::new());
}

/// Turns the profile on or off for every thread. It starts off.
pub fn set_step_profile(enabled: bool) {
    ENABLED.store(enabled, Ordering::Relaxed);
}

pub(crate) fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// The current thread's profiles, by the address of the executable. Leaves them empty.
pub fn take_step_profiles() -> HashMap<usize, Box<[u64]>> {
    PROFILES.take()
}

/// The legacy interpreter's loop, counting each instruction before it executes.
pub(crate) fn run_interpreter<C: crate::vm::ContextObject>(
    mut interpreter: crate::interpreter::Interpreter<C>,
) {
    let executable = std::ptr::from_ref(interpreter.executable) as usize;
    let mut counts = PROFILES
        .with_borrow_mut(|profiles| profiles.remove(&executable))
        .unwrap_or_else(|| vec![0u64; STEPS].into_boxed_slice());
    loop {
        let offset = (interpreter.reg[11] as usize).saturating_mul(crate::ebpf::INSN_SIZE);
        if let Some([low, high]) = interpreter.program.get(offset..offset.saturating_add(2)) {
            counts[usize::from(u16::from_le_bytes([*low, *high]))] += 1;
        }
        if !interpreter.step() {
            break;
        }
    }
    interpreter.vm.registers[11] = interpreter.reg[11];
    // A program this one called into has put its own profile back by now.
    PROFILES.with_borrow_mut(|profiles| match profiles.remove(&executable) {
        // The program called itself, and the inner run counted separately.
        Some(inner) => {
            for (count, inner) in counts.iter_mut().zip(inner.iter()) {
                *count += inner;
            }
            profiles.insert(executable, counts);
        }
        None => {
            profiles.insert(executable, counts);
        }
    });
}
