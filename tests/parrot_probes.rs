//! Microbenchmarks behind the parrot diagnostics in seashell's `PARROT_DIAGNOSTICS.md`. All are
//! `#[ignore]`d and print a table. Run one pinned to a core:
//!
//! ```text
//! cargo test --release --test parrot_probes --no-run
//! taskset -c 12 target/release/deps/parrot_probes-<hash> --ignored --nocapture <name>
//! ```
//!
//! - `same_opcode_sweep`, `distinct_instruction_sweep`, `distinct_memory_instruction_sweep`:
//!   interpreter cost per instruction against the number of distinct (opcode, dst, src)
//!   instructions in the loop body.
//! - `layout_predictions`: the two loops whose speed the interpreter's step layout swaps. Run
//!   it once per value of `PARROT_STEP_LAYOUT` (`opcode-low`, `registers-low`, `mixed`).
//! - `internal_call_cost`, `syscall_dispatch_cost`: one call to a function of the program, and
//!   one syscall, in each engine and for SBPF v0 and v3.
#![cfg(all(feature = "jit", not(target_os = "windows"), target_arch = "x86_64"))]

use solana_sbpf::{
    assembler::assemble,
    create_vm,
    program::{BuiltinProgram, SBPFVersion},
    test_utils::TestContextObject,
    verifier::RequisiteVerifier,
    vm::{CallFrame, Config, ExecutionMode},
};
use std::{sync::Arc, time::Instant};

const OPS: [&str; 14] = [
    "add64", "sub64", "or64", "and64", "xor64", "lsh64", "rsh64", "arsh64", "mov64", "add32",
    "sub32", "or32", "and32", "xor32",
];

/// Loads and stores only: r1..r4 hold pointers into the input, r5..r8 hold the data.
#[test]
#[ignore]
fn distinct_memory_instruction_sweep() {
    use solana_sbpf::{ebpf, memory_region::MemoryRegion};
    let mut all = Vec::new();
    for size in ["dw", "w", "h", "b"] {
        for pointer in 1..5 {
            for value in 5..9 {
                all.push(format!("ldx{size} r{value}, [r{pointer}+8]"));
                all.push(format!("stx{size} [r{pointer}+16], r{value}"));
            }
        }
    }
    println!("| aligned mapping | distinct insns | legacy interp ns/insn | parrot interp ns/insn | legacy jit | parrot jit |");
    println!("|---|---:|---:|---:|---:|---:|");
    for aligned in [true, false] {
        for distinct in [8, 32, 128] {
            let body = all[..distinct].join("\n");
            let iterations = 2_000_000 / distinct;
            let assembly = format!(
                "mov64 r2, r1\nmov64 r3, r1\nmov64 r4, r1\nmov64 r9, {iterations}\nloop:\n{body}\nadd64 r9, -1\njne r9, 0, loop\nexit"
            );
            let config = Config {
                enabled_sbpf_versions: SBPFVersion::V0..=SBPFVersion::V0,
                aligned_memory_mapping: aligned,
                ..Config::default()
            };
            let executable = assemble::<TestContextObject>(
                &assembly,
                Arc::new(BuiltinProgram::new_loader(config)),
            )
            .unwrap();
            executable.verify::<RequisiteVerifier>().unwrap();

            let mut run = |mode: ExecutionMode| {
                let mut samples = Vec::new();
                let mut count = 0;
                for _ in 0..40 {
                    let mut mem = [0u8; 64];
                    let mut context_object = TestContextObject::new(100_000_000);
                    let region = MemoryRegion::new(&raw mut mem[..], ebpf::MM_INPUT_START);
                    create_vm!(vm, &executable, &mut context_object, stack, heap, vec![region], None);
                    let mut call_frames =
                        vec![CallFrame::default(); Config::default().max_call_depth];
                    let mut mode = mode;
                    let start = Instant::now();
                    let (instructions, result) =
                        vm.execute_program(&executable, &mut mode, &mut call_frames);
                    samples.push(start.elapsed().as_nanos() as u64);
                    assert!(result.is_ok(), "{result:?}");
                    count = instructions;
                }
                samples.sort_unstable();
                samples[samples.len() / 2] as f64 / count as f64
            };
            let legacy = run(ExecutionMode::Interpreted);
            let parrot = run(ExecutionMode::ParrotInterpreted);
            executable.jit_compile().unwrap();
            let legacy_jit = run(ExecutionMode::Jit);
            executable.parrot_compile().unwrap();
            let parrot_jit = run(ExecutionMode::ParrotJit);
            println!("| {aligned} | {distinct} | {legacy:.2} | {parrot:.2} | {legacy_jit:.2} | {parrot_jit:.2} |");
        }
    }
}

/// What one call to a function of the program itself costs, per SBPF version: a loop that
/// calls an empty function, against the same loop without the call.
#[test]
#[ignore]
fn internal_call_cost() {
    const ITERATIONS: u64 = 2_000_000;
    println!("| version | mode | ns per call + return |");
    println!("|---|---|---:|");
    for version in [SBPFVersion::V0, SBPFVersion::V3] {
        let per_iteration = |with_call: bool, mode: ExecutionMode| {
            let call = if with_call { "call function_foo" } else { "add64 r8, 1" };
            let assembly = format!(
                "mov64 r9, {ITERATIONS}\nloop:\n{call}\nadd64 r9, -1\njne r9, 0, loop\nexit\n\
                 function_foo:\nexit"
            );
            let config = Config {
                enabled_sbpf_versions: version..=version,
                ..Config::default()
            };
            let executable = assemble::<TestContextObject>(
                &assembly,
                Arc::new(BuiltinProgram::new_loader(config)),
            )
            .unwrap();
            executable.verify::<RequisiteVerifier>().unwrap();
            match mode {
                ExecutionMode::Jit => executable.jit_compile().unwrap(),
                ExecutionMode::ParrotJit => executable.parrot_compile().unwrap(),
                _ => {}
            }
            let mut samples = Vec::new();
            for _ in 0..20 {
                let mut context_object = TestContextObject::new(100_000_000);
                create_vm!(vm, &executable, &mut context_object, stack, heap, vec![], None);
                let mut call_frames = vec![CallFrame::default(); Config::default().max_call_depth];
                let mut mode = mode;
                let start = Instant::now();
                let (_, result) = vm.execute_program(&executable, &mut mode, &mut call_frames);
                samples.push(start.elapsed().as_nanos() as u64);
                assert!(result.is_ok(), "{result:?}");
            }
            samples.sort_unstable();
            samples[samples.len() / 2] as f64 / ITERATIONS as f64
        };
        for (name, mode) in [
            ("legacy interp", ExecutionMode::Interpreted),
            ("parrot interp", ExecutionMode::ParrotInterpreted),
            ("legacy jit", ExecutionMode::Jit),
            ("parrot jit", ExecutionMode::ParrotJit),
        ] {
            let cost = per_iteration(true, mode) - per_iteration(false, mode);
            println!("| {version:?} | {name} | {cost:.1} |");
        }
    }
}

/// What one syscall costs, per SBPF version: a loop that makes a trivial syscall, against the
/// same loop without it. The syscall's body is the same in every mode, so what differs between
/// modes is how the call gets there.
#[test]
#[ignore]
fn syscall_dispatch_cost() {
    use solana_sbpf::program::BuiltinFunctionDefinition;
    const ITERATIONS: u64 = 2_000_000;
    println!("| version | mode | ns per syscall |");
    println!("|---|---|---:|");
    for version in [SBPFVersion::V0, SBPFVersion::V3] {
        let per_iteration = |with_syscall: bool, mode: ExecutionMode| {
            let body = if with_syscall { "syscall bpf_gather_bytes" } else { "add64 r8, 1" };
            let assembly = format!(
                "mov64 r9, {ITERATIONS}\nloop:\n{body}\nadd64 r9, -1\njne r9, 0, loop\nexit"
            );
            let config = Config {
                enabled_sbpf_versions: version..=version,
                ..Config::default()
            };
            let mut loader = BuiltinProgram::new_loader(config);
            <solana_sbpf::test_syscalls::SyscallGatherBytes as BuiltinFunctionDefinition<
                TestContextObject,
            >>::register(&mut loader, "bpf_gather_bytes")
            .unwrap();
            let executable = assemble::<TestContextObject>(&assembly, Arc::new(loader)).unwrap();
            executable.verify::<RequisiteVerifier>().unwrap();
            match mode {
                ExecutionMode::Jit => executable.jit_compile().unwrap(),
                ExecutionMode::ParrotJit => executable.parrot_compile().unwrap(),
                _ => {}
            }
            let mut samples = Vec::new();
            for _ in 0..20 {
                let mut context_object = TestContextObject::new(100_000_000);
                create_vm!(vm, &executable, &mut context_object, stack, heap, vec![], None);
                let mut call_frames = vec![CallFrame::default(); Config::default().max_call_depth];
                let mut mode = mode;
                let start = Instant::now();
                let (_, result) = vm.execute_program(&executable, &mut mode, &mut call_frames);
                samples.push(start.elapsed().as_nanos() as u64);
                assert!(result.is_ok(), "{result:?}");
            }
            samples.sort_unstable();
            samples[samples.len() / 2] as f64 / ITERATIONS as f64
        };
        for (name, mode) in [
            ("legacy interp", ExecutionMode::Interpreted),
            ("parrot interp", ExecutionMode::ParrotInterpreted),
            ("legacy jit", ExecutionMode::Jit),
            ("parrot jit", ExecutionMode::ParrotJit),
        ] {
            let cost = per_iteration(true, mode) - per_iteration(false, mode);
            println!("| {version:?} | {name} | {cost:.1} |");
        }
    }
}

/// The two loops whose speed the step layout (`PARROT_STEP_LAYOUT`) should swap: N instructions
/// that share an opcode and differ in registers, and N instructions that share registers and
/// differ in opcode.
#[test]
#[ignore]
fn layout_predictions() {
    let pairs: Vec<(u8, u8)> = (0..9).flat_map(|dst| (0..9).map(move |src| (dst, src))).collect();
    let layout = std::env::var("PARROT_STEP_LAYOUT").unwrap_or_else(|_| "opcode-low".into());
    println!("| layout | loop body | N | legacy interp ns/insn | parrot interp ns/insn |");
    println!("|---|---|---:|---:|---:|");
    for distinct in [4, 8, 12, 14] {
        let same_opcode: Vec<String> = pairs[..distinct]
            .iter()
            .map(|(dst, src)| format!("add64 r{dst}, r{src}"))
            .collect();
        let same_registers: Vec<String> =
            OPS[..distinct].iter().map(|op| format!("{op} r1, r2")).collect();
        for (name, body) in [
            ("one opcode, N register pairs", same_opcode),
            ("N opcodes, one register pair", same_registers),
        ] {
            let body = body.join("\n");
            let iterations = 2_000_000 / distinct;
            let assembly = format!(
                "mov64 r9, {iterations}\nloop:\n{body}\nadd64 r9, -1\njne r9, 0, loop\nexit"
            );
            let config = Config {
                enabled_sbpf_versions: SBPFVersion::V0..=SBPFVersion::V0,
                ..Config::default()
            };
            let executable = assemble::<TestContextObject>(
                &assembly,
                Arc::new(BuiltinProgram::new_loader(config)),
            )
            .unwrap();
            executable.verify::<RequisiteVerifier>().unwrap();
            let run = |mode: ExecutionMode| {
                let mut samples = Vec::new();
                let mut count = 0;
                for _ in 0..40 {
                    let mut context_object = TestContextObject::new(100_000_000);
                    create_vm!(vm, &executable, &mut context_object, stack, heap, vec![], None);
                    let mut call_frames =
                        vec![CallFrame::default(); Config::default().max_call_depth];
                    let mut mode = mode;
                    let start = Instant::now();
                    let (instructions, result) =
                        vm.execute_program(&executable, &mut mode, &mut call_frames);
                    samples.push(start.elapsed().as_nanos() as u64);
                    assert!(result.is_ok(), "{result:?}");
                    count = instructions;
                }
                samples.sort_unstable();
                samples[samples.len() / 2] as f64 / count as f64
            };
            let legacy = run(ExecutionMode::Interpreted);
            let parrot = run(ExecutionMode::ParrotInterpreted);
            println!("| {layout} | {name} | {distinct} | {legacy:.2} | {parrot:.2} |");
        }
    }
}

/// The same number of distinct instructions, either all of one opcode (differing in registers
/// only) or spread over many opcodes.
#[test]
#[ignore]
fn same_opcode_sweep() {
    let pairs: Vec<(u8, u8)> = (0..9).flat_map(|dst| (0..9).map(move |src| (dst, src))).collect();
    println!("| distinct insns | layout | legacy interp ns/insn | parrot interp ns/insn |");
    println!("|---:|---|---:|---:|");
    for distinct in [4, 8, 12, 16, 32, 64] {
        let one_opcode: Vec<String> = pairs[..distinct]
            .iter()
            .map(|(dst, src)| format!("add64 r{dst}, r{src}"))
            .collect();
        // As many opcodes as possible, as few register pairs as possible.
        let many_opcodes: Vec<String> = (0..distinct)
            .map(|i| {
                let (dst, src) = pairs[i / OPS.len()];
                format!("{} r{dst}, r{src}", OPS[i % OPS.len()])
            })
            .collect();
        for (layout, body) in [("one opcode", one_opcode), ("many opcodes", many_opcodes)] {
            let body = body.join("\n");
            let iterations = 2_000_000 / distinct;
            let assembly = format!(
                "mov64 r9, {iterations}\nloop:\n{body}\nadd64 r9, -1\njne r9, 0, loop\nexit"
            );
            let config = Config {
                enabled_sbpf_versions: SBPFVersion::V0..=SBPFVersion::V0,
                ..Config::default()
            };
            let executable = assemble::<TestContextObject>(
                &assembly,
                Arc::new(BuiltinProgram::new_loader(config)),
            )
            .unwrap();
            executable.verify::<RequisiteVerifier>().unwrap();
            let run = |mode: ExecutionMode| {
                let mut samples = Vec::new();
                let mut count = 0;
                for _ in 0..40 {
                    let mut context_object = TestContextObject::new(100_000_000);
                    create_vm!(vm, &executable, &mut context_object, stack, heap, vec![], None);
                    let mut call_frames =
                        vec![CallFrame::default(); Config::default().max_call_depth];
                    let mut mode = mode;
                    let start = Instant::now();
                    let (instructions, result) =
                        vm.execute_program(&executable, &mut mode, &mut call_frames);
                    samples.push(start.elapsed().as_nanos() as u64);
                    assert!(result.is_ok(), "{result:?}");
                    count = instructions;
                }
                samples.sort_unstable();
                samples[samples.len() / 2] as f64 / count as f64
            };
            let legacy = run(ExecutionMode::Interpreted);
            let parrot = run(ExecutionMode::ParrotInterpreted);
            println!("| {distinct} | {layout} | {legacy:.2} | {parrot:.2} |");
        }
    }
}

#[test]
#[ignore]
fn distinct_instruction_sweep() {
    // Every (op, dst, src) over r0..r8, in an order that walks the registers first, so that
    // neighbours in the program are far apart in the interpreter table.
    let mut all = Vec::new();
    for dst in 0..9 {
        for src in 0..9 {
            for op in OPS {
                all.push(format!("{op} r{dst}, r{src}"));
            }
        }
    }
    println!("| distinct insns | legacy interp ns/insn | parrot interp ns/insn | legacy jit | parrot jit |");
    println!("|---:|---:|---:|---:|---:|");
    for distinct in [8, 32, 64, 128, 256, 512, 1024] {
        let body = all[..distinct].join("\n");
        let iterations = 2_000_000 / distinct;
        let assembly = format!(
            "mov64 r9, {iterations}\nloop:\n{body}\nadd64 r9, -1\njne r9, 0, loop\nexit"
        );
        let config = Config {
            enabled_sbpf_versions: SBPFVersion::V0..=SBPFVersion::V0,
            ..Config::default()
        };
        let executable = assemble::<TestContextObject>(
            &assembly,
            Arc::new(BuiltinProgram::new_loader(config)),
        )
        .unwrap();
        executable.verify::<RequisiteVerifier>().unwrap();

        let mut run = |mode: ExecutionMode| {
            let mut samples = Vec::new();
            let mut count = 0;
            for _ in 0..40 {
                let mut context_object = TestContextObject::new(100_000_000);
                create_vm!(vm, &executable, &mut context_object, stack, heap, vec![], None);
                let mut call_frames = vec![CallFrame::default(); Config::default().max_call_depth];
                let mut mode = mode;
                let start = Instant::now();
                let (instructions, result) =
                    vm.execute_program(&executable, &mut mode, &mut call_frames);
                samples.push(start.elapsed().as_nanos() as u64);
                assert!(result.is_ok(), "{result:?}");
                count = instructions;
            }
            samples.sort_unstable();
            samples[samples.len() / 2] as f64 / count as f64
        };
        let legacy = run(ExecutionMode::Interpreted);
        let parrot = run(ExecutionMode::ParrotInterpreted);
        executable.jit_compile().unwrap();
        let legacy_jit = run(ExecutionMode::Jit);
        executable.parrot_compile().unwrap();
        let parrot_jit = run(ExecutionMode::ParrotJit);
        println!("| {distinct} | {legacy:.2} | {parrot:.2} | {legacy_jit:.2} | {parrot_jit:.2} |");
    }
}
