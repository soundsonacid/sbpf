
use solana_sbpf::{
    assembler::assemble,
    declare_builtin_function,
    memory_region::{MemoryMapping, MemoryRegion},
    program::{BuiltinFunctionDefinition, BuiltinProgram, SBPFVersion},
    verifier::RequisiteVerifier,
    vm::{CallFrame, Config, ContextObject, ExecutionMode},
};
use std::{collections::BTreeMap, ptr, sync::Arc};
use test_utils::create_vm;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum FrameKey {
    Func(u64),
    Syscall(u32),
}

struct CctNode {
    #[allow(dead_code)]
    key: FrameKey,
    parent: Option<usize>,
    /// CUs charged directly to this activation (not its callees).
    self_cu: u64,
    /// `(callsite_pc, child frame) -> node index`: the same callee invoked from
    /// two different pcs gets two nodes (call-site context, for DWARF inline
    /// attribution downstream).
    children: BTreeMap<(u64, FrameKey), usize>,
}

#[derive(Default)]
struct Cct {
    nodes: Vec<CctNode>,
    current: Option<usize>,
    /// Depth of syscall nesting. While > 0 we are inside a syscall's host code
    /// (possibly a CPI running a nested VM): base/call/return events are ignored
    /// so the syscall stays one opaque leaf and its cost — measured once via the
    /// get_remaining() delta — is not double counted. (Slice 4 flips this to
    /// descend into interpreted CPI callees instead of suppressing.)
    syscall_depth: u32,
}

impl Cct {
    fn push_child(&mut self, parent: usize, callsite_pc: u64, key: FrameKey) -> usize {
        if let Some(&idx) = self.nodes[parent].children.get(&(callsite_pc, key)) {
            return idx;
        }
        let idx = self.nodes.len();
        self.nodes.push(CctNode {
            key,
            parent: Some(parent),
            self_cu: 0,
            children: BTreeMap::new(),
        });
        self.nodes[parent].children.insert((callsite_pc, key), idx);
        idx
    }

    fn on_instruction(&mut self, pc: u64) {
        if self.syscall_depth > 0 {
            return;
        }
        let cur = match self.current {
            Some(cur) => cur,
            None => {
                // Lazily establish the root from the entrypoint pc.
                self.nodes.push(CctNode {
                    key: FrameKey::Func(pc),
                    parent: None,
                    self_cu: 0,
                    children: BTreeMap::new(),
                });
                self.current = Some(0);
                0
            }
        };
        self.nodes[cur].self_cu += 1;
    }

    fn on_call(&mut self, callsite_pc: u64, target_pc: u64) {
        if self.syscall_depth > 0 {
            return;
        }
        let cur = self.current.expect("call before any instruction executed");
        self.current = Some(self.push_child(cur, callsite_pc, FrameKey::Func(target_pc)));
    }

    fn on_return(&mut self) {
        if self.syscall_depth > 0 {
            return;
        }
        let cur = self.current.expect("return before any instruction executed");
        self.current = self.nodes[cur].parent;
    }

    fn on_syscall_enter(&mut self, callsite_pc: u64, key: u32) {
        // Only the outermost syscall creates a frame; nested syscalls (reached
        // through a CPI) fold into it.
        if self.syscall_depth == 0 {
            let cur = self.current.expect("syscall before any instruction executed");
            self.current = Some(self.push_child(cur, callsite_pc, FrameKey::Syscall(key)));
        }
        self.syscall_depth += 1;
    }

    fn on_syscall_exit(&mut self, cost: u64) {
        self.syscall_depth -= 1;
        if self.syscall_depth == 0 {
            let cur = self.current.expect("syscall exit without a current frame");
            self.nodes[cur].self_cu += cost;
            self.current = self.nodes[cur].parent;
        }
    }

    fn total_self_cu(&self) -> u64 {
        self.nodes.iter().map(|n| n.self_cu).sum()
    }

    /// The single child of `idx`, panicking if the fan-out is not exactly one.
    fn only_child(&self, idx: usize) -> usize {
        assert_eq!(
            self.nodes[idx].children.len(),
            1,
            "node {idx} expected exactly one child"
        );
        *self.nodes[idx].children.values().next().unwrap()
    }
}

struct ProfilingContextObject {
    cct: Cct,
    remaining: u64,
    memory_mapping: MemoryMapping,
}

impl Default for ProfilingContextObject {
    fn default() -> Self {
        Self {
            cct: Cct::default(),
            remaining: 0,
            memory_mapping: unsafe {
                MemoryMapping::new(vec![], &Config::default(), SBPFVersion::Reserved)
            }
            .unwrap(),
        }
    }
}

impl ContextObject for ProfilingContextObject {
    fn active_mapping_ptr(&mut self) -> ptr::NonNull<MemoryMapping> {
        ptr::NonNull::from_ref(&mut self.memory_mapping)
    }
    fn consume(&mut self, amount: u64) {
        self.remaining = self.remaining.saturating_sub(amount);
    }
    fn get_remaining(&self) -> u64 {
        self.remaining
    }
    fn on_instruction(&mut self, pc: u64) {
        self.cct.on_instruction(pc);
    }
    fn on_call(&mut self, callsite_pc: u64, target_pc: u64) {
        self.cct.on_call(callsite_pc, target_pc);
    }
    fn on_return(&mut self) {
        self.cct.on_return();
    }
    fn on_syscall_enter(&mut self, callsite_pc: u64, key: u32) {
        self.cct.on_syscall_enter(callsite_pc, key);
    }
    fn on_syscall_exit(&mut self, cost: u64) {
        self.cct.on_syscall_exit(cost);
    }
}

declare_builtin_function!(
    /// Test syscall: burns `arg1` compute units (via `consume`) and returns 0.
    /// Lets a test dial in an exact, known syscall cost.
    SyscallBurnCu,
    fn rust(
        context_object: &mut ProfilingContextObject,
        cost: u64,
        _arg2: u64,
        _arg3: u64,
        _arg4: u64,
        _arg5: u64,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        context_object.consume(cost);
        Ok(0)
    }
);

const INITIAL_CU: u64 = 1_000_000;

#[test]
fn test_cu_conservation_nested_calls() {
    // root ── calls function_a twice; each function_a calls function_b once.
    //
    // Under SBPF >= V3 stricter verification every function must begin with the
    // `add64 r10, imm` start marker (ebpf::is_function_start_marker) and end with
    // ja/return, so each function opens with `add64 r10, 0` and ends with `exit`.
    //
    //   root:        add64 r10,0 | call a | call a | exit    (4 insns)
    //   function_a:  add64 r10,0 | call b | exit    (x2 activations = 6 insns)
    //   function_b:  add64 r10,0 | exit             (x2 activations = 4 insns)
    //
    // Children are keyed by call site, so the two calls to function_a (pc 1 and
    // pc 2) become two distinct nodes under root; each calls function_b from the
    // same site (pc 5), folding into one b node per a activation. Expected:
    // root=4, a=3 each, b=2 each, total=14.
    let source = "
        add64 r10, 0
        call function_a
        call function_a
        exit
        function_a:
        add64 r10, 0
        call function_b
        exit
        function_b:
        add64 r10, 0
        exit";

    let config = Config::default();
    assert!(
        config.enable_instruction_meter,
        "instruction_count is only meaningful with the meter enabled"
    );
    let loader = Arc::new(BuiltinProgram::<ProfilingContextObject>::new_loader(config));
    let executable = assemble::<ProfilingContextObject>(source, loader).unwrap();
    executable.verify::<RequisiteVerifier>().unwrap();

    let mut context_object = ProfilingContextObject {
        remaining: INITIAL_CU,
        ..Default::default()
    };
    create_vm!(
        vm,
        &executable,
        &mut context_object,
        stack,
        heap,
        Vec::<MemoryRegion>::new(),
        None
    );
    let mut call_frames = vec![CallFrame::default(); Config::default().max_call_depth];
    let (instruction_count, result) =
        vm.execute_program(&executable, &mut ExecutionMode::Interpreted, &mut call_frames);
    result.unwrap();

    let cct = &vm.context().cct;

    // Conservation: every consumed CU landed on exactly one node.
    assert_eq!(
        cct.total_self_cu(),
        instruction_count,
        "CCT self-CU ({}) must equal consumed instruction count ({})",
        cct.total_self_cu(),
        instruction_count,
    );
    assert_eq!(instruction_count, 14, "sanity: expected 14 instructions");

    // Structure: root ─▶ function_a (per call site) ─▶ function_b.
    assert_eq!(
        cct.nodes.len(),
        5,
        "root + function_a x2 (call sites pc 1 / pc 2) + function_b under each"
    );
    let root = 0;
    assert_eq!(cct.nodes[root].parent, None);
    let callsites: Vec<u64> = cct.nodes[root].children.keys().map(|&(cs, _)| cs).collect();
    assert_eq!(callsites, vec![1, 2], "function_a called from pc 1 and pc 2");
    for (&(_, key), &a) in cct.nodes[root].children.iter() {
        assert!(matches!(key, FrameKey::Func(_)));
        let b = cct.only_child(a);
        assert_eq!(cct.nodes[b].children.len(), 0, "function_b is a leaf");
        assert_eq!(cct.nodes[a].self_cu, 3, "function_a self-CU per call site");
        assert_eq!(cct.nodes[b].self_cu, 2, "function_b self-CU per activation");
    }
    assert_eq!(cct.nodes[root].self_cu, 4, "root self-CU");
}

#[test]
fn test_cu_syscall_gets_its_own_frame() {
    // A syscall that burns a known 100 CU must appear as its own frame under the
    // caller, carrying exactly those 100 CU — not folded into the caller's base
    // cost. This is the "where did the CUs go" property for syscall-heavy code.
    //
    //   root: add64 r10,0 | mov r1, SYSCALL_COST | syscall burn_cu | exit   (4 base)
    //         └─ burn_cu: 100 CU
    //
    // Expected: root=4 (base), syscall frame=100, total=104.
    const SYSCALL_COST: u64 = 100;
    let source = "
        add64 r10, 0
        mov r1, 100
        syscall burn_cu
        exit";

    // Force V3: static-syscall era, so `syscall <name>` dispatches through
    // dispatch_syscall (matches how test_utils exercises syscalls).
    let config = Config {
        enabled_sbpf_versions: SBPFVersion::V3..=SBPFVersion::V3,
        ..Config::default()
    };
    let mut loader = BuiltinProgram::<ProfilingContextObject>::new_loader(config);
    SyscallBurnCu::register(&mut loader, "burn_cu").unwrap();
    let loader = Arc::new(loader);
    let executable = assemble::<ProfilingContextObject>(source, loader).unwrap();
    executable.verify::<RequisiteVerifier>().unwrap();

    let mut context_object = ProfilingContextObject {
        remaining: INITIAL_CU,
        ..Default::default()
    };
    create_vm!(
        vm,
        &executable,
        &mut context_object,
        stack,
        heap,
        Vec::<MemoryRegion>::new(),
        None
    );
    let mut call_frames = vec![CallFrame::default(); Config::default().max_call_depth];
    let (instruction_count, result) =
        vm.execute_program(&executable, &mut ExecutionMode::Interpreted, &mut call_frames);
    result.unwrap();

    let consumed = INITIAL_CU - vm.context().get_remaining();
    let cct = &vm.context().cct;

    eprintln!(
        "instruction_count={instruction_count} consumed(meter)={consumed} \
         total_self_cu={} nodes={}",
        cct.total_self_cu(),
        cct.nodes.len(),
    );
    for (i, n) in cct.nodes.iter().enumerate() {
        let k = match n.key {
            FrameKey::Func(pc) => format!("Func(pc={pc})"),
            FrameKey::Syscall(key) => format!("Syscall(key={key:#x})"),
        };
        eprintln!("  node[{i}] {k} parent={:?} self_cu={}", n.parent, n.self_cu);
    }

    // Conservation against the real meter delta (base instructions + syscall CU).
    assert_eq!(
        cct.total_self_cu(),
        consumed,
        "CCT self-CU must equal the CU the meter actually consumed",
    );

    // Structure: root ─▶ syscall(burn_cu), keyed by the syscall instruction's pc.
    assert_eq!(cct.nodes.len(), 2, "root + syscall frame");
    let root = 0;
    let syscall = cct.only_child(root);
    assert!(
        matches!(cct.nodes[syscall].key, FrameKey::Syscall(_)),
        "child of root must be a syscall frame",
    );
    let &(callsite, _) = cct.nodes[root].children.keys().next().unwrap();
    assert_eq!(callsite, 2, "syscall invoked from pc 2");
    assert_eq!(
        cct.nodes[syscall].self_cu, SYSCALL_COST,
        "syscall frame carries exactly the CU it burned",
    );
    assert_eq!(
        cct.nodes[root].self_cu,
        consumed - SYSCALL_COST,
        "root carries only the base instruction CU",
    );
}

#[test]
fn test_cpi_descent_state_machine() {
    // Slice 4 (CPI): drives the real `CuProfiler` hook sequence a CPI produces
    // (the runtime's on_program_enter/exit around a nested interpreted VM) and
    // checks structure + cost conservation. The interpreter-level integration
    // is covered by seashell's end-to-end CPI test; this validates the state
    // machine in isolation.
    use solana_sbpf::profiler::{CuProfiler, FrameKey as Key};

    let caller_id = [1u8; 32];
    let callee_id = [2u8; 32];
    let mut p = CuProfiler::new();

    // Runtime enters the top-level program; it executes pcs 0..=2.
    p.on_program_enter(caller_id);
    p.on_instruction(0);
    p.on_instruction(1);
    p.on_instruction(2);

    // pc 2 is sol_invoke_signed; the runtime descends into an interpreted
    // callee which runs 5 instructions and one 100-CU syscall of its own.
    p.on_syscall_enter(2, 0xAB);
    p.on_program_enter(callee_id);
    for pc in 0..5 {
        p.on_instruction(pc);
    }
    p.on_syscall_enter(4, 0xCD);
    p.on_syscall_exit(100);
    p.on_program_exit();
    // Gross meter delta across the invoke: 1200. The 105 CU attributed inside
    // the callee subtree (5 base + 100 syscall) must be subtracted, leaving
    // 1095 of pure invoke overhead on the syscall frame.
    p.on_syscall_exit(1200);

    // One more caller instruction after the CPI returns.
    p.on_instruction(3);
    p.on_program_exit();

    // Structure: Program(caller) > Func(0) > Syscall(0xAB) > Program(callee) > Func(0) > Syscall(0xCD).
    let keys: Vec<Key> = p.nodes.iter().map(|n| n.key).collect();
    assert_eq!(
        keys,
        vec![
            Key::Program(caller_id),
            Key::Func(0),
            Key::Syscall(0xAB),
            Key::Program(callee_id),
            Key::Func(0),
            Key::Syscall(0xCD),
        ],
    );
    assert_eq!(p.nodes[1].parent, Some(0));
    assert_eq!(p.nodes[2].parent, Some(1));
    assert_eq!(p.nodes[3].parent, Some(2));
    assert_eq!(p.nodes[4].parent, Some(3));
    assert_eq!(p.nodes[5].parent, Some(4));
    // Entry functions have no call site; syscalls carry their invoking pc.
    assert_eq!(p.nodes[1].callsite_pc, None);
    assert_eq!(p.nodes[2].callsite_pc, Some(2));
    assert_eq!(p.nodes[4].callsite_pc, None);
    assert_eq!(p.nodes[5].callsite_pc, Some(4));

    // Costs: caller base 4, invoke overhead 1095, callee base 5, callee syscall 100.
    assert_eq!(p.nodes[1].self_cu, 4, "caller base instructions");
    assert_eq!(p.nodes[2].self_cu, 1095, "invoke frame = gross - nested");
    assert_eq!(p.nodes[4].self_cu, 5, "callee base instructions");
    assert_eq!(p.nodes[5].self_cu, 100, "callee syscall");

    // Conservation: what the meter saw = caller base (4) + gross invoke (1200).
    assert_eq!(p.total_self_cu(), 4 + 1200);
}
