//! Compute-unit profiler: a calling-context tree built from the interpreter's
//! instrumentation hooks.

use std::collections::BTreeMap;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum FrameKey {
    Func(u64),
    Syscall(u32),
    Program([u8; 32]),
}

#[derive(Clone, Debug)]
pub struct CctNode {
    pub key: FrameKey,
    pub parent: Option<usize>,
    pub callsite_pc: Option<u64>,
    pub self_cu: u64,
    pub pc_cu: BTreeMap<u64, u64>,
    pub children: BTreeMap<(Option<u64>, FrameKey), usize>,
}

#[derive(Clone, Default, Debug)]
pub struct CuProfiler {
    pub nodes: Vec<CctNode>,
    current: Option<usize>,
    syscall_depth: u32,
    attributed: u64,
    syscall_snapshots: Vec<u64>,
    program_saves: Vec<(Option<usize>, u32, bool)>,
    entering: bool,
}

impl CuProfiler {
    pub fn new() -> Self {
        Self::default()
    }

    fn push_child(&mut self, parent: usize, callsite_pc: Option<u64>, key: FrameKey) -> usize {
        if let Some(&idx) = self.nodes[parent].children.get(&(callsite_pc, key)) {
            return idx;
        }
        let idx = self.nodes.len();
        self.nodes.push(CctNode {
            key,
            parent: Some(parent),
            callsite_pc,
            self_cu: 0,
            pc_cu: BTreeMap::new(),
            children: BTreeMap::new(),
        });
        self.nodes[parent].children.insert((callsite_pc, key), idx);
        idx
    }

    fn push_root(&mut self, key: FrameKey) -> usize {
        self.nodes.push(CctNode {
            key,
            parent: None,
            callsite_pc: None,
            self_cu: 0,
            pc_cu: BTreeMap::new(),
            children: BTreeMap::new(),
        });
        self.nodes.len() - 1
    }

    pub fn on_instruction(&mut self, pc: u64) {
        if self.syscall_depth > 0 {
            return;
        }
        let cur = match self.current {
            Some(cur) if self.entering => {
                self.entering = false;
                let entry = self.push_child(cur, None, FrameKey::Func(pc));
                self.current = Some(entry);
                entry
            }
            Some(cur) => cur,
            None => {
                let root = self.push_root(FrameKey::Func(pc));
                self.current = Some(root);
                root
            }
        };
        self.nodes[cur].self_cu += 1;
        *self.nodes[cur].pc_cu.entry(pc).or_insert(0) += 1;
        self.attributed += 1;
    }

    pub fn on_call(&mut self, callsite_pc: u64, target_pc: u64) {
        if self.syscall_depth > 0 {
            return;
        }
        let cur = self.current.expect("call before any instruction executed");
        self.current = Some(self.push_child(cur, Some(callsite_pc), FrameKey::Func(target_pc)));
    }

    pub fn on_return(&mut self) {
        if self.syscall_depth > 0 {
            return;
        }
        let cur = self.current.expect("return before any instruction executed");
        self.current = self.nodes[cur].parent;
    }

    pub fn on_syscall_enter(&mut self, callsite_pc: u64, key: u32) {
        if self.syscall_depth == 0 {
            let cur = self.current.expect("syscall before any instruction executed");
            self.current = Some(self.push_child(cur, Some(callsite_pc), FrameKey::Syscall(key)));
            self.syscall_snapshots.push(self.attributed);
        }
        self.syscall_depth += 1;
    }

    pub fn on_syscall_exit(&mut self, cost: u64) {
        self.syscall_depth = self.syscall_depth.saturating_sub(1);
        if self.syscall_depth == 0 {
            let cur = self.current.expect("syscall exit without a current frame");
            let snapshot = self.syscall_snapshots.pop().unwrap_or(self.attributed);
            let nested = self.attributed.saturating_sub(snapshot);
            let own = cost.saturating_sub(nested);
            self.nodes[cur].self_cu += own;
            self.attributed += own;
            self.current = self.nodes[cur].parent;
        }
    }

    pub fn on_program_enter(&mut self, program_id: [u8; 32]) {
        self.program_saves
            .push((self.current, self.syscall_depth, self.entering));
        let node = match self.current {
            Some(cur) => self.push_child(cur, None, FrameKey::Program(program_id)),
            None => self.push_root(FrameKey::Program(program_id)),
        };
        self.current = Some(node);
        self.syscall_depth = 0;
        self.entering = true;
    }

    pub fn on_program_exit(&mut self) {
        if let Some((current, syscall_depth, entering)) = self.program_saves.pop() {
            self.current = current;
            self.syscall_depth = syscall_depth;
            self.entering = entering;
        }
    }

    pub fn root(&self) -> Option<usize> {
        (!self.nodes.is_empty()).then_some(0)
    }

    pub fn total_self_cu(&self) -> u64 {
        self.nodes.iter().map(|n| n.self_cu).sum()
    }

    pub fn stack_path(&self, mut idx: usize) -> Vec<FrameKey> {
        let mut path = vec![self.nodes[idx].key];
        while let Some(parent) = self.nodes[idx].parent {
            path.push(self.nodes[parent].key);
            idx = parent;
        }
        path.reverse();
        path
    }

    pub fn folded_stacks(&self) -> Vec<(Vec<FrameKey>, u64)> {
        let mut folded: BTreeMap<Vec<FrameKey>, u64> = BTreeMap::new();
        for (idx, n) in self.nodes.iter().enumerate() {
            if n.self_cu > 0 {
                *folded.entry(self.stack_path(idx)).or_insert(0) += n.self_cu;
            }
        }
        folded.into_iter().collect()
    }
}
