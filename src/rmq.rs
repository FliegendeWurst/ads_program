use std::{fmt::Debug, collections::{HashSet, HashMap}};

use itertools::Itertools;

pub trait RMQ {
    fn build(data: Vec<u64>) -> Self;

    fn query(&self, a: usize, b: usize) -> usize;

    fn bits(&self) -> usize;
}

pub struct NaiveRMQ {
    items: usize,
    results: Vec<usize>
}

impl RMQ for NaiveRMQ {
    fn build(data: Vec<u64>) -> Self {
        let items = data.len();
        let mut results = vec![0; (items * (items - 1)) / 2 + items];
        let mut i = 0;
        for a in 0..items {
            let mut min_idx = a;
            let mut min_val = data[a];
            for b in a..items {
                if data[b] < min_val {
                    min_idx = b;
                    min_val = data[b];
                }
                println!("{a} {b} {i}   {min_idx} {min_val}   {}", results.len());
                results[i] = min_idx;
                i += 1;
            }
        }
        NaiveRMQ {
            items,
            results
        }
    }

    fn query(&self, a: usize, b: usize) -> usize {
        self.results[a * self.items + b - (a * (a+1)) / 2]
    }

    fn bits(&self) -> usize {
        self.results.len() * 64 + 8 * std::mem::size_of::<Vec<usize>>() + 8
    }
}

pub struct LinearLogRMQ {
    n: usize,
    data: Vec<u64>,
    m: Vec<usize>
}

impl RMQ for LinearLogRMQ {
    fn build(data: Vec<u64>) -> Self {
        let n = data.len();
        let logn = n.ilog2() as usize;

        let mut m = vec![0; n * logn];
        let mut i = 0;

        for l in 0..logn {
            for x in 0..n {
                if l == 0 {
                    m[i] = x;
                } else {
                    let idx_a = (l-1) * n + x;
                    let idx_b = (l-1) * n + x + 2usize.pow(l as u32 - 1);
                    if data[m[idx_a]] < data[m[idx_b]] {
                        m[i] = idx_a;
                    } else {
                        m[i] = idx_b;
                    }
                }
                i += 1;
            }
        }
        
        LinearLogRMQ { n, data, m }
    }

    fn query(&self, a: usize, b: usize) -> usize {
        let l = (b - a - 1).checked_ilog2().unwrap_or(0) as usize;
        let idx_a = l * self.n + a;
        let idx_b = l * self.n + b + 1 - 2usize.pow(l as u32);
        if self.data[self.m[idx_a]] < self.data[self.m[idx_b]] {
            self.m[idx_a]
        } else {
            self.m[idx_b]
        }
    }

    fn bits(&self) -> usize {
        64 + (self.m.len() + self.data.len()) * 64 + 2 * 8 * std::mem::size_of::<Vec<u64>>()
    }
}

pub struct LinearRMQ {
    inner: LinearLogRMQ,
    positions: Vec<usize>
}

impl RMQ for LinearRMQ {
    fn build(data: Vec<u64>) -> Self {
        let n = data.len();
        let s = 1.max(n.ilog2() as usize / 4);
        let mut b = vec![];
        let mut positions = vec![];

        for block in data.chunks(s) {
            let mut min_idx = 0;
            let mut min_val = block[0];
            for i in 1..block.len() {
                if block[i] < min_val {
                    min_val = block[i];
                    min_idx = i;
                }
            }
            positions.push(s * b.len() + min_idx);
            b.push(min_val);
        }
        let inner = LinearLogRMQ::build(b);

        let k = 4; // data.len();
        let mut perm = vec![0; k];
        for i in 0..perm.len() {
            perm[i] = i as u64;
        }
        let mut unique_bvs = HashSet::new();
        let mut checked = 0;
        for perm in perm.into_iter().permutations(k) {
            println!("permutation {:?}", perm);
            let cart = CartesianTree::build(perm);
            let bits = cart.bits();
            println!("{:?}", bits);
            unique_bvs.insert(bits.0);
            checked += 1;
        }
        println!("checked {checked}");
        println!("unique {:?}", unique_bvs.len());

        LinearRMQ { inner, positions }
    }

    fn query(&self, a: usize, b: usize) -> usize {
        todo!()
    }

    fn bits(&self) -> usize {
        todo!()
    }
}

struct CartesianTree {
    nodes: Vec<TreeNode>,
    root: usize,
}

#[derive(Clone, Copy, Debug)]
struct TreeNode {
    pub parent: Option<usize>,
    pub left: Option<usize>,
    pub right: Option<usize>,
    pub min: usize
}

fn try_all(d: &[u64], lvl: u64) -> Vec<Vec<u64>> {
    if d.is_empty() {
        return vec![d.to_vec()];
    }
    let mut r = vec![];
    for i in 0..d.len() {
        if d[i] == 0 {
            // free spot
            let (a, b) = d.split_at(i);
            let b = &b[1..];
            let all_a = try_all(a, lvl + 1);
            let all_b = try_all(b, lvl + 1);
            for a in all_a {
                let mut new = a;
                new.push(lvl);
                for b in &all_b {
                    let mut new = new.clone();
                    new.extend_from_slice(&b);
                    r.push(new);
                }
            }
        } else {
            continue;
        }
    }
    r
}

static mut COUNTERS: [usize; 4] = [0; 4];

impl CartesianTree {
    fn build_all(size: usize) -> (HashMap<Vec<u64>, CartesianTree>, usize) {
        let mut all = HashMap::new();
        let mut total = 0;

        let d = vec![0; size];
        for data in try_all(&d, 1) {
            let tree = CartesianTree::build(data);
            let bits = tree.bits();
            all.insert(bits.0, tree);
            total += bits.1;
        }

        (all, total)
    }

    fn build(data: Vec<u64>) -> Self {
        let mut root = 0;

        let mut nodes = vec![];
        nodes.push(TreeNode { parent: None, left: None, right: None, min: 0 });
        let mut rightmost_node = 0;

        for i in 1..data.len() {
            let value = data[i];
            let mut node = nodes[rightmost_node];
            while node.right.is_some() {
                rightmost_node = node.right.unwrap();
                node = nodes[rightmost_node];
            }
            loop {
                if data[node.min] <= value {
                        let new_node_idx = nodes.len();
                        let prev_right = nodes[rightmost_node].right;
                        nodes[rightmost_node].right = Some(new_node_idx);
                        let new_node = TreeNode { parent: Some(rightmost_node), left: prev_right, right: None, min: i };
                        //println!("new node {new_node:?}");
                        nodes.push(new_node);
                        rightmost_node = new_node_idx;
                        break;
                } else {
                    // found lower minimum
                    if let Some(p) = node.parent {
                        rightmost_node = p;
                        node = nodes[rightmost_node];
                    } else {
                        let new_node_idx = nodes.len();
                        nodes[rightmost_node].parent = Some(new_node_idx);
                        let new_node = TreeNode { parent: None, left: Some(rightmost_node), right: None, min: i };
                        nodes.push(new_node);
                        rightmost_node = new_node_idx;

                        root = new_node_idx;
                        break;
                    }
                }
            }
        }

        CartesianTree { nodes, root }
    }

    fn bits(&self) -> (Vec<u64>, usize) {
        let num_bits = 2 * self.nodes.len() + 1;
        let mut bv = vec![0; num_bits / 64 + if num_bits % 64 != 0 { 1 } else { 0 }];
        let mut i = 0;
        let mut append_bit = |bit: u64| {
            println!("setting bit {i} to {bit}");
            bv[i / 64] |= bit << (i % 64);
            i += 1;
        };
        println!("root {:?}", self.nodes[self.root]);

        let mut q = vec![self.root];
        while !q.is_empty() {
            let mut new_q = vec![];
            for x in q {
                println!("processing {x} {:?}", self.nodes[x]);
                let deg = self.nodes[x].left.map(|_| 1).unwrap_or(0) + self.nodes[x].right.map(|_| 1).unwrap_or(0);
                // optimized encoding
                // degree 0: 0
                // degree 1: 10*
                // degree 2: 11
                match deg {
                    0 => {
                        append_bit(0);
                        append_bit(1);
                        unsafe { COUNTERS[0] += 1 };
                    },
                    1 => {
                        if self.nodes[x].left.is_some() {
                            append_bit(1);
                            append_bit(0);
                            unsafe { COUNTERS[1] += 1 };
                        } else {
                            append_bit(0);
                            append_bit(0);
                            unsafe { COUNTERS[2] += 1 };
                        }
                    },
                    2 => {
                        append_bit(1);
                        append_bit(1);
                        unsafe { COUNTERS[3] += 1 };
                    },
                    _ => unreachable!()
                }
                new_q.extend(self.nodes[x].left);
                new_q.extend(self.nodes[x].right);
            }
            q = new_q;
        }

        (bv, i)
    }
}

#[test]
fn cartesian_tree() {
    let c = CartesianTree::build(vec![0, 1, 2]);
    assert_eq!(45, c.bits().0[0]);
    let c = CartesianTree::build(vec![0, 2, 1]);
    assert_eq!(13, c.bits().0[0]);
    let c = CartesianTree::build(vec![1, 0, 2]);
    assert_eq!(0b11, c.bits().0[0]);
    let c = CartesianTree::build(vec![2, 0, 1]);
    assert_eq!(0b11, c.bits().0[0]);
    let c = CartesianTree::build(vec![1, 2, 0]);
    assert_eq!(41, c.bits().0[0]);
    let c = CartesianTree::build(vec![2, 1, 0]);
    assert_eq!(9, c.bits().0[0]);
}

#[test]
fn cartesian_tree_small_bv() {
    let (map, size) = CartesianTree::build_all(6);

    unsafe {
        println!("{COUNTERS:?}");
    }

    assert_eq!(42, map.len());
    // encoding 0, 100, 101, 11: 462
    // encoding two bits each: 420
    assert_eq!(42, size);
}