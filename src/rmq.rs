use std::{collections::HashMap, fmt::Debug};

use crate::allocated_size::AllocatedSize;

/// Range minimum query datastructure.
pub trait RMQ {
    /// Build the data structure based on the provided data.
    fn build(data: Vec<u64>) -> Self;

    /// Query the RMQ. This will return the index in `s..=e`
    /// with the lowest value in the data array.
    fn query(&self, s: usize, e: usize) -> usize;

    /// Get the size of the data structure in bits.
    fn bits(&self) -> usize;
}

/// Naive RMQ structure. Precomputes and stores all answers.
///
/// Space, time to build: O(n^2). Query time: O(1).
pub struct NaiveRMQ {
    /// Number of items in the original data array.
    items: usize,
    /// Results for all queries.
    results: Vec<usize>,
}

impl RMQ for NaiveRMQ {
    fn build(data: Vec<u64>) -> Self {
        let items = data.len();
        let mut results = vec![0; (items * (items - 1)) / 2 + items];
        let mut i = 0;
        // use dynamic programming to derive each query answer in O(1)
        // for each a, compute (a, a) (a, a+1) (a, a+2) ...
        // each step only requires to check one additional element of the data array
        for a in 0..items {
            let mut min_idx = a;
            let mut min_val = data[a];
            for b in a..items {
                if data[b] < min_val {
                    min_idx = b;
                    min_val = data[b];
                }
                results[i] = min_idx;
                i += 1;
            }
        }
        NaiveRMQ { items, results }
    }

    fn query(&self, a: usize, b: usize) -> usize {
        self.results[a * self.items + b - (a * (a + 1)) / 2]
    }

    fn bits(&self) -> usize {
        self.results.len() * 64 + 8 * std::mem::size_of::<Vec<usize>>() + 8
    }
}

/// Somewhat optimized RMQ data structure. Precomputes only some query results.
/// Queries are answered by cleverly combining these.
///
/// Space, time to build: O(n log n). Query time: O(1).
pub struct LinearLogRMQ {
    data: Vec<u64>,
    /// Table storing the results for some queries.
    /// For each index i (0 <= i < n) and length l (1 <= l < log2(n)),
    /// this stores the index of the lowest value in data[i..i+2^l].
    m: Vec<usize>,
}

impl RMQ for LinearLogRMQ {
    fn build(data: Vec<u64>) -> Self {
        let n = data.len();
        let logn = n.ilog2() as usize;

        let mut m = vec![0; n * logn];
        let mut i = 0;

        for l in 1..=logn {
            for x in 0..n {
                // to derive the query result for (x, x+2^l)
                // we simply check (x, x+2^(l-1)) and (x+2^(l-1), x+2^l)
                if l == 1 {
                    // level 1, base case: check two values in data array
                    if x == n - 1 || data[x] < data[x + 1] {
                        m[i] = x;
                    } else {
                        m[i] = x + 1;
                    }
                } else {
                    // further levels: check as described above
                    let idx_a = (l - 2) * n + x;
                    let mut idx_b = (l - 2) * n + x + 2usize.pow(l as u32 - 1);
                    // handle case where the second half of the query is outside the ddata array
                    if x + 2usize.pow(l as u32 - 1) >= data.len() {
                        idx_b = idx_a;
                    }
                    if data[m[idx_a]] < data[m[idx_b]] {
                        m[i] = m[idx_a];
                    } else {
                        m[i] = m[idx_b];
                    }
                }
                i += 1;
            }
        }

        LinearLogRMQ { data, m }
    }

    fn query(&self, s: usize, e: usize) -> usize {
        // if a data structure could get frustrated, this case would be it
        if s == e {
            return s;
        }
        let l = (e - s - 1).checked_ilog2().unwrap_or(0) as usize;
        if l == 0 {
            // b - a == 1, only need to check two elements
            if self.data[s] < self.data[e] {
                return s;
            } else {
                return e;
            }
        }
        // split query into (s, s+2^l) and (e-2^l, e)
        // this works because 2 * 2^l >= e - s (see calculation of l above)
        let n = self.data.len();
        let idx_a = (l - 1) * n + s;
        let idx_b = (l - 1) * n + e + 1 - 2usize.pow(l as u32);
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

/// Optimal (up to a constant) RMQ data structure.
/// Works by splitting the input into blocks of size (log n) / 4.
///
/// Space, time to build: O(n). Query time: O(1).
pub struct LinearRMQ {
    data: Vec<u64>,
    /// RMQ storing representatives of each block.
    inner: LinearLogRMQ,
    /// Lookup table for the position of the representative of each block.
    positions: Vec<usize>,
    /// For each block: the bit vector of its cartesian tree.
    cartesian_trees: Vec<Vec<u64>>,
    /// For each cartesian tree: the precomputed answers to RMQ queries.
    cartesian_answers: HashMap<Vec<u64>, Vec<usize>>,
    /// Block size, equal to log2(data.len()) / 4.
    s: usize,
}

impl RMQ for LinearRMQ {
    fn build(data: Vec<u64>) -> Self {
        let n = data.len();
        let s = 1.max(n.ilog2() as usize / 4);
        let mut b = vec![];
        let mut positions = vec![];
        // precompute only those cartesian trees that actually appear in the input
        let mut cartesian_answers = HashMap::new();
        let mut cartesian_trees = Vec::new();

        for block in data.chunks(s) {
            let mut min_idx = 0;
            let mut min_val = block[0];
            for i in 1..block.len() {
                if block[i] < min_val {
                    min_val = block[i];
                    min_idx = i;
                }
            }
            let cart_tree = CartesianTree::build(block);
            let (bits, _) = cart_tree.bits();
            if !cartesian_answers.contains_key(&bits) {
                let q = cart_tree.query_all();
                //panic!("{:?}", q.len());
                cartesian_answers.insert(bits.clone(), q);
            }
            cartesian_trees.push(bits);
            positions.push(s * b.len() + min_idx);
            b.push(min_val);
        }
        let inner = LinearLogRMQ::build(b);

        LinearRMQ {
            data,
            inner,
            positions,
            cartesian_trees,
            cartesian_answers,
            s,
        }
    }

    fn query(&self, s: usize, e: usize) -> usize {
        // first, query across blocks:
        // limit indices to block boundaries
        let next_a = next_multiple_of(s, self.s);
        let mut prev_b = next_multiple_of(e, self.s) - if self.s > 1 { 1 } else { 0 };
        if prev_b != e {
            prev_b -= self.s;
        }
        let idxa = next_a / self.s;
        let idxb = prev_b / self.s;
        let mut min_val = u64::MAX;
        let mut min_idx = s;
        // look up result in inner RMQ
        if idxa <= idxb {
            let result = self.inner.query(idxa, idxb);
            let idx = self.positions[result];
            min_val = self.data[idx];
            min_idx = idx;
        }

        // now, we must check the remaining elements to the left and right of
        // the queried inner blocks!
        if prev_b != e {
            // check last block
            let s = 0;
            let e = e - prev_b;
            let min_idx2 = (idxb + 1) * self.s
                + self.cartesian_answers[&self.cartesian_trees[idxb + 1]]
                    [s * self.s + e - (s * (s + 1)) / 2];
            if self.data[min_idx2] < min_val {
                min_val = self.data[min_idx2];
                min_idx = min_idx2;
            }
        }
        if s != next_a {
            // check first block
            let s = s + self.s - next_a;
            let e = self.s - 1;
            let min_idx2 = (idxa - 1) * self.s
                + self.cartesian_answers[&self.cartesian_trees[idxa - 1]]
                    [s * self.s + e - (s * (s + 1)) / 2];
            if self.data[min_idx2] < min_val {
                min_idx = min_idx2;
            }
        }

        min_idx
    }

    fn bits(&self) -> usize {
        self.inner.bits()
            + 64
            + 8 * (self.cartesian_answers.allocated_size()
                + self.cartesian_trees.allocated_size()
                + self.data.allocated_size()
                + self.positions.allocated_size())
    }
}

/// One cartesian tree.
struct CartesianTree {
    nodes: Vec<TreeNode>,
    root: usize,
}

/// Cartesian tree node.
/// This represents a certain range in the input data for which
/// it stores the index of the minimum value (min).
/// Parent, left, right fields are pointers to further nodes.
/// Left corresponds to (..min) and right corresponds to (min+1..).
#[derive(Clone, Copy, Debug)]
struct TreeNode {
    pub parent: Option<usize>,
    pub left: Option<usize>,
    pub right: Option<usize>,
    pub min: usize,
}

/// Construct all possible data permutations for cartesian trees.
/// For example, with d = [0, 0, 0] and lvl = 1 it will return:
/// [1, 2, 3]
/// [1, 3, 2]
/// [2, 1, 2]
/// [2, 3, 1]
/// [3, 2, 1]
/// Using these, all cartesian trees can be constructed.
/// Consider [2, 1, 3]: it will have the same cartesian tree as
/// [2, 1, 2]. Therefore, it is not generated by this function.
///
/// This smart construction is considerably faster than
/// the naive O(n!) method.
#[allow(unused)]
fn try_all(d: &[u64], lvl: u64) -> Vec<Vec<u64>> {
    if d.is_empty() {
        return vec![d.to_vec()];
    }
    let mut r = vec![];
    for i in 0..d.len() {
        if d[i] == 0 {
            // free spot at index i
            // remaining spaces are filled recursively
            let (a, b) = d.split_at(i);
            let b = &b[1..];
            let all_a = try_all(a, lvl + 1);
            let all_b = try_all(b, lvl + 1);
            // save all combinations
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

impl CartesianTree {
    /// Build all possible cartesian trees.
    #[allow(unused)]
    fn build_all(size: usize) -> (HashMap<Vec<u64>, CartesianTree>, usize) {
        let mut all = HashMap::new();
        let mut total = 0;

        let d = vec![0; size];
        for data in try_all(&d, 1) {
            let tree = CartesianTree::build(&data);
            let bits = tree.bits();
            all.insert(bits.0, tree);
            total += bits.1;
        }

        (all, total)
    }

    /// Build the cartesian tree for the provided data array.
    fn build(data: &[u64]) -> Self {
        let mut root = 0;

        let mut nodes = vec![];
        nodes.push(TreeNode {
            parent: None,
            left: None,
            right: None,
            min: 0,
        });
        let mut rightmost_node = 0;

        // insert each node at the rightmost possible position
        for i in 1..data.len() {
            let value = data[i];
            let mut node = nodes[rightmost_node];
            while node.right.is_some() {
                rightmost_node = node.right.unwrap();
                node = nodes[rightmost_node];
            }

            loop {
                // check the current node
                if data[node.min] <= value {
                    // our value is larger: insert a new node to the right of the current node
                    let new_node_idx = nodes.len();
                    let prev_right = nodes[rightmost_node].right;
                    nodes[rightmost_node].right = Some(new_node_idx);
                    let new_node = TreeNode {
                        parent: Some(rightmost_node),
                        left: prev_right,
                        right: None,
                        min: i,
                    };
                    nodes.push(new_node);
                    rightmost_node = new_node_idx;
                    break;
                } else {
                    // found lower minimum
                    if let Some(p) = node.parent {
                        // first, keep traversing the tree upwards
                        // until we hit the case above
                        rightmost_node = p;
                        node = nodes[rightmost_node];
                    } else {
                        // we are at the root, create a new root node
                        let new_node_idx = nodes.len();
                        nodes[rightmost_node].parent = Some(new_node_idx);
                        let new_node = TreeNode {
                            parent: None,
                            left: Some(rightmost_node),
                            right: None,
                            min: i,
                        };
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
            bv[i / 64] |= bit << (i % 64);
            i += 1;
        };

        let mut q = vec![self.root];
        while !q.is_empty() {
            let mut new_q = vec![];
            for x in q {
                let deg = self.nodes[x].left.map(|_| 1).unwrap_or(0)
                    + self.nodes[x].right.map(|_| 1).unwrap_or(0);
                // there is some optimization potential here (counting occurences): |degree 0| > |degree 1| > |degree 2|
                // unfortunately, a 1/2/3/3 encoding is not more efficient
                // than a simple 2 bit encoding
                match deg {
                    0 => {
                        append_bit(0);
                        append_bit(0);
                    }
                    1 => {
                        if self.nodes[x].left.is_some() {
                            append_bit(0);
                            append_bit(1);
                        } else {
                            append_bit(1);
                            append_bit(0);
                        }
                    }
                    2 => {
                        append_bit(1);
                        append_bit(1);
                    }
                    _ => unreachable!(),
                }
                new_q.extend(self.nodes[x].left);
                new_q.extend(self.nodes[x].right);
            }
            q = new_q;
        }

        (bv, i)
    }

    /// Range minimum query. Returns the index i (s <= i <= e) with the lowest data value data[i].
    fn query(&self, s: usize, e: usize) -> usize {
        assert!(s <= e);

        let mut node = self.nodes[self.root];

        // descend tree until we find a node where the minimum is inside our query range
        loop {
            if s <= node.min && node.min <= e {
                return node.min;
            } else {
                if node.min < s {
                    node = self.nodes[node.right.unwrap()];
                } else {
                    node = self.nodes[node.left.unwrap()];
                }
            }
        }
    }

    /// Get the results of all possible queries on this tree.
    fn query_all(&self) -> Vec<usize> {
        let mut a = vec![];
        for s in 0..self.nodes.len() {
            for e in s..self.nodes.len() {
                a.push(self.query(s, e));
            }
        }
        a
    }
}

#[test]
fn cartesian_tree() {
    let answers = vec![0, 0, 0, 1, 1, 2];
    let c = CartesianTree::build(&[0, 1, 2]);
    assert_eq!(5, c.bits().0[0]);
    check(c, answers);
    let c = CartesianTree::build(&[0, 2, 1]);
    assert_eq!(9, c.bits().0[0]);
    check(c, vec![0, 0, 0, 1, 2, 2]);
    let c = CartesianTree::build(&[1, 0, 2]);
    assert_eq!(0b11, c.bits().0[0]);
    check(c, vec![0, 1, 1, 1, 1, 2]);
    let c = CartesianTree::build(&[2, 0, 1]);
    assert_eq!(0b11, c.bits().0[0]);
    check(c, vec![0, 1, 1, 1, 1, 2]);
    let c = CartesianTree::build(&[1, 2, 0]);
    assert_eq!(6, c.bits().0[0]);
    check(c, vec![0, 0, 2, 1, 2, 2]);
    let c = CartesianTree::build(&[2, 1, 0]);
    assert_eq!(10, c.bits().0[0]);
    check(c, vec![0, 1, 2, 1, 2, 2]);
}

#[cfg(test)]
fn check(c: CartesianTree, answers: Vec<usize>) {
    let mut i = 0;
    for s in 0..c.nodes.len() {
        for e in s..c.nodes.len() {
            assert_eq!(answers[i], c.query(s, e));
            i += 1;
        }
    }
}

#[test]
fn cartesian_tree_small_bv() {
    let (map, size) = CartesianTree::build_all(8);

    assert_eq!(1430, map.len());
    assert_eq!(22880, size);

    // not optimal: only need 10.48 bits to identify tree
    // solution: save only valid cartesian trees in hash map
}

/// Calculate the next multiple of rhs that is larger or equal than x.
const fn next_multiple_of(x: usize, rhs: usize) -> usize {
    let m = x % rhs;

    if m == 0 {
        x
    } else {
        x + (rhs - m)
    }
}
