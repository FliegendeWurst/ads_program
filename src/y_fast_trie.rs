use std::collections::{BTreeSet, HashMap};

use crate::allocated_size::AllocatedSize;

/// Trait for predecessor / successor datastructures.
pub trait PredSucc {
    /// Construct the data structure. Input array has to be sorted.
    fn build(values: &[u64]) -> Self;

    /// Get the predecessor of the given value, i.e. the highest value x such that x <= value.
    fn pred(&self, value: u64) -> Option<u64>;

    /// Get the successor of the given value, i.e. the lowest value x such that x >= value.
    fn succ(&self, value: u64) -> Option<u64>;
}

pub struct YFastTrie {
    x: XFastTrie,
    blocks: HashMap<u64, BTreeSet<u64>>,
    prev_next_rep: HashMap<u64, (Option<u64>, Option<u64>)>,
    highest_block: u64,
}

impl AllocatedSize for YFastTrie {
    fn allocated_size(&self) -> usize {
        self.x.allocated_size()
            + self.blocks.allocated_size()
            + self.prev_next_rep.capacity() * (24)
            + 8
    }
}

impl PredSucc for YFastTrie {
    fn build(values: &[u64]) -> Self {
        let mut x = XFastTrie::new();
        let mut blocks = HashMap::new();
        let mut highest_block = 0;
        let mut xs = Vec::new();
        let mut prev_rep = HashMap::new();
        for block in values.chunks(64) {
            let rep = *block.iter().max().unwrap();
            xs.push(rep);
            blocks.insert(rep, block.into_iter().copied().collect());
            highest_block = highest_block.max(rep);
        }
        for i in 0..xs.len() {
            let prev = if i > 0 { Some(xs[i - 1]) } else { None };
            let next = if i < xs.len() - 1 {
                Some(xs[i + 1])
            } else {
                None
            };
            prev_rep.insert(xs[i], (prev, next));
        }
        xs.into_iter().for_each(|y| x.add(y));
        YFastTrie {
            x,
            blocks,
            highest_block,
            prev_next_rep: prev_rep,
        }
    }

    fn pred(&self, x: u64) -> Option<u64> {
        let block = self.x.succ(x);
        if let Some(b) = block {
            self.blocks[&b]
                .range(..=x)
                .next_back()
                .copied()
                .or_else(|| self.prev_next_rep[&b].0)
        } else {
            self.blocks[&self.highest_block]
                .range(..=x)
                .next_back()
                .copied()
        }
    }

    fn succ(&self, x: u64) -> Option<u64> {
        let block = self.x.succ(x);
        if let Some(b) = block {
            self.blocks[&b].range(x..).next_back().copied()
        } else {
            None
        }
    }
}

/// Predecessor data structure.
///
/// Space: O(w n). Query time: O(log w).
pub struct XFastTrie {
    map: HashMap<u64, TrieNode>,
}

impl AllocatedSize for XFastTrie {
    fn allocated_size(&self) -> usize {
        self.map.capacity() * (8 + std::mem::size_of::<TrieNode>())
    }
}

/// Node in the x-Fast-Trie, representing a bit prefix.
/// Stores the minimum / maximum in the left / right subtree.
/// Up to two leaves may be attached.
#[derive(Clone, Copy, Debug)]
struct TrieNode {
    left_min: Option<u64>,
    left_max: Option<u64>,
    right_min: Option<u64>,
    right_max: Option<u64>,
    leaves: [Option<TrieLeaf>; 2],
}

/// Leaf of the x-Fast-Trie, representing a stored value.
/// Maintains the value of the previous and next value in the sorted data.
#[derive(Clone, Copy, Debug)]
struct TrieLeaf {
    value: u64,
    prev: Option<u64>,
    next: Option<u64>,
}

impl XFastTrie {
    pub fn new() -> Self {
        XFastTrie {
            map: HashMap::new(),
        }
    }

    pub fn add(&mut self, value: u64) {
        let ps = bit_prefixes(value);
        let mut prev_node = None;
        let mut right = None;
        let mut left = None;
        for (i, p) in ps.into_iter().enumerate() {
            let bit = (value >> (63 - i)) & 1;
            if let Some(n) = self.map.get_mut(&p) {
                // check which subtree to update
                if bit == 1 {
                    n.right_min = Some(n.right_min.unwrap_or(u64::MAX).min(value));
                    n.right_max = Some(n.right_max.unwrap_or(0).max(value));
                    if n.left_max.is_some() {
                        left = n.left_max;
                    }
                } else {
                    n.left_min = Some(n.left_min.unwrap_or(u64::MAX).min(value));
                    n.left_max = Some(n.left_max.unwrap_or(0).max(value));
                    if n.right_min.is_some() {
                        right = n.right_min;
                    }
                }
                prev_node = Some(p);
                if i == 63 {
                    // final level: update leaf node
                    if bit == 1 {
                        n.leaves[1] = Some(TrieLeaf {
                            value,
                            prev: Some(n.leaves[0].unwrap().value),
                            next: right,
                        });
                    } else {
                        n.leaves[1] = Some(TrieLeaf {
                            value,
                            next: Some(n.leaves[0].unwrap().value),
                            prev: left,
                        });
                    }
                    // update prev/next pointers
                    let x = n.leaves[1].unwrap();
                    if let Some(p) = x.prev {
                        let entry = self.map.get_mut(&(p | 1)).unwrap();
                        if entry.leaves[0].is_some() && entry.leaves[0].unwrap().value == p {
                            entry.leaves[0].as_mut().unwrap().next = Some(value);
                        } else {
                            entry.leaves[1].as_mut().unwrap().next = Some(value);
                        }
                    }
                    if let Some(p) = x.next {
                        let entry = self.map.get_mut(&(p | 1)).unwrap();
                        if entry.leaves[0].is_some() && entry.leaves[0].unwrap().value == p {
                            entry.leaves[0].as_mut().unwrap().prev = Some(value);
                        } else {
                            entry.leaves[1].as_mut().unwrap().prev = Some(value);
                        }
                    }
                }
            } else {
                // create new node
                if prev_node.is_none() {
                    // root node
                    let node = TrieNode {
                        left_min: if bit == 0 { Some(value) } else { None },
                        left_max: if bit == 0 { Some(value) } else { None },
                        right_min: if bit == 1 { Some(value) } else { None },
                        right_max: if bit == 1 { Some(value) } else { None },
                        leaves: if i == 63 {
                            [
                                Some(TrieLeaf {
                                    value,
                                    prev: None,
                                    next: None,
                                }),
                                None,
                            ]
                        } else {
                            [None; 2]
                        },
                    };
                    self.map.insert(p, node);
                } else {
                    // new node as child of existing node
                    let node = TrieNode {
                        left_min: if bit == 0 { Some(value) } else { None },
                        left_max: if bit == 0 { Some(value) } else { None },
                        right_min: if bit == 1 { Some(value) } else { None },
                        right_max: if bit == 1 { Some(value) } else { None },
                        leaves: if i == 63 {
                            [
                                Some(TrieLeaf {
                                    value,
                                    prev: left,
                                    next: right,
                                }),
                                None,
                            ]
                        } else {
                            [None; 2]
                        },
                    };

                    // update prev/next pointers
                    if node.leaves[0].is_some() {
                        let x = node.leaves[0].unwrap();
                        if let Some(p) = x.prev {
                            let entry = self.map.get_mut(&(p | 1)).unwrap();
                            if entry.leaves[0].is_some() && entry.leaves[0].unwrap().value == p {
                                entry.leaves.get_mut(0).unwrap().as_mut().unwrap().next =
                                    Some(value);
                            } else {
                                entry.leaves.get_mut(1).unwrap().as_mut().unwrap().next =
                                    Some(value);
                            }
                        }
                        if let Some(p) = x.next {
                            let entry = self.map.get_mut(&(p | 1)).unwrap();
                            if entry.leaves[0].is_some() && entry.leaves[0].unwrap().value == p {
                                entry.leaves.get_mut(0).unwrap().as_mut().unwrap().prev =
                                    Some(value);
                            } else {
                                entry.leaves.get_mut(1).unwrap().as_mut().unwrap().prev =
                                    Some(value);
                            }
                        }
                    }
                    self.map.insert(p, node);
                }
            }
        }
    }

    #[allow(unused)]
    pub fn pred(&self, value: u64) -> Option<u64> {
        // see succ for comments, this method works the same way
        let calc_prefix = |shift: u32| {
            let shift = 64 - shift;
            (value.checked_shr(shift).unwrap_or(0))
                .checked_shl(shift)
                .unwrap_or(0)
                | (1 << (shift - 1))
        };
        let mut lo = 0;
        let mut hi = 64;
        while hi - lo > 1 {
            // check middle
            let x = calc_prefix((lo + hi) / 2);
            if self.map.get(&x).is_some() {
                let x = self.map[&x];

                let lowest_val = x
                    .left_min
                    .or(x.left_max)
                    .or(x.right_min)
                    .or(x.right_max)
                    .unwrap();
                let highest_val = x
                    .right_max
                    .or(x.right_min)
                    .or(x.left_max)
                    .or(x.left_min)
                    .unwrap();
                if lowest_val <= value && highest_val >= value {
                    lo = (lo + hi) / 2;
                } else {
                    hi = (lo + hi) / 2;
                }
            } else {
                hi = (lo + hi) / 2;
            }
        }
        let node = self.map.get(&calc_prefix(lo));
        if node.is_none() {
            return None;
        }
        let node = node.unwrap();
        let v;
        if lo == 63 {
            // found
            if node.leaves[0].is_some() && node.leaves[0].unwrap().value == value {
                v = node.leaves[0].unwrap().prev;
            } else {
                v = node.leaves[1].map(|x| x.prev).flatten();
            }
        } else {
            let mut values = vec![];
            values.extend(node.left_min);
            values.extend(node.left_max);
            values.extend(node.right_min);
            values.extend(node.right_max);
            for val in values.into_iter().rev() {
                if val <= value {
                    return Some(val);
                }
            }
            v = None
        }
        return if v.unwrap_or(u64::MAX) <= value {
            v
        } else {
            None
        };
    }

    pub fn succ(&self, value: u64) -> Option<u64> {
        let ps = bit_prefixes(value);
        let mut lo = 0;
        let mut hi = 64;
        while hi - lo > 1 {
            // check middle
            let x = ps[(lo + hi) / 2];
            // node is present:
            if self.map.get(&x).is_some() {
                let x = self.map[&x];

                // if the interval represented by the node contains our value: keep descending the tree
                let lowest_val = x
                    .left_min
                    .or(x.left_max)
                    .or(x.right_min)
                    .or(x.right_max)
                    .unwrap();
                let highest_val = x
                    .right_max
                    .or(x.right_min)
                    .or(x.left_max)
                    .or(x.left_min)
                    .unwrap();
                if lowest_val <= value && highest_val >= value {
                    lo = (lo + hi) / 2;
                } else {
                    // otherwise: stay in the first half of the tree
                    hi = (lo + hi) / 2;
                }
            } else {
                // node not present: stay in first half of tree
                hi = (lo + hi) / 2;
            }
        }
        let node = self.map.get(&ps[lo]);
        if node.is_none() {
            return None;
        }
        let node = node.unwrap();
        let v;
        if lo == 63 {
            // found
            if node.leaves[0].is_some() && node.leaves[0].unwrap().value == value {
                v = node.leaves[0].unwrap().next;
            } else {
                v = node.leaves[1].map(|x| x.next).flatten();
            }
        } else {
            let mut values = vec![];
            values.extend(node.left_min);
            values.extend(node.left_max);
            values.extend(node.right_min);
            values.extend(node.right_max);
            for x in values {
                if x >= value {
                    return Some(x);
                }
            }
            v = None;
        }
        if v.unwrap_or(u64::MAX) >= value {
            v
        } else {
            None
        }
    }
}

/// Return the bit prefixes of x, in order:
/// - 1000..
/// - [bit 1]100...
/// ...
/// - [bit 1,..,62]10
/// - [bit 1,..,63]1
fn bit_prefixes(x: u64) -> Vec<u64> {
    (1..65)
        .rev()
        .map(move |shift| {
            (x.checked_shr(shift).unwrap_or(0))
                .checked_shl(shift)
                .unwrap_or(0)
                | (1 << (shift - 1))
        })
        .collect()
}

#[test]
fn x_fast_trie() {
    let mut m = XFastTrie::new();

    m.add(15);
    assert_eq!(None, m.pred(15));
    assert_eq!(None, m.succ(15));

    m.add(23);
    assert_eq!(None, m.pred(15));
    assert_eq!(Some(23), m.succ(15));
    assert_eq!(15, m.pred(23).unwrap());
    assert_eq!(None, m.succ(23));

    m.add(127);
    assert_eq!(None, m.pred(15));
    assert_eq!(Some(23), m.succ(15));
    assert_eq!(15, m.pred(23).unwrap());
    assert_eq!(Some(127), m.succ(23));
    assert_eq!(23, m.pred(127).unwrap());
    assert_eq!(None, m.succ(127));

    m.add(52);
    assert_eq!(Some(15), m.succ(13));
    assert_eq!(Some(15), m.succ(14));
    assert_eq!(None, m.pred(15));
    assert_eq!(Some(23), m.succ(15));
    assert_eq!(15, m.pred(23).unwrap());
    assert_eq!(Some(52), m.succ(23));
    assert_eq!(23, m.pred(52).unwrap());
    assert_eq!(52, m.pred(53).unwrap());
    assert_eq!(Some(127), m.succ(52));
    assert_eq!(Some(127), m.succ(126));
    assert_eq!(52, m.pred(127).unwrap());
    assert_eq!(127, m.pred(128).unwrap());
    assert_eq!(None, m.succ(127));
    assert_eq!(None, m.succ(128));
    assert_eq!(None, m.succ(129));
    assert_eq!(None, m.succ(250));

    m.add(128);
    assert_eq!(None, m.succ(129));
    assert_eq!(None, m.succ(250));
}

#[test]
fn x_fast_trie2() {
    let mut m = XFastTrie::new();

    m.add(0);
    m.add(1);
    m.add(4);
    m.add(7);

    assert_eq!(Some(1), m.pred(2));
}
