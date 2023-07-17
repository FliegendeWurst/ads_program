use std::collections::{HashMap, HashSet, BTreeSet};

pub struct YFastTrie {
    x: XFastTrie,
    blocks: HashMap<u64, BTreeSet<u64>>
}

impl YFastTrie {
    pub fn build(values: &[u64]) -> Self {
        let mut x = XFastTrie::new();
        let mut blocks = HashMap::new();
        for block in values.chunks(64) {
            let rep = *block.iter().max().unwrap();
            x.add(rep);
            blocks.insert(rep, block.into_iter().copied().collect());
        }
        YFastTrie { x, blocks }
    }

    pub fn pred(&self, x: u64) -> Option<u64> {
        let block = self.x.pred(x);
        if let Some(b) = block {
            self.blocks[&b].range(..=x).next_back().copied()
        } else {
            None
        }
    }

    pub fn succ(&self, x: u64) -> Option<u64> {
        let block = self.x.succ(x);
        if let Some(b) = block {
            self.blocks[&b].range(x..).next_back().copied()
        } else {
            None
        }
    }
}

pub struct XFastTrie {
    map: HashMap<u64, TrieNode>
}

#[derive(Clone, Copy, Debug)]
struct TrieNode {
    left_min: Option<u64>,
    left_max: Option<u64>,
    right_min: Option<u64>,
    right_max: Option<u64>,
    leaves: [Option<TrieLeaf>; 2]
}

#[derive(Clone, Copy, Debug)]
struct TrieLeaf {
    value: u64,
    prev: Option<u64>,
    next: Option<u64>
}

impl XFastTrie {
    pub fn new() -> Self {
        XFastTrie { map: HashMap::new() }
    }

    pub fn add(&mut self, value: u64) {
        let mut ps = bit_prefixes(value);
        let mut prev_node = None;
        let mut right = None;
        let mut left = None;
        for (i, p) in ps.into_iter().enumerate() {
            println!("have prefix {p}");
            let bit = (value >> (63 - i)) & 1;
            if let Some(n) = self.map.get_mut(&p) {
                // check which subtree to update
                if bit == 1 {
                    n.right_min = Some(n.right_min.unwrap_or(u64::MAX).min(value));
                    n.right_max = Some(n.right_max.unwrap_or(0).max(value));
                    if n.left_max.is_some() {
                        left = n.left_max;
                    }
                    println!("picked up left {left:?}");
                } else {
                    n.left_min = Some(n.left_min.unwrap_or(u64::MAX).min(value));
                    n.left_max = Some(n.left_max.unwrap_or(0).max(value));
                    if n.right_min.is_some() {
                        right = n.right_min;
                    }
                    println!("picked up right {right:?}");
                }
                prev_node = Some(p);
                if i == 63 {
                    if bit == 1 {
                        n.leaves[1] = Some(TrieLeaf {
                            value,
                            prev: Some(n.leaves[0].unwrap().value),
                            next: right
                        });
                        let l = &n.leaves[1];
                        println!("new leaf {l:?}");
                    } else {
                        n.leaves[1] = Some(TrieLeaf {
                            value,
                            next: Some(n.leaves[0].unwrap().value),
                            prev: left
                        });
                        let l = &n.leaves[1];
                        println!("new leaf {l:?}");
                    }
                    let x = n.leaves[1].unwrap();
                        if let Some(p) = x.prev {
                            println!("updating entry for {p}");
                            let entry = self.map.get_mut(&(p | 1)).unwrap();
                            if entry.leaves[0].is_some() && entry.leaves[0].unwrap().value == p {
                                entry.leaves[0].as_mut().unwrap().next = Some(value);
                            } else {
                                entry.leaves[1].as_mut().unwrap().next = Some(value);
                            }
                        }
                        if let Some(p) = x.next {
                            println!("updating entry for {p}");
                            let entry = self.map.get_mut(&(p | 1)).unwrap();
                            if entry.leaves[0].is_some() && entry.leaves[0].unwrap().value == p {
                                entry.leaves[0].as_mut().unwrap().prev = Some(value);
                            } else {
                                entry.leaves[1].as_mut().unwrap().prev = Some(value);
                            }
                        }
                }
            } else {
                if prev_node.is_none() {
                    // first entry
                    let mut node = TrieNode {
                        left_min: if bit == 0 { Some(value) } else { None },
                        left_max: if bit == 0 { Some(value) } else { None },
                        right_min: if bit == 1 { Some(value) } else { None },
                        right_max: if bit == 1 { Some(value) } else { None },
                        leaves: if i == 63 { [Some(TrieLeaf { value, prev: None, next: None }), None] } else { [None; 2] }
                    };
                    if i == 63 {
                        if bit == 1 {
                            node.left_min = None;
                            node.left_max = None;
                        } else {
                            node.right_min = None;
                            node.right_max = None;
                        }
                    }
                    println!("inserting node ? {node:?}");
                    self.map.insert(p, node);
                } else {
                    let node = TrieNode {
                        left_min: if bit == 0 { Some(value) } else { None },
                        left_max: if bit == 0 { Some(value) } else { None },
                        right_min: if bit == 1 { Some(value) } else { None },
                        right_max: if bit == 1 { Some(value) } else { None },
                        leaves: if i == 63 { [Some(TrieLeaf { value, prev: left, next: right }), None] } else { [None; 2] }
                    };

                    // update prev/next pointers
                    if node.leaves[0].is_some() {
                        let x = node.leaves[0].unwrap();
                        if let Some(p) = x.prev {
                            println!("updating entry for {p}");
                            let entry = self.map.get_mut(&(p | 1)).unwrap();
                            if entry.leaves[0].is_some() && entry.leaves[0].unwrap().value == p {
                                entry.leaves.get_mut(0).unwrap().as_mut().unwrap().next = Some(value);
                                println!("new {:?}", entry.leaves[0]);
                            } else {
                                entry.leaves.get_mut(1).unwrap().as_mut().unwrap().next = Some(value);
                                println!("new {:?}", entry.leaves[1]);
                            }
                        }
                        if let Some(p) = x.next {
                            println!("updating entry for {p}");
                            let entry = self.map.get_mut(&(p | 1)).unwrap();
                            if entry.leaves[0].is_some() && entry.leaves[0].unwrap().value == p {
                                entry.leaves.get_mut(0).unwrap().as_mut().unwrap().prev = Some(value);
                                println!("new {:?}", entry.leaves[0]);
                            } else {
                                entry.leaves.get_mut(1).unwrap().as_mut().unwrap().prev = Some(value);
                                println!("new {:?}", entry.leaves[1]);
                            }
                        }
                    }

                    println!("inserting node {node:?}");
                    self.map.insert(p, node);
                }
            }
        }
    }

    pub fn pred(&self, value: u64) -> Option<u64> {
        let mut ps = bit_prefixes(value);
        let mut lo = 0;
        let mut hi = 64;
        let mut max = None;
        while hi - lo > 1 {
            // check middle
            let x = ps[(lo + hi) / 2];
            let bit = value >> (63 - (lo + hi) / 2) & 1;
            println!("pred: checking node {x}");
            if self.map.get(&x).is_some() {
                let x = self.map[&x];
                if bit == 1 && x.right_min.is_some() {
                    max = x.right_max;
            } else if bit == 0 && x.left_min.is_some() {
                max = x.left_max;
            }

                lo = (lo + hi) / 2;
            } else {
                hi = (lo + hi) / 2 - 1;
            }
        }
        let node = self.map.get(&ps[lo]);
        if node.is_none() {
            return None;
        }
        println!("final node {node:?} @ lo = {lo}");
        let node = node.unwrap();
        let v;
        if lo == 63 {
            // found
            if node.leaves[0].is_some() && node.leaves[0].unwrap().value == value {
                v = node.leaves[0].unwrap().prev;
            } else {
                v = node.leaves[1].map(|x| x.prev).unwrap_or(max);
            }
        } else {
            if (value >> (63 - lo)) & 1 == 1 {
                v = node.right_min.or(node.left_max);
            } else {
                v = node.left_min.or(node.right_max);
            }
        }
        return if v.unwrap_or(u64::MAX) <= value { v } else { None };
    }

    pub fn succ(&self, value: u64) -> Option<u64> {
        let mut ps = bit_prefixes(value);
        let mut lo = 0;
        let mut hi = 64;
        let mut min = None;
        while hi - lo > 1 {
            // check middle
            let x = ps[(lo + hi) / 2];
            let bit = (value >> (63 - (lo + hi) / 2)) & 1;
            println!("succ: checking node {x}, bit = {bit}");
            if self.map.get(&x).is_some() {
                let x = self.map[&x];
                println!("succ: {x:?}");

                if bit == 1 && x.right_min.is_some() {
                        min = x.right_min;
                        println!("picked up {min:?}");
                } else if bit == 0 && x.left_min.is_some() {
                    min = x.left_min;
                    println!("picked up {min:?}");
                }
                lo = (lo + hi) / 2;
            } else {
                hi = (lo + hi) / 2 - 1;
            }
        }
        let node = self.map.get(&ps[lo]);
        if node.is_none() {
            return None;
        }
        println!("final node {node:?} @ lo = {lo}");
        let node = node.unwrap();
        let v;
        if lo == 63 {
            // found
            if node.leaves[0].is_some() && node.leaves[0].unwrap().value == value {
                v = node.leaves[0].unwrap().next;
            } else {
                v = node.leaves[1].map(|x| x.next).unwrap_or(min);
            }
        } else {
            if (value >> (63 - lo)) & 1 == 1 {
                v = node.right_max;
            } else {
                v = node.left_max.or(node.right_min);
            }
        }
        return if v.unwrap_or(u64::MAX) >= value { v } else { None };
    }
}

/// Return the bit prefixes of x, in order:
/// - 1000..
/// - [bit 1]100...
/// ...
/// - [bit 1,..,62]10
/// - [bit 1,..,63]1
fn bit_prefixes(x: u64) -> Vec<u64> {
    (1..65).rev().map(move |shift| (x.checked_shr(shift).unwrap_or(0)).checked_shl(shift).unwrap_or(0) | (1 << (shift - 1))).collect()
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
