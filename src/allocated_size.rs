use std::{collections::HashMap, hash::Hash};

pub trait AllocatedSize {
    fn allocated_size(&self) -> usize;
}

macro_rules! imp_alloc {
    ($t:ty, $s:expr) => {
        impl AllocatedSize for $t {
            fn allocated_size(&self) -> usize {
                let x = &self;
                $s(x)
            }
        }
    };
}

imp_alloc!(u64, |_| 8);
imp_alloc!(usize, |_| 8);
imp_alloc!(Vec<u64>, |s: &Vec<u64>| s.capacity() * 8 + 24);
imp_alloc!(Vec<usize>, |s: &Vec<usize>| s.capacity() * 8 + 24);
imp_alloc!(Vec<Vec<u64>>, |s: &Vec<Vec<u64>>| s.iter().map(|x| x.allocated_size()).sum::<usize>() + 24);

impl<K, V> AllocatedSize for HashMap<K, V> 
where K: AllocatedSize + Eq + Hash, V: AllocatedSize {
    fn allocated_size(&self) -> usize {
        // every element in the map directly owns its key and its value
        let ELEMENT_SIZE: usize = if self.len() == 0 { 0 } else {
            let key = self.keys().next().unwrap();
            std::mem::size_of_val(key) + std::mem::size_of_val(&self[&key])
        };

        // directly owned allocation
        // NB: self.capacity() may be an underestimate, see its docs
        // NB: also ignores control bytes, see hashbrown implementation
        let directly_owned = (self.capacity() - self.len()) * ELEMENT_SIZE;

        // transitively owned allocations
        let transitively_owned: usize = self
            .iter()
            .map(|(key, val)| key.allocated_size() + val.allocated_size())
            .sum();

        directly_owned + transitively_owned
    }
}