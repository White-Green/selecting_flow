use std::collections::{BTreeSet, HashMap};

use num_traits::Float;
use rand::prelude::SliceRandom;
use rand::thread_rng;
use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};

use crate::data_types::{Dense, TensorEither};
use crate::hasher::buckets::Buckets;
use crate::hasher::{FullyConnectedHasher, IntoFullyConnectedHasher};

pub struct DWTAHash {
    hash_table_count: usize,
    hash_table_key_bit_depth: usize,
    bin_size_bit_depth: usize,
    bucket_size: usize,
    node_appear_threshold: usize,
}

impl DWTAHash {
    pub fn new(hash_table_count: usize, hash_table_key_bit_depth: usize, bin_size_bit_depth: usize, bucket_size: usize, node_appear_threshold: usize) -> Self {
        DWTAHash {
            hash_table_count,
            hash_table_key_bit_depth,
            bin_size_bit_depth,
            bucket_size,
            node_appear_threshold,
        }
    }
}

impl<T: Clone, P> IntoFullyConnectedHasher<T, P> for DWTAHash
where
    DWTAHashInstance: FullyConnectedHasher<T, P>,
{
    type Hasher = DWTAHashInstance;

    fn into_hasher(self, input_size: usize, _: usize) -> Self::Hasher {
        let DWTAHash {
            hash_table_count,
            hash_table_key_bit_depth,
            bin_size_bit_depth,
            bucket_size,
            node_appear_threshold,
        } = self;
        let hash_count_per_hash_table_key = (hash_table_key_bit_depth + bin_size_bit_depth - 1) / bin_size_bit_depth;
        let hash_count = hash_table_count * hash_count_per_hash_table_key;
        DWTAHashInstance {
            hash_table_count,
            hash_table_key_bit_depth,
            bin_size_bit_depth,
            node_appear_threshold,
            hash_count_per_hash_table_key,
            hash_count,
            hash_indexes: Dense::new([input_size, (hash_count + input_size - 1) / input_size]),
            buckets: Buckets::new(1 << hash_table_key_bit_depth, hash_table_count, bucket_size),
        }
    }
}

#[derive(Debug, Default, Clone)]
struct HashIndexItem {
    bin_index: usize,
    index_in_bin: usize,
}

pub struct DWTAHashInstance {
    hash_table_count: usize,
    hash_table_key_bit_depth: usize,
    bin_size_bit_depth: usize,
    node_appear_threshold: usize,
    hash_count_per_hash_table_key: usize,
    hash_count: usize,
    hash_indexes: Dense<HashIndexItem, 2>,
    buckets: Buckets<usize>,
}

impl DWTAHashInstance {
    fn get_hash_values_by_slice<T: Float>(&self, input: &[T]) -> Vec<usize> {
        self.hash_indexes
            .partial_dimension(1)
            .fold(vec![(0, T::neg_infinity()); self.hash_count], |hashes, slice| {
                input
                    .iter()
                    .zip(slice.slice_one_line(&[]))
                    .fold(hashes, |mut hashes, (tensor_value, HashIndexItem { bin_index, index_in_bin })| {
                        if let Some((key, value)) = hashes.get_mut(*bin_index) {
                            if *value < *tensor_value {
                                *key = *index_in_bin;
                                *value = *tensor_value;
                            }
                        }
                        hashes
                    })
            })
            .into_iter()
            .map(|(i, _)| i)
            .collect::<Vec<_>>()
    }

    fn get_active_nodes_inner<T: Float>(&self, value: TensorEither<T, 1>, already: BTreeSet<usize>) -> BTreeSet<usize> {
        let hash_values = match value {
            TensorEither::Dense(tensor) => self.get_hash_values_by_slice(tensor.slice_one_line(&[])),
            TensorEither::Sparse(tensor) => {
                let mut hash_values = self.hash_indexes.partial_dimension(1).fold(vec![None; self.hash_count], |hashes, slice| {
                    tensor.iter().fold(hashes, |mut hashes, ([i], tensor_value)| {
                        let HashIndexItem { bin_index, index_in_bin } = slice.get(&[i]).unwrap();
                        if let Some(hash) = hashes.get_mut(*bin_index) {
                            match *hash {
                                None => *hash = Some((*index_in_bin, *tensor_value)),
                                Some((_, value)) if value < *tensor_value => *hash = Some((*index_in_bin, *tensor_value)),
                                _ => {}
                            }
                        }
                        hashes
                    })
                });
                if hash_values.iter().all(Option::is_none) {
                    hash_values[0] = Some((0, T::zero()));
                }
                let hash_count = hash_values.len();
                let hash = |i: usize, attempt: usize| (i + attempt * 1_000_000_009) % hash_count;
                for i in 0..hash_values.len() {
                    for attempt in 1.. {
                        if hash_values[i].is_some() {
                            break;
                        }
                        hash_values[i] = hash_values[hash(i, attempt)].map(|(h, v)| ((h + i) & ((1 << self.bin_size_bit_depth) - 1), v));
                    }
                }
                hash_values.into_iter().map(|h| h.unwrap().0).collect()
            }
        };
        let map_init = already.into_iter().map(|i| (i, self.node_appear_threshold)).collect::<HashMap<_, _>>();
        let node_appear_count = hash_values
            .chunks(self.hash_count_per_hash_table_key)
            .map(|hashes| hashes.iter().fold(0, |acc, h| acc << self.bin_size_bit_depth | *h) & ((1 << self.hash_table_key_bit_depth) - 1))
            .zip(self.buckets.partial_buckets(1))
            .fold(map_init, |mut map, (hash_table_key, table)| {
                for node_index in table.get_items(&[hash_table_key]) {
                    *map.entry(*node_index).or_default() += 1;
                }
                map
            });
        node_appear_count
            .into_iter()
            .filter_map(|(node_index, count)| if count >= self.node_appear_threshold { Some(node_index) } else { None })
            .collect::<BTreeSet<_>>()
    }

    fn rehash_inner(&mut self) {
        self.hash_indexes
            .as_all_slice_mut()
            .par_chunks_mut(1 << self.bin_size_bit_depth)
            .enumerate()
            .for_each(|(bin_index, chunk)| chunk.iter_mut().enumerate().for_each(|(index_in_bin, item)| *item = HashIndexItem { bin_index, index_in_bin }));
        self.hash_indexes.par_partial_dimension_mut(1).for_each(|mut a| a.slice_one_line_mut(&[]).shuffle(&mut thread_rng()));
    }

    fn rebuild_inner<P: Float + Default + Send + Sync>(&mut self, weight: &Dense<P, 2>, _: &Dense<P, 1>) {
        self.buckets.clear();
        let bucket_indexes = weight
            .par_partial_dimension(1)
            .enumerate()
            .fold(
                || vec![Vec::new(); self.hash_table_count],
                |mut bucket_indexes, (output_index, one_line)| {
                    let hash_values = self.get_hash_values_by_slice(one_line.slice_one_line(&[]));
                    hash_values
                        .chunks(self.hash_count_per_hash_table_key)
                        .map(|hashes| hashes.iter().fold(0, |acc, h| acc << self.bin_size_bit_depth | *h) & ((1 << self.hash_table_key_bit_depth) - 1))
                        .zip(bucket_indexes.iter_mut())
                        .for_each(|(i, bucket_index)| bucket_index.push((output_index, i)));
                    bucket_indexes
                },
            )
            .reduce_with(|mut a, b| {
                a.iter_mut().zip(b).for_each(|(a, b)| a.extend(b));
                a
            })
            .unwrap();
        let hash_table_key_bit_depth = self.hash_table_key_bit_depth;
        self.buckets.par_partial_buckets_mut(1).zip(bucket_indexes).for_each(|(mut bucket, indexes)| {
            for (output_index, hash_index) in indexes {
                bucket.add_item(&[hash_index], output_index);
                for i in 0.. {
                    let hash_index = hash_index ^ (1 << i);
                    if hash_index < 1 << hash_table_key_bit_depth {
                        bucket.add_item(&[hash_index], output_index);
                        for i in i + 1.. {
                            let hash_index = hash_index ^ (1 << i);
                            if hash_index < 1 << hash_table_key_bit_depth {
                                bucket.add_item(&[hash_index], output_index);
                            } else {
                                break;
                            }
                        }
                    } else {
                        break;
                    }
                }
            }
        });
    }
}

impl<T, P> FullyConnectedHasher<T, P> for DWTAHashInstance
where
    T: Float,
    P: Float + Default + Send + Sync,
{
    type ActiveNodesIter = BTreeSet<usize>;

    fn get_active_nodes(&self, value: TensorEither<T, 1>, already: BTreeSet<usize>) -> Self::ActiveNodesIter {
        self.get_active_nodes_inner(value, already)
    }

    fn rebuild(&mut self, weight: &Dense<P, 2>, bias: &Dense<P, 1>) {
        self.rehash_inner();
        self.rebuild_inner(weight, bias);
    }
}
