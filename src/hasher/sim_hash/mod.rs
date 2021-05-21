use std::collections::{BTreeSet, HashMap};

use itertools::Itertools;
use num_traits::NumAssign;
use rand::prelude::SliceRandom;
use rand::{thread_rng, Rng};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelIterator;

use crate::data_types::{Dense, TensorEither};
use crate::hasher::buckets::Buckets;
use crate::hasher::{FullyConnectedHasher, IntoFullyConnectedHasher};

#[cfg(test)]
mod tests;

const SIGN_NEGATIVE_MASK: usize = usize::MAX & !(usize::MAX >> 1);

pub struct SimHash {
    hash_table_count: usize,
    hash_table_key_bit_depth: usize,
    bucket_size: usize,
    node_appear_threshold: usize,
    valid_value_percentage: f64,
}

impl SimHash {
    pub fn new(hash_table_count: usize, hash_table_key_bit_depth: usize, bucket_size: usize, node_appear_threshold: usize, valid_value_percentage: f64) -> Self {
        Self {
            hash_table_count,
            hash_table_key_bit_depth,
            bucket_size,
            node_appear_threshold,
            valid_value_percentage,
        }
    }
}

impl<T: Clone, P> IntoFullyConnectedHasher<T, P> for SimHash
where
    SimHashInstance: FullyConnectedHasher<T, P>,
{
    type Hasher = SimHashInstance;

    fn into_hasher(self, input_size: usize, output_size: usize) -> Self::Hasher {
        let Self {
            hash_table_count,
            hash_table_key_bit_depth,
            bucket_size,
            node_appear_threshold,
            valid_value_percentage,
        } = self;
        assert!(hash_table_key_bit_depth < 64);
        assert!(0.0 < valid_value_percentage && valid_value_percentage <= 1.0);
        let valid_value_size = (((input_size / hash_table_key_bit_depth) as f64) * valid_value_percentage).ceil() as usize;
        let valid_value_size = valid_value_size.clamp(1, input_size / hash_table_key_bit_depth) * hash_table_key_bit_depth;
        assert!(0 < valid_value_size && valid_value_size <= input_size);
        assert_eq!(valid_value_size % hash_table_key_bit_depth, 0);

        SimHashInstance {
            input_size,
            output_size,
            node_appear_threshold,
            hash_table_count,
            hash_table_key_bit_depth,
            hash_table: Dense::new([valid_value_size, hash_table_count]),
            buckets: Buckets::new(1 << hash_table_key_bit_depth, hash_table_count, bucket_size),
        }
    }
}

#[derive(Debug)]
pub struct SimHashInstance {
    input_size: usize,
    output_size: usize,
    node_appear_threshold: usize,
    hash_table_count: usize,
    hash_table_key_bit_depth: usize,
    /// [hash_value_index,hash_table_key_bit,table_index]
    hash_table: Dense<usize, 2>,
    buckets: Buckets<usize>,
}

impl SimHashInstance {
    fn get_active_nodes_inner(&self, already: BTreeSet<usize>, hasher: impl Fn(&[usize], usize) -> usize) -> BTreeSet<usize> {
        let mut counts = already.into_iter().map(|i| (i, self.node_appear_threshold)).collect::<HashMap<_, _>>();
        for hash_table_index in 0..self.hash_table_count {
            let hash_table_key = hasher(self.hash_table.slice_one_line(&[hash_table_index]), self.hash_table_key_bit_depth);
            for &i in self.buckets.get_items(hash_table_key, hash_table_index) {
                *counts.entry(i).or_default() += 1;
            }
        }
        counts
            .into_iter()
            .filter_map(|(k, v)| if v >= self.node_appear_threshold { Some(k) } else { None })
            .collect::<BTreeSet<_>>()
    }

    fn rehash_inner(&mut self) {
        let hash_table_key_bit_depth = self.hash_table_key_bit_depth;
        let hash_table_size = self.hash_table.size();
        let input_size = self.input_size;
        (0..self.hash_table_count)
            .into_par_iter()
            .zip_eq(self.hash_table.par_partial_dimension_mut(1))
            .fold(
                || (0..input_size).collect::<Vec<_>>(),
                |mut all_input_index: Vec<usize>, (_table_index, mut chunk)| {
                    let mut rng = thread_rng();
                    let mut input_index = &mut all_input_index[..];
                    let mut all_valid_index = Vec::with_capacity(hash_table_key_bit_depth);
                    for hash_table_key_bit in 0..hash_table_key_bit_depth {
                        let (current_input_index, next_input_index) =
                            input_index.split_at_mut((hash_table_key_bit + 1) * input_size / hash_table_key_bit_depth - hash_table_key_bit * input_size / hash_table_key_bit_depth);
                        input_index = next_input_index;
                        current_input_index.shuffle(&mut rng);
                        let hash_valid_indexes = &mut current_input_index[..hash_table_size[0] / hash_table_key_bit_depth];
                        hash_valid_indexes.sort_unstable();
                        all_valid_index.push(&hash_valid_indexes[..]);
                    }
                    all_valid_index
                        .into_iter()
                        .flatten()
                        .zip_eq(chunk.slice_one_line_mut(&[]))
                        .for_each(|(hash_value_index, chunk)| *chunk = if rng.gen() { *hash_value_index } else { !*hash_value_index });
                    all_input_index
                },
            )
            .for_each(drop);
    }
    fn rebuild_inner<P>(&mut self, weight: &Dense<P, 2>, _: &Dense<P, 1>)
    where
        P: NumAssign + Default + Clone + PartialOrd + Send + Sync,
    {
        assert_eq!(weight.size(), [self.input_size, self.output_size]);
        let &mut SimHashInstance {
            output_size,
            hash_table_key_bit_depth,
            ref mut hash_table,
            ref mut buckets,
            ..
        } = self;
        buckets.clear();
        for output_index in 0..output_size {
            let weight_for_one_node = weight.slice_one_line(&[output_index]);
            buckets.par_partial_buckets_mut(1).zip_eq(hash_table.par_partial_dimension(1)).for_each(|(mut buckets, hash_table)| {
                let hash_table_key = hash_table
                    .slice_one_line(&[])
                    .chunks(hash_table.size()[0] / hash_table_key_bit_depth)
                    .fold(0, |hash_table_key, hash_table| {
                        let sum = hash_table.iter().fold(P::zero(), |sum, weight_index| match weight_index & SIGN_NEGATIVE_MASK {
                            0 => sum + weight_for_one_node[*weight_index].clone(),
                            SIGN_NEGATIVE_MASK => sum - weight_for_one_node[!*weight_index].clone(),
                            _ => unreachable!(),
                        });
                        hash_table_key << 1 | if sum < P::zero() { 1 } else { 0 }
                    });
                buckets.add_item(&[hash_table_key], output_index);
            });
        }
    }
}

impl<T, P> FullyConnectedHasher<T, P> for SimHashInstance
where
    T: NumAssign + Default + Clone + PartialOrd + Send + Sync,
    P: NumAssign + Default + Clone + PartialOrd + Send + Sync,
{
    type ActiveNodesIter = BTreeSet<usize>;

    fn get_active_nodes(&self, value: TensorEither<T, 1>, already: BTreeSet<usize>) -> Self::ActiveNodesIter {
        match value {
            TensorEither::Dense(tensor) => self.get_active_nodes_inner(already, |list, bit_depth| {
                list.chunks(list.len() / bit_depth).fold(0, |acc, list| {
                    acc << 1
                        | if list.iter().fold(T::zero(), |acc, index| match index & SIGN_NEGATIVE_MASK {
                            0 => acc + tensor.get([*index]).unwrap().clone(),
                            SIGN_NEGATIVE_MASK => acc - tensor.get([!*index]).unwrap().clone(),
                            _ => unreachable!(),
                        }) < T::zero()
                        {
                            1
                        } else {
                            0
                        }
                })
            }),
            TensorEither::Sparse(tensor) => self.get_active_nodes_inner(already, |list, bit_depth| {
                list.chunks(list.len() / bit_depth).fold(0, |acc, list| {
                    acc << 1
                        | if list.iter().fold(T::zero(), |sum, index| match index & SIGN_NEGATIVE_MASK {
                            0 => tensor.get([*index]).cloned().map_or(sum.clone(), |v| sum + v),
                            SIGN_NEGATIVE_MASK => tensor.get([!*index]).cloned().map_or(sum.clone(), |v| sum - v),
                            _ => unreachable!(),
                        }) < T::zero()
                        {
                            1
                        } else {
                            0
                        }
                })
            }),
        }
    }

    fn rebuild(&mut self, weight: &Dense<P, 2>, bias: &Dense<P, 1>) {
        self.rehash_inner();
        self.rebuild_inner(weight, bias)
    }
}
