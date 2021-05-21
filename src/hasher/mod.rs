use std::collections::BTreeSet;

use crate::data_types::{Dense, TensorEither};

pub(crate) mod buckets;
pub mod densified_winner_take_all;
pub mod none_hash;
pub mod sim_hash;

pub trait IntoFullyConnectedHasher<T: Clone, P> {
    type Hasher: FullyConnectedHasher<T, P>;
    fn into_hasher(self, input_size: usize, output_size: usize) -> Self::Hasher;
}

pub trait FullyConnectedHasher<T: Clone, P> {
    type ActiveNodesIter: IntoIterator<Item = usize>;
    fn get_active_nodes(&self, value: TensorEither<T, 1>, already: BTreeSet<usize>) -> Self::ActiveNodesIter;
    fn rebuild(&mut self, weight: &Dense<P, 2>, bias: &Dense<P, 1>);
}
