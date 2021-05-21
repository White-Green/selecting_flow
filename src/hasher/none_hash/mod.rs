use std::collections::BTreeSet;
use std::ops::Range;

use crate::data_types::{Dense, TensorEither};
use crate::hasher::{FullyConnectedHasher, IntoFullyConnectedHasher};

#[derive(Debug)]
pub struct NoneHash;

#[derive(Debug)]
pub struct NoneHashInstance {
    output_size: usize,
}

impl<T: Clone, P> IntoFullyConnectedHasher<T, P> for NoneHash {
    type Hasher = NoneHashInstance;

    fn into_hasher(self, _: usize, output_size: usize) -> Self::Hasher {
        NoneHashInstance { output_size }
    }
}

impl<T: Clone, P> FullyConnectedHasher<T, P> for NoneHashInstance {
    type ActiveNodesIter = Range<usize>;

    fn get_active_nodes(&self, _: TensorEither<T, 1>, _: BTreeSet<usize>) -> Self::ActiveNodesIter {
        0..self.output_size
    }

    fn rebuild(&mut self, _: &Dense<P, 2>, _: &Dense<P, 1>) {}
}
