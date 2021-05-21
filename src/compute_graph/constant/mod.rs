use std::num::NonZeroUsize;

use num_traits::NumAssign;

use crate::compute_graph::{ComputeGraphNode, DynNode, ExactDimensionComputeGraphNode, ValueWithSequence};
use crate::data_types::{Dense, Sparse, TensorEither, TensorEitherOwned};

pub(crate) struct Constant<T> {
    tensor: T,
}

impl<T> Constant<T> {
    pub(crate) fn new(tensor: T) -> Self {
        Constant { tensor }
    }
}

impl<G> ComputeGraphNode for Constant<G> {
    fn generation(&self) -> usize {
        0
    }

    fn prev_nodes(&self, _: &mut Vec<DynNode>) {}

    fn clear_gradient(&mut self) {}

    fn apply_back_propagation(&mut self) {}
}

impl<T: NumAssign + Default + Clone, const N: usize> ExactDimensionComputeGraphNode<N> for Constant<TensorEitherOwned<T, N>> {
    type Item = T;

    fn output_shape(&self) -> [usize; N] {
        self.tensor.size()
    }

    fn get_output(&mut self) -> ValueWithSequence<TensorEither<T, N>> {
        ValueWithSequence::new(self.tensor.as_ref(), NonZeroUsize::new(1).unwrap())
    }

    fn add_gradient(&mut self, _: TensorEither<Self::Item, N>) {}
}

impl<T: NumAssign + Default + Clone, const N: usize> ExactDimensionComputeGraphNode<N> for Constant<Dense<T, N>> {
    type Item = T;

    fn output_shape(&self) -> [usize; N] {
        self.tensor.size()
    }

    fn get_output(&mut self) -> ValueWithSequence<TensorEither<T, N>> {
        ValueWithSequence::new((&self.tensor).into(), NonZeroUsize::new(1).unwrap())
    }

    fn add_gradient(&mut self, _: TensorEither<Self::Item, N>) {}
}

impl<T: NumAssign + Default + Clone, const N: usize> ExactDimensionComputeGraphNode<N> for Constant<Sparse<T, N>> {
    type Item = T;

    fn output_shape(&self) -> [usize; N] {
        self.tensor.size()
    }

    fn get_output(&mut self) -> ValueWithSequence<TensorEither<T, N>> {
        ValueWithSequence::new((&self.tensor).into(), NonZeroUsize::new(1).unwrap())
    }

    fn add_gradient(&mut self, _: TensorEither<Self::Item, N>) {}
}
