use std::num::NonZeroUsize;

use num_traits::NumAssign;

use crate::compute_graph::{write, ComputeGraphNode, DynNode, ExactDimensionComputeGraphNode, GraphNode, ValueWithSequence};
use crate::data_types::{Sparse, TensorEither, TensorEitherOwned};

pub struct InputBox<T, const N: usize> {
    size: [usize; N],
    input_value: TensorEitherOwned<T, N>,
    gradient: Option<TensorEitherOwned<T, N>>,
    sequence: NonZeroUsize,
}

impl<T: NumAssign + Default + Clone, const N: usize> InputBox<T, N> {
    pub fn new(size: [usize; N]) -> GraphNode<Self, N> {
        GraphNode::new(InputBox {
            size,
            input_value: Sparse::new(size).into(),
            gradient: None,
            sequence: NonZeroUsize::new(1).unwrap(),
        })
    }
}

impl<T: NumAssign + Default + Clone, const N: usize> GraphNode<InputBox<T, N>, N> {
    pub fn set_value(&mut self, value: TensorEitherOwned<T, N>) {
        let mut input_box = write(&self.0);
        assert_eq!(input_box.size, value.size());
        input_box.input_value = value;
        input_box.sequence = NonZeroUsize::new(input_box.sequence.get() + 1).unwrap();
    }
}

impl<T: NumAssign + Default + Clone, const N: usize> ComputeGraphNode for InputBox<T, N> {
    fn generation(&self) -> usize {
        0
    }

    fn prev_nodes(&self, _: &mut Vec<DynNode>) {}

    fn clear_gradient(&mut self) {
        self.gradient = None;
    }

    fn apply_back_propagation(&mut self) {}
}

impl<T: NumAssign + Default + Clone, const N: usize> ExactDimensionComputeGraphNode<N> for InputBox<T, N> {
    type Item = T;

    fn output_shape(&self) -> [usize; N] {
        self.size
    }

    fn get_output(&mut self) -> ValueWithSequence<TensorEither<Self::Item, N>> {
        ValueWithSequence::new(self.input_value.as_ref(), self.sequence)
    }

    fn add_gradient(&mut self, tensor: TensorEither<Self::Item, N>) {
        match &mut self.gradient {
            Some(gradient) => *gradient += tensor,
            None => self.gradient = Some(tensor.clone().into_owned()),
        }
    }
}
