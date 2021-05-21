use std::collections::BTreeSet;

use crate::data_types::TensorEither;

pub mod rectified_linear;
pub mod softmax_with_loss;

pub trait FullyConnectedLayerActivation<T: Clone, const OUTPUT_DIM: usize> {
    fn apply_activation(&mut self, tensor: TensorEither<T, 1>) -> TensorEither<T, OUTPUT_DIM>;
    fn back_propagate_activation(&mut self, gradient: TensorEither<T, OUTPUT_DIM>, activation_input: TensorEither<T, 1>, activation_output: TensorEither<T, OUTPUT_DIM>) -> TensorEither<T, 1>;
    fn get_active_nodes(&self) -> BTreeSet<usize>;
    fn get_output_size(&self, pre_activation_size: [usize; 1]) -> [usize; OUTPUT_DIM];
}
