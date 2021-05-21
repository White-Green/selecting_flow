use std::collections::BTreeSet;

use num_traits::{Float, NumAssign};

use crate::compute_graph::activation::FullyConnectedLayerActivation;
use crate::compute_graph::fully_connected_layer::ApplyFullyConnectedLayer;
use crate::compute_graph::{write, ExactDimensionComputeGraphNode, GraphNode};
use crate::data_types::{Dense, Sparse, TensorEither, TensorEitherOwned};
use crate::hasher::FullyConnectedHasher;

pub struct SoftmaxWithLoss<T> {
    answer: Sparse<T, 1>,
    without_loss: TensorEitherOwned<T, 1>,
}

impl<T: Default + Float + NumAssign> Default for SoftmaxWithLoss<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Default + Float + NumAssign> SoftmaxWithLoss<T> {
    pub fn new() -> Self {
        Self {
            answer: Sparse::new([1]),
            without_loss: Sparse::new([1]).into(),
        }
    }

    fn set_expect_output(&mut self, value: Sparse<T, 1>) {
        self.answer = value;
    }
    fn get_output_without_loss(&self) -> TensorEither<T, 1> {
        self.without_loss.as_ref()
    }
}

impl<T: Default + Float + NumAssign> FullyConnectedLayerActivation<T, 0> for SoftmaxWithLoss<T> {
    fn apply_activation(&mut self, tensor: TensorEither<T, 1>) -> TensorEither<T, 0> {
        assert_eq!(self.answer.size(), tensor.size());
        let default = match tensor.into_owned() {
            TensorEitherOwned::Dense(mut tensor) => {
                let max = tensor.as_all_slice().iter().copied().fold(T::neg_infinity(), T::max);
                tensor.as_all_slice_mut().iter_mut().for_each(|v| *v = (*v - max).exp());
                let sum = tensor.as_all_slice().iter().copied().reduce(T::add).unwrap_or_else(T::zero);
                tensor.as_all_slice_mut().iter_mut().for_each(|v| *v /= sum);
                self.without_loss = tensor.into();
                (-max).exp() / sum
            }
            TensorEitherOwned::Sparse(mut tensor) => {
                let max = tensor.iter().map(|(_, &v)| v).fold(T::neg_infinity(), T::max);
                let max = if !max.is_finite() { T::zero() } else { max };
                tensor.iter_mut().for_each(|(_, v)| *v = (*v - max).exp());
                let max = (-max).exp();
                let sum = tensor.iter().map(|(_, &v)| v).fold(T::from(tensor.size()[0] - tensor.value_count()).unwrap() * max, T::add);
                tensor.iter_mut().for_each(|(_, v)| *v /= sum);
                self.without_loss = tensor.into();
                max / sum
            }
        };
        let mut loss = T::zero();
        for ([i], v) in self.answer.iter() {
            loss += *v * self.without_loss.get([i]).copied().unwrap_or(default).ln();
        }
        let mut result = Dense::new([]);
        result.set([], -loss);
        result.into()
    }

    fn back_propagate_activation(&mut self, gradient: TensorEither<T, 0>, activation_input: TensorEither<T, 1>, _activation_output: TensorEither<T, 0>) -> TensorEither<T, 1> {
        let gradient = if let Some(gradient) = gradient.get([]) {
            *gradient
        } else {
            return Sparse::new(activation_input.size()).into();
        };
        match self.without_loss.as_ref() {
            TensorEither::Dense(tensor) => {
                let mut tensor = tensor.into_owned();
                let [len] = tensor.size();
                for i in 0..len {
                    let without_loss = tensor.get_mut([i]).unwrap();
                    *without_loss = gradient * (*without_loss - self.answer.get([i]).copied().unwrap_or_else(T::zero));
                }
                tensor.into()
            }
            TensorEither::Sparse(tensor) => {
                let mut tensor = tensor.into_owned();
                for ([i], v) in self.answer.iter() {
                    *tensor.get_or_insert_mut([i], T::zero()).unwrap() -= *v;
                }
                tensor.iter_mut().for_each(|(_, v)| *v *= gradient);
                tensor.into()
            }
        }
    }

    fn get_active_nodes(&self) -> BTreeSet<usize> {
        self.answer.iter().map(|([i], _)| i).collect()
    }

    fn get_output_size(&self, _: [usize; 1]) -> [usize; 0] {
        []
    }
}

impl<I: 'static + ExactDimensionComputeGraphNode<1, Item = T>, T: Float + NumAssign + Default + Clone, H: FullyConnectedHasher<T, T>>
    GraphNode<ApplyFullyConnectedLayer<I, T, H, SoftmaxWithLoss<T>, 0>, 0>
{
    pub fn set_expect_output(&mut self, tensor: Sparse<T, 1>) {
        write(&self.0).activation.set_expect_output(tensor);
    }

    pub fn get_output_without_loss(&self) -> TensorEitherOwned<T, 1> {
        let mut node = write(&self.0);
        node.get_output();
        node.activation.get_output_without_loss().into_owned()
    }
}
