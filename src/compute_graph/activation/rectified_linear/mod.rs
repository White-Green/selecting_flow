use std::collections::BTreeSet;

use num_traits::Float;

use crate::compute_graph::activation::FullyConnectedLayerActivation;
use crate::data_types::TensorEither;

#[allow(clippy::upper_case_acronyms)]
#[derive(Default)]
pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        ReLU
    }
}

impl<T: Default + Float> FullyConnectedLayerActivation<T, 1> for ReLU {
    fn apply_activation(&mut self, tensor: TensorEither<T, 1>) -> TensorEither<T, 1> {
        match tensor {
            TensorEither::Dense(tensor) => {
                let mut tensor = tensor.into_owned();
                let [len] = tensor.size();
                for i in 0..len {
                    let x = tensor.get_mut([i]).unwrap();
                    *x = x.max(T::zero());
                }
                tensor.into()
            }
            TensorEither::Sparse(tensor) => {
                let mut tensor = tensor.into_owned();
                let indexes = tensor.iter().filter_map(|([k], v)| if *v <= T::zero() { Some(k) } else { None }).collect::<Vec<_>>();
                for i in indexes {
                    tensor.remove([i]);
                }
                tensor.into()
            }
        }
    }

    fn back_propagate_activation(&mut self, gradient: TensorEither<T, 1>, activation_input: TensorEither<T, 1>, _: TensorEither<T, 1>) -> TensorEither<T, 1> {
        match gradient {
            TensorEither::Dense(gradient) => {
                let mut gradient = gradient.into_owned();
                match activation_input {
                    TensorEither::Dense(input) => {
                        let [len] = input.size();
                        for i in 0..len {
                            if *input.get([i]).unwrap() <= T::zero() {
                                gradient.set([i], T::zero());
                            }
                        }
                    }
                    TensorEither::Sparse(input) => {
                        for i in input.iter().filter_map(|([k], v)| if *v <= T::zero() { Some(k) } else { None }) {
                            gradient.set([i], T::zero());
                        }
                    }
                }
                gradient.into()
            }
            TensorEither::Sparse(gradient) => {
                let mut gradient = gradient.into_owned();
                match activation_input {
                    TensorEither::Dense(input) => {
                        let [len] = input.size();
                        for i in 0..len {
                            if *input.get([i]).unwrap() <= T::zero() {
                                gradient.remove([i]);
                            }
                        }
                    }
                    TensorEither::Sparse(input) => {
                        for i in input.iter().filter_map(|([k], v)| if *v <= T::zero() { Some(k) } else { None }) {
                            gradient.remove([i]);
                        }
                    }
                }
                gradient.into()
            }
        }
    }

    fn get_active_nodes(&self) -> BTreeSet<usize> {
        BTreeSet::new()
    }

    fn get_output_size(&self, pre_activation_size: [usize; 1]) -> [usize; 1] {
        pre_activation_size
    }
}
