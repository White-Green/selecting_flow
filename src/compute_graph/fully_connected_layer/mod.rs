use std::cell::UnsafeCell;
use std::collections::BTreeSet;
use std::fmt::Debug;
use std::num::NonZeroUsize;
use std::ops::DerefMut;
use std::rc::Rc;
use std::sync::{Arc, RwLock};

use num_traits::{Float, NumAssign};
use rand::prelude::Distribution;
use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelRefMutIterator;

use crate::compute_graph::activation::FullyConnectedLayerActivation;
use crate::compute_graph::{read, write, ComputeGraphNode, DynNode, ExactDimensionComputeGraphNode, GraphNode, ShareCell, ValueWithSequence};
use crate::data_types::{Dense, Sparse, TensorEither, TensorEitherOwned};
use crate::hasher::{FullyConnectedHasher, IntoFullyConnectedHasher};
use crate::optimizer::{FullyConnectedLayerOptimizer, IntoFullyConnectedLayerOptimizer};

#[cfg(test)]
mod tests;

#[derive(Debug)]
pub(crate) struct FullyConnectedLayerParameter<T, H> {
    weight: Dense<T, 2>,
    bias: Dense<T, 1>,
    hasher: H,
}

#[derive(Debug)]
pub(crate) struct FullyConnectedLayerParameterGradient<T> {
    weight: Dense<T, 2>,
    bias: Dense<T, 1>,
}

impl<T: Float + NumAssign + Default + Clone, H: FullyConnectedHasher<T, T>> FullyConnectedLayerParameter<T, H> {
    fn invoke(&self, input: TensorEither<T, 1>, already_active: BTreeSet<usize>, use_hash: bool) -> TensorEitherOwned<T, 1> {
        let [input_size, output_size] = self.weight.size();
        assert_eq!(input.size(), [input_size]);
        if use_hash {
            let mut result = Sparse::new([output_size]);
            let active_nodes = self.hasher.get_active_nodes(input.as_ref(), already_active);
            match input {
                TensorEither::Dense(tensor) => {
                    for i in active_nodes {
                        let sum = self
                            .weight
                            .slice_one_line(&[i])
                            .iter()
                            .cloned()
                            .zip(tensor.slice_one_line(&[]).iter().cloned())
                            .fold(T::zero(), |acc, (a, b)| acc + a * b);
                        result.set([i], sum + *self.bias.get([i]).unwrap());
                    }
                }
                TensorEither::Sparse(tensor) => {
                    for i in active_nodes {
                        let sum = tensor.iter().fold(T::zero(), |acc, ([j], v)| acc + *v * self.weight.get([j, i]).copied().unwrap());
                        result.set([i], sum + *self.bias.get([i]).unwrap());
                    }
                }
            }
            result.into()
        } else {
            let mut result = Dense::new([output_size]);
            match input {
                TensorEither::Dense(tensor) => {
                    for i in 0..output_size {
                        let sum = self
                            .weight
                            .slice_one_line(&[i])
                            .iter()
                            .cloned()
                            .zip(tensor.slice_one_line(&[]).iter().cloned())
                            .fold(T::zero(), |acc, (a, b)| acc + a * b);
                        result.set([i], sum + *self.bias.get([i]).unwrap());
                    }
                }
                TensorEither::Sparse(tensor) => {
                    for i in 0..output_size {
                        let sum = tensor.iter().fold(T::zero(), |acc, ([j], v)| acc + *v * self.weight.get([j, i]).copied().unwrap());
                        result.set([i], sum + *self.bias.get([i]).unwrap());
                    }
                }
            }
            result.into()
        }
    }

    fn invoke_back(&self, param_grad: &mut FullyConnectedLayerParameterGradient<T>, gradient: TensorEither<T, 1>, input: TensorEither<T, 1>) -> TensorEither<T, 1> {
        // TODO: これ計算式が未証明なので確認する
        let FullyConnectedLayerParameterGradient { weight, bias } = param_grad;
        assert_eq!(bias.size(), gradient.size());
        assert_eq!(weight.size(), [input.size()[0], gradient.size()[0]]);
        match &gradient {
            TensorEither::Dense(gradient) => bias.slice_one_line_mut(&[]).iter_mut().zip(gradient.slice_one_line(&[]).iter()).for_each(|(bias, grad)| *bias += *grad),
            TensorEither::Sparse(gradient) => gradient.iter().for_each(|(k, grad)| *bias.get_mut(k).unwrap() += *grad),
        }
        match input {
            TensorEither::Dense(input) => {
                let mut result = Dense::new(input.size());
                match gradient {
                    TensorEither::Dense(gradient) => {
                        for j in 0..gradient.size()[0] {
                            let gradient = gradient.get([j]).unwrap();
                            for i in 0..input.size()[0] {
                                let input = *input.get([i]).unwrap();
                                *weight.get_mut([i, j]).unwrap() += *gradient * input;
                                *result.get_mut([i]).unwrap() += *gradient * *self.weight.get([i, j]).unwrap();
                            }
                        }
                    }
                    TensorEither::Sparse(gradient) => {
                        for ([j], gradient) in gradient.iter() {
                            for i in 0..input.size()[0] {
                                let input = *input.get([i]).unwrap();
                                *weight.get_mut([i, j]).unwrap() += *gradient * input;
                                *result.get_mut([i]).unwrap() += *gradient * *self.weight.get([i, j]).unwrap();
                            }
                        }
                    }
                }
                result.into()
            }
            TensorEither::Sparse(input) => {
                let mut result = Sparse::new(input.size());
                match gradient {
                    TensorEither::Dense(gradient) => {
                        for j in 0..gradient.size()[0] {
                            let gradient = gradient.get([j]).unwrap();
                            for ([i], input) in input.iter() {
                                *weight.get_mut([i, j]).unwrap() += *gradient * *input;
                                *result.get_or_insert_mut([i], T::zero()).unwrap() += *gradient * *self.weight.get([i, j]).unwrap();
                            }
                        }
                    }
                    TensorEither::Sparse(gradient) => {
                        for ([j], gradient) in gradient.iter() {
                            for ([i], input) in input.iter() {
                                *weight.get_mut([i, j]).unwrap() += *gradient * *input;
                                *result.get_or_insert_mut([i], T::zero()).unwrap() += *gradient * *self.weight.get([i, j]).unwrap();
                            }
                        }
                    }
                }
                result.into()
            }
        }
    }
}

impl<T: Clone, H: FullyConnectedHasher<T, T>> FullyConnectedLayerParameter<T, H> {
    fn rebuild_hash(&mut self) {
        self.hasher.rebuild(&self.weight, &self.bias);
    }
}

#[derive(Debug)]
pub struct FullyConnectedLayer<T, H, O> {
    input_size: usize,
    output_size: usize,
    parameter: Arc<RwLock<FullyConnectedLayerParameter<T, H>>>,
    gradient: Arc<UnsafeCell<FullyConnectedLayerParameterGradient<T>>>,
    optimizer: O,
}

impl<T: Float + Default + Send + Sync, H: FullyConnectedHasher<T, T>, O: FullyConnectedLayerOptimizer<T>> FullyConnectedLayer<T, H, O>
where
    StandardNormal: Distribution<T>,
{
    pub fn new_random_param(
        input_size: usize,
        output_size: usize,
        hasher: impl IntoFullyConnectedHasher<T, T, Hasher = H>,
        optimizer: impl IntoFullyConnectedLayerOptimizer<T, Optimizer = O>,
    ) -> Self {
        let hasher = hasher.into_hasher(input_size, output_size);
        let optimizer = optimizer.into_optimizer(input_size, output_size);
        let mut parameter = FullyConnectedLayerParameter {
            weight: {
                let mut weight = Dense::new([input_size, output_size]);
                let input_size = T::one() / T::from(input_size).unwrap();
                let sd = (input_size + input_size).sqrt();
                weight.as_all_slice_mut().par_iter_mut().for_each(|v| *v = thread_rng().sample(StandardNormal) * sd);
                weight
            },
            bias: {
                let mut bias = Dense::new([output_size]);
                bias.as_all_slice_mut().par_iter_mut().for_each(|v| *v = thread_rng().sample(StandardNormal));
                bias
            },
            hasher,
        };
        parameter.rebuild_hash();
        let gradient = FullyConnectedLayerParameterGradient {
            weight: Dense::new([input_size, output_size]),
            bias: Dense::new([output_size]),
        };
        FullyConnectedLayer {
            input_size,
            output_size,
            parameter: Arc::new(RwLock::new(parameter)),
            gradient: Arc::new(UnsafeCell::new(gradient)),
            optimizer,
        }
    }

    pub fn update_parameter(&mut self) {
        let FullyConnectedLayer { parameter, gradient, optimizer, .. } = self;

        let mut parameter = parameter.write().unwrap();
        let FullyConnectedLayerParameter {
            weight: param_weight,
            bias: param_bias,
            ..
        } = parameter.deref_mut();
        let FullyConnectedLayerParameterGradient {
            weight: gradient_weight,
            bias: gradient_bias,
        } = unsafe { &mut *gradient.get() };
        optimizer.update(param_weight, param_bias, gradient_weight, gradient_bias);
        gradient_weight.as_all_slice_mut().par_iter_mut().for_each(|v| *v = T::zero());
        gradient_bias.as_all_slice_mut().par_iter_mut().for_each(|v| *v = T::zero());
    }

    pub fn rebuild_hash(&mut self) {
        self.parameter.write().unwrap().rebuild_hash();
    }
}

impl<T, H, O> FullyConnectedLayer<T, H, O> {
    pub fn apply_to<I, A, const N: usize>(&self, prev_node: GraphNode<I, 1>, activation: A) -> GraphNode<ApplyFullyConnectedLayer<I, T, H, A, N>, N>
    where
        ApplyFullyConnectedLayer<I, T, H, A, N>: ExactDimensionComputeGraphNode<N>,
        I: ExactDimensionComputeGraphNode<1>,
    {
        let generation = read(&prev_node.0).generation() + 1;
        GraphNode::new(ApplyFullyConnectedLayer {
            use_hash: true,
            prev_node: prev_node.0,
            generation,
            last_used_value_sequence: None,
            output_size: self.output_size,
            pre_activation_cache: None,
            output_cache: None,
            activation,
            parameter: Arc::clone(&self.parameter),
            parameter_gradient: Arc::clone(&self.gradient),
            gradient: None,
        })
    }

    pub fn apply_unhash_to<I, A, const N: usize>(&self, prev_node: GraphNode<I, 1>, activation: A) -> GraphNode<ApplyFullyConnectedLayer<I, T, H, A, N>, N>
    where
        ApplyFullyConnectedLayer<I, T, H, A, N>: ExactDimensionComputeGraphNode<N>,
        I: ExactDimensionComputeGraphNode<1>,
    {
        let generation = read(&prev_node.0).generation() + 1;
        GraphNode::new(ApplyFullyConnectedLayer {
            use_hash: false,
            prev_node: prev_node.0,
            generation,
            last_used_value_sequence: None,
            output_size: self.output_size,
            pre_activation_cache: None,
            output_cache: None,
            activation,
            parameter: Arc::clone(&self.parameter),
            parameter_gradient: Arc::clone(&self.gradient),
            gradient: None,
        })
    }
}

unsafe impl<T, H, O> Send for FullyConnectedLayer<T, H, O> {}

unsafe impl<T, H, O> Sync for FullyConnectedLayer<T, H, O> {}

pub struct ApplyFullyConnectedLayer<I, T, H, A, const OUTPUT_DIM: usize> {
    use_hash: bool,
    prev_node: Rc<ShareCell<I>>,
    generation: usize,
    last_used_value_sequence: Option<NonZeroUsize>,
    output_size: usize,
    pre_activation_cache: Option<TensorEitherOwned<T, 1>>,
    output_cache: Option<ValueWithSequence<TensorEitherOwned<T, OUTPUT_DIM>>>,
    pub(crate) activation: A,
    parameter: Arc<RwLock<FullyConnectedLayerParameter<T, H>>>,
    parameter_gradient: Arc<UnsafeCell<FullyConnectedLayerParameterGradient<T>>>,
    gradient: Option<TensorEitherOwned<T, OUTPUT_DIM>>,
}

impl<I: 'static + ExactDimensionComputeGraphNode<1, Item = T>, T: Float + NumAssign + Default + Clone, H: FullyConnectedHasher<T, T>, A: FullyConnectedLayerActivation<T, N>, const N: usize>
    ComputeGraphNode for ApplyFullyConnectedLayer<I, T, H, A, N>
{
    fn generation(&self) -> usize {
        self.generation
    }

    fn prev_nodes(&self, next_list: &mut Vec<DynNode>) {
        let prev_node = read(&self.prev_node);
        let generation = prev_node.generation();
        next_list.push(DynNode::new(Rc::clone(&self.prev_node), generation));
    }

    fn clear_gradient(&mut self) {
        self.gradient = None;
    }

    fn apply_back_propagation(&mut self) {
        let mut prev_node = write(&self.prev_node);
        let current_gradient = self.gradient.as_ref().expect("call add_gradient before apply_back_propagation");
        let current_gradient = self.activation.back_propagate_activation(
            current_gradient.as_ref(),
            self.pre_activation_cache.as_ref().unwrap().into(),
            self.output_cache.as_ref().unwrap().value.as_ref(),
        );
        let ValueWithSequence { value: prev_output, .. } = prev_node.get_output();
        let parameter = self.parameter.read().unwrap();
        let next_gradient = parameter.invoke_back(unsafe { &mut *self.parameter_gradient.get() }, current_gradient, prev_output);
        prev_node.add_gradient(next_gradient);
    }
}

impl<I: 'static + ExactDimensionComputeGraphNode<1, Item = T>, T: Float + NumAssign + Default + Clone, H: FullyConnectedHasher<T, T>, A: FullyConnectedLayerActivation<T, N>, const N: usize>
    ExactDimensionComputeGraphNode<N> for ApplyFullyConnectedLayer<I, T, H, A, N>
{
    type Item = T;

    fn output_shape(&self) -> [usize; N] {
        self.activation.get_output_size([self.output_size])
    }

    fn get_output(&mut self) -> ValueWithSequence<TensorEither<T, N>> {
        let mut prev_node = write(&self.prev_node);
        let ValueWithSequence {
            value: prev_node_output,
            sequence: prev_sequence,
        } = prev_node.get_output();
        let need_update = self.last_used_value_sequence != Some(prev_sequence);
        self.last_used_value_sequence = Some(prev_sequence);
        let value = match self.output_cache.take() {
            Some(value) if !need_update => value,
            Some(ValueWithSequence { value: _, sequence }) => {
                let pre_activation = self.parameter.read().unwrap().invoke(prev_node_output, self.activation.get_active_nodes(), self.use_hash);
                let value = self.activation.apply_activation(pre_activation.as_ref());
                self.pre_activation_cache = Some(pre_activation);
                ValueWithSequence {
                    value: value.into_owned(),
                    sequence: NonZeroUsize::new(sequence.get() + 1).unwrap(),
                }
            }
            None => {
                let pre_activation = self.parameter.read().unwrap().invoke(prev_node_output, self.activation.get_active_nodes(), self.use_hash);
                let value = self.activation.apply_activation(pre_activation.as_ref());
                self.pre_activation_cache = Some(pre_activation);
                ValueWithSequence {
                    value: value.into_owned(),
                    sequence: NonZeroUsize::new(1).unwrap(),
                }
            }
        };
        self.output_cache = Some(value);
        let ValueWithSequence { value, sequence } = self.output_cache.as_ref().unwrap();
        ValueWithSequence {
            value: value.as_ref(),
            sequence: *sequence,
        }
    }

    fn add_gradient(&mut self, tensor: TensorEither<Self::Item, N>) {
        assert_eq!(tensor.size(), self.output_shape());
        match (&mut self.gradient, tensor) {
            (Some(this), rhs) => *this += rhs,
            (None, tensor) => self.gradient = Some(tensor.clone().into_owned()),
        }
    }
}
