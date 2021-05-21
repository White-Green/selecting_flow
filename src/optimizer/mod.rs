use crate::data_types::Dense;

pub mod adam;

pub trait IntoFullyConnectedLayerOptimizer<T> {
    type Optimizer: FullyConnectedLayerOptimizer<T>;
    fn into_optimizer(self, input_size: usize, output_size: usize) -> Self::Optimizer;
}

pub trait FullyConnectedLayerOptimizer<T> {
    fn update(&mut self, param_weight: &mut Dense<T, 2>, param_bias: &mut Dense<T, 1>, gradient_weight: &Dense<T, 2>, gradient_bias: &Dense<T, 1>);
}
