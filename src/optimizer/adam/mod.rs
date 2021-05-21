use num_traits::Float;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::data_types::Dense;
use crate::optimizer::{FullyConnectedLayerOptimizer, IntoFullyConnectedLayerOptimizer};

#[cfg(test)]
mod tests;

#[derive(Debug)]
pub struct Adam<T> {
    beta_1: T,
    beta_2: T,
    alpha: T,
}

impl<T: Float + Default + Send + Sync> Adam<T> {
    pub fn new(beta_1: T, beta_2: T, alpha: T) -> Self {
        Adam { beta_1, beta_2, alpha }
    }
}

#[derive(Debug, Default)]
struct AdamOptimizerCachedValue<T> {
    v: T,
    s: T,
}

#[derive(Debug)]
pub struct AdamOptimizer<T> {
    beta_1: T,
    beta_2: T,
    beta_1_t: T,
    beta_2_t: T,
    alpha: T,
    weight_cached_value: Dense<AdamOptimizerCachedValue<T>, 2>,
    bias_cached_value: Dense<AdamOptimizerCachedValue<T>, 1>,
}

impl<T: Float + Default> IntoFullyConnectedLayerOptimizer<T> for Adam<T>
where
    AdamOptimizer<T>: FullyConnectedLayerOptimizer<T>,
{
    type Optimizer = AdamOptimizer<T>;

    fn into_optimizer(self, input_size: usize, output_size: usize) -> Self::Optimizer {
        let Adam { beta_1, beta_2, alpha } = self;
        AdamOptimizer {
            beta_1,
            beta_2,
            beta_1_t: beta_1,
            beta_2_t: beta_2,
            alpha,
            weight_cached_value: Dense::new([input_size, output_size]),
            bias_cached_value: Dense::new([output_size]),
        }
    }
}

impl<T: Float + Default + Send + Sync> FullyConnectedLayerOptimizer<T> for AdamOptimizer<T> {
    fn update(&mut self, param_weight: &mut Dense<T, 2>, param_bias: &mut Dense<T, 1>, gradient_weight: &Dense<T, 2>, gradient_bias: &Dense<T, 1>) {
        let &mut AdamOptimizer {
            beta_1,
            beta_2,
            ref mut beta_1_t,
            ref mut beta_2_t,
            alpha,
            ref mut weight_cached_value,
            ref mut bias_cached_value,
        } = self;
        param_weight
            .as_all_slice_mut()
            .par_iter_mut()
            .zip_eq(gradient_weight.as_all_slice().par_iter().zip_eq(weight_cached_value.as_all_slice_mut().par_iter_mut()))
            .for_each(update_once(beta_1, beta_2, *beta_1_t, *beta_2_t, alpha));
        param_bias
            .as_all_slice_mut()
            .par_iter_mut()
            .zip_eq(gradient_bias.as_all_slice().par_iter().zip_eq(bias_cached_value.as_all_slice_mut().par_iter_mut()))
            .for_each(update_once(beta_1, beta_2, *beta_1_t, *beta_2_t, alpha));
        *beta_1_t = *beta_1_t * beta_1;
        *beta_2_t = *beta_2_t * beta_2;
    }
}

fn update_once<T: Float + Send + Sync>(beta_1: T, beta_2: T, beta_1_t: T, beta_2_t: T, alpha: T) -> impl Fn((&mut T, (&T, &mut AdamOptimizerCachedValue<T>))) + Send + Sync {
    let eps = T::from(1e-8).unwrap();
    move |(param, (grad, AdamOptimizerCachedValue { v, s })): (&mut T, (&T, &mut AdamOptimizerCachedValue<T>))| {
        if !grad.is_finite() && !grad.is_zero() {
            dbg!(grad.to_f64());
        }
        *v = beta_1 * *v + (T::one() - beta_1) * *grad;
        *s = beta_2 * *s + (T::one() - beta_2) * grad.powi(2);
        let delta = alpha * ((*v / (T::one() - beta_1_t)) / ((*s / (T::one() - beta_2_t)) + eps).sqrt());
        *param = *param - delta;
        if !param.is_finite() && !param.is_zero() {
            dbg!(param.to_f64());
        }
    }
}
