use crate::data_types::Dense;
use crate::optimizer::adam::Adam;
use crate::optimizer::{FullyConnectedLayerOptimizer, IntoFullyConnectedLayerOptimizer};

#[test]
fn adam() {
    let mut adam = Adam::new(0.9, 0.999, 0.01).into_optimizer(1, 2);
    let mut weight = Dense::new([1, 2]);
    let weight_grad = Dense::new([1, 2]);
    for i in 0..8 {
        let mut bias = Dense::new([2]);
        bias.set([0], 10f32 * (i as f32 / 4f32).sin());
        bias.set([1], 10f32 * (i as f32 / 4f32).cos());
        let mut log = Vec::new();
        let mut bias_grad = Dense::new([2]);
        for _ in 0..10000 {
            log.push((*bias.get([0]).unwrap(), *bias.get([1]).unwrap()));
            for i in 0..2 {
                bias_grad.set([i], *bias.get([i]).unwrap() * 2f32);
            }
            adam.update(&mut weight, &mut bias, &weight_grad, &bias_grad);
        }
        let loss = bias.as_all_slice().iter().fold(0f32, |sum, &v| sum + v * v);
        assert!(loss < 1e-2, "i:{},log:{:?}", i, log);
    }
}
