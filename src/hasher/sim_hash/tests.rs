use std::collections::BTreeSet;

use rand::{thread_rng, Rng};

use crate::data_types::Dense;
use crate::hasher::sim_hash::SimHash;
use crate::hasher::{FullyConnectedHasher, IntoFullyConnectedHasher};

#[test]
fn sim_hash() {
    let mut instance = IntoFullyConnectedHasher::<f64, f64>::into_hasher(SimHash::new(50, 4, 16, 1, 1.0), 4, 16);
    let mut weight = Dense::new([4, 16]);
    let bias = Dense::new([16]);
    for i in 0..16 {
        for j in 0..4 {
            weight.set([j, i], if (i >> j & 1) == 1 { 1.0 } else { -1.0 });
        }
    }
    FullyConnectedHasher::<f64, _>::rebuild(&mut instance, &weight, &bias);
    let mut rng = thread_rng();
    for _ in 0..10000 {
        let mut input = Dense::new([4]);
        let mut out_index = 0;
        for i in 0..4 {
            let x = rng.gen_range(-1.0..1.0);
            out_index |= if x > 0.0 { 1 } else { 0 } << i;
            input.set([i], x);
        }
        let active_nodes = FullyConnectedHasher::<f64, f64>::get_active_nodes(&instance, (&input).into(), BTreeSet::new());
        assert!(
            active_nodes.contains(&out_index)
                || active_nodes.contains(&(out_index ^ 1))
                || active_nodes.contains(&(out_index ^ 2))
                || active_nodes.contains(&(out_index ^ 4))
                || active_nodes.contains(&(out_index ^ 8)),
            "{:?},{},{:?}",
            input,
            out_index,
            active_nodes
        );
    }
}
