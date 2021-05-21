use std::collections::BTreeSet;
use std::ops::DerefMut;

use rand::prelude::SliceRandom;
use rand::thread_rng;

use crate::compute_graph::fully_connected_layer::{FullyConnectedLayer, FullyConnectedLayerParameter, FullyConnectedLayerParameterGradient};
use crate::data_types::{Dense, Sparse, TensorEitherOwned};
use crate::hasher::none_hash::NoneHash;
use crate::optimizer::adam::Adam;

#[test]
fn forward() {
    let FullyConnectedLayer {
        input_size, output_size, parameter, ..
    } = FullyConnectedLayer::new_random_param(8, 1 << 8, NoneHash, Adam::new(0.0, 0.0, 0.0));
    assert_eq!(input_size, 8);
    assert_eq!(output_size, 1 << 8);
    let mut param = parameter.write().unwrap();
    let FullyConnectedLayerParameter { weight, bias, .. } = param.deref_mut();
    assert_eq!(weight.size(), [8, 1 << 8]);
    assert_eq!(bias.size(), [1 << 8]);
    for i in 0..output_size {
        let slice = weight.slice_one_line_mut(&[i]);
        assert_eq!(slice.len(), input_size);
        for j in 0..input_size {
            slice[j] = if ((i >> j) & 1) == 1 { -((1 << j) as f64) } else { (1 << j) as f64 };
        }
    }
    let slice = bias.slice_one_line_mut(&[]);
    assert_eq!(slice.len(), output_size);
    for i in 0..output_size {
        slice[i] = i as f64;
    }

    let mut input = Dense::new([8]);
    let slice = input.as_all_slice_mut();
    assert_eq!(slice.len(), 8);
    for i in 0..8 {
        slice[i] = (1 << i) as f64;
    }

    let output = param.invoke((&input).into(), BTreeSet::new(), true);
    assert_eq!(output.size(), [output_size]);
    for i in 0..1 << 8 {
        let mut sum = i as i32;
        for j in 0..8 {
            sum += (1 << (2 * j)) * if ((i >> j) & 1) == 1 { -1 } else { 1 };
        }
        assert_eq!(*output.get([i]).unwrap(), sum as f64);
    }

    let output = param.invoke(input.into(), BTreeSet::new(), true);
    assert_eq!(output.size(), [output_size]);
    for i in 0..1 << 8 {
        let mut sum = i as i32;
        for j in 0..8 {
            sum += (1 << (2 * j)) * if ((i >> j) & 1) == 1 { -1 } else { 1 };
        }
        assert_eq!(*output.get([i]).unwrap(), sum as f64);
    }

    let mut input = Sparse::new([8]);
    for i in 0..8 {
        input.set([i], (1 << i) as f64);
    }

    let output = param.invoke((&input).into(), BTreeSet::new(), true);
    assert_eq!(output.size(), [output_size]);
    for i in 0..1 << 8 {
        let mut sum = i as i32;
        for j in 0..8 {
            sum += (1 << (2 * j)) * if ((i >> j) & 1) == 1 { -1 } else { 1 };
        }
        assert_eq!(*output.get([i]).unwrap(), sum as f64);
    }

    let output = param.invoke(input.into(), BTreeSet::new(), true);
    assert_eq!(output.size(), [output_size]);
    for i in 0..1 << 8 {
        let mut sum = i as i32;
        for j in 0..8 {
            sum += (1 << (2 * j)) * if ((i >> j) & 1) == 1 { -1 } else { 1 };
        }
        assert_eq!(*output.get([i]).unwrap(), sum as f64);
    }
}

#[test]
fn backward() {
    let FullyConnectedLayer {
        input_size, output_size, parameter, ..
    } = FullyConnectedLayer::new_random_param(8, 1 << 8, NoneHash, Adam::new(0.0, 0.0, 0.0));
    assert_eq!(input_size, 8);
    assert_eq!(output_size, 1 << 8);
    let mut param = parameter.write().unwrap();
    let FullyConnectedLayerParameter { weight, bias, .. } = param.deref_mut();
    assert_eq!(weight.size(), [8, 1 << 8]);
    assert_eq!(bias.size(), [1 << 8]);
    for i in 0..output_size {
        let slice = weight.slice_one_line_mut(&[i]);
        assert_eq!(slice.len(), input_size);
        for j in 0..input_size {
            slice[j] = ((i + 1) * (j + 1)) as f64;
        }
    }
    let slice = bias.slice_one_line_mut(&[]);
    assert_eq!(slice.len(), output_size);
    for i in 0..output_size {
        slice[i] = i as f64;
    }

    let check = |output_grad: TensorEitherOwned<f64, 1>, input: TensorEitherOwned<f64, 1>| {
        assert_eq!(output_grad.size(), [output_size]);
        assert_eq!(input.size(), [input_size]);
        let mut grad = FullyConnectedLayerParameterGradient {
            weight: Dense::new([8, 1 << 8]),
            bias: Dense::new([1 << 8]),
        };
        let output = param.invoke_back(&mut grad, output_grad.as_ref(), input.as_ref());
        for i in 0..1 << 8 {
            assert_eq!(grad.bias.get([i]).copied().unwrap(), output_grad.get([i]).map(|_| (i + 1) as f64).unwrap_or_default());
            for j in 0..8 {
                assert_eq!(
                    grad.weight.get([j, i]).copied().unwrap(),
                    output_grad.get([i]).and(input.get([j])).map(|_| ((i + 1) * (j + 1)) as f64).unwrap_or_default()
                );
            }
        }
        for i in 0..8 {
            let mut sum = 0;
            for j in 0..1 << 8 {
                if output_grad.get([j]).is_some() {
                    sum += (i + 1) * (j + 1) * (j + 1);
                }
            }
            assert_eq!(output.get([i]).copied(), input.get([i]).map(|_| sum as f64));
        }

        let mut grad = FullyConnectedLayerParameterGradient {
            weight: Dense::new([8, 1 << 8]),
            bias: Dense::new([1 << 8]),
        };
        let output = param.invoke_back(&mut grad, output_grad.as_ref(), input.as_ref());
        for i in 0..1 << 8 {
            assert_eq!(grad.bias.get([i]).copied().unwrap(), output_grad.get([i]).map(|_| (i + 1) as f64).unwrap_or_default());
            for j in 0..8 {
                assert_eq!(
                    grad.weight.get([j, i]).copied().unwrap(),
                    output_grad.get([i]).and(input.get([j])).map(|_| ((i + 1) * (j + 1)) as f64).unwrap_or_default()
                );
            }
        }
        for i in 0..8 {
            let mut sum = 0;
            for j in 0..1 << 8 {
                if output_grad.get([j]).is_some() {
                    sum += (i + 1) * (j + 1) * (j + 1);
                }
            }
            assert_eq!(output.get([i]).copied(), input.get([i]).map(|_| sum as f64));
        }

        let mut grad = FullyConnectedLayerParameterGradient {
            weight: Dense::new([8, 1 << 8]),
            bias: Dense::new([1 << 8]),
        };
        let output = param.invoke_back(&mut grad, output_grad.as_ref(), input.as_ref());
        for i in 0..1 << 8 {
            assert_eq!(grad.bias.get([i]).copied().unwrap(), output_grad.get([i]).map(|_| (i + 1) as f64).unwrap_or_default());
            for j in 0..8 {
                assert_eq!(
                    grad.weight.get([j, i]).copied().unwrap(),
                    output_grad.get([i]).and(input.get([j])).map(|_| ((i + 1) * (j + 1)) as f64).unwrap_or_default()
                );
            }
        }
        for i in 0..8 {
            let mut sum = 0;
            for j in 0..1 << 8 {
                if output_grad.get([j]).is_some() {
                    sum += (i + 1) * (j + 1) * (j + 1);
                }
            }
            assert_eq!(output.get([i]).copied(), input.get([i]).map(|_| sum as f64));
        }

        let mut grad = FullyConnectedLayerParameterGradient {
            weight: Dense::new([8, 1 << 8]),
            bias: Dense::new([1 << 8]),
        };
        let output = param.invoke_back(&mut grad, output_grad.as_ref(), input.as_ref());
        for i in 0..1 << 8 {
            assert_eq!(grad.bias.get([i]).copied().unwrap(), output_grad.get([i]).map(|_| (i + 1) as f64).unwrap_or_default());
            for j in 0..8 {
                assert_eq!(
                    grad.weight.get([j, i]).copied().unwrap(),
                    output_grad.get([i]).and(input.get([j])).map(|_| ((i + 1) * (j + 1)) as f64).unwrap_or_default()
                );
            }
        }
        for i in 0..8 {
            let mut sum = 0;
            for j in 0..1 << 8 {
                if output_grad.get([j]).is_some() {
                    sum += (i + 1) * (j + 1) * (j + 1);
                }
            }
            assert_eq!(output.get([i]).copied(), input.get([i]).map(|_| sum as f64));
        }
    };

    let mut output_grad = Dense::new([1 << 8]);
    let mut input = Dense::new([8]);
    for i in 0..1 << 8 {
        output_grad.set([i], (i + 1) as f64);
    }
    for i in 0..8 {
        input.set([i], (i + 1) as f64);
    }
    check(output_grad.into(), input.into());

    let mut output_grad = Sparse::new([1 << 8]);
    let mut input = Dense::new([8]);
    for i in 0..1 << 8 {
        output_grad.set([i], (i + 1) as f64);
    }
    for i in 0..8 {
        input.set([i], (i + 1) as f64);
    }
    check(output_grad.into(), input.into());

    let mut output_grad = Dense::new([1 << 8]);
    let mut input = Sparse::new([8]);
    for i in 0..1 << 8 {
        output_grad.set([i], (i + 1) as f64);
    }
    for i in 0..8 {
        input.set([i], (i + 1) as f64);
    }
    check(output_grad.into(), input.into());

    let mut output_grad = Sparse::new([1 << 8]);
    let mut input = Sparse::new([8]);
    for i in 0..1 << 8 {
        output_grad.set([i], (i + 1) as f64);
    }
    for i in 0..8 {
        input.set([i], (i + 1) as f64);
    }
    check(output_grad.into(), input.into());

    for _ in 0..100 {
        let mut output_grad = Sparse::new([1 << 8]);
        let mut input = Dense::new([8]);
        for &i in (0..1 << 8).collect::<Vec<_>>().choose_multiple(&mut thread_rng(), 1 << 6) {
            output_grad.set([i], (i + 1) as f64);
        }
        for i in 0..8 {
            input.set([i], (i + 1) as f64);
        }
        check(output_grad.into(), input.into());

        let mut output_grad = Dense::new([1 << 8]);
        let mut input = Sparse::new([8]);
        for i in 0..1 << 8 {
            output_grad.set([i], (i + 1) as f64);
        }
        for &i in (0..8).collect::<Vec<_>>().choose_multiple(&mut thread_rng(), 3) {
            input.set([i], (i + 1) as f64);
        }
        check(output_grad.into(), input.into());

        let mut output_grad = Sparse::new([1 << 8]);
        let mut input = Sparse::new([8]);
        for &i in (0..1 << 8).collect::<Vec<_>>().choose_multiple(&mut thread_rng(), 1 << 6) {
            output_grad.set([i], (i + 1) as f64);
        }
        for &i in (0..8).collect::<Vec<_>>().choose_multiple(&mut thread_rng(), 3) {
            input.set([i], (i + 1) as f64);
        }
        check(output_grad.into(), input.into());
    }
}
