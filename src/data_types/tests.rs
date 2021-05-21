use crate::data_types::{Dense, Sparse};

#[test]
fn tensor_dense() {
    let mut tensor = Dense::new([2, 4, 6]);
    assert_eq!(tensor.size(), [2, 4, 6]);
    assert_eq!(tensor.weights, [1, 2, 8]);
    assert_eq!(tensor.data.len(), 48);
    for i in 0..48 {
        tensor.data[i] = i;
    }
    let mut val = 0;
    for k in 0..6 {
        for j in 0..4 {
            for i in 0..2 {
                assert_eq!(tensor.get([i, j, k]), Some(&val));
                assert_eq!(tensor.get_mut([i, j, k]), Some(&mut val));
                tensor.set([i, j, k], 100);
                val += 1;
            }
        }
    }
    for k in 0..6 {
        for j in 0..4 {
            let i = 2;
            assert_eq!(tensor.get([i, j, k]), None);
            assert_eq!(tensor.get_mut([i, j, k]), None);
        }
    }
    for k in 0..6 {
        let j = 4;
        for i in 0..2 {
            assert_eq!(tensor.get([i, j, k]), None);
            assert_eq!(tensor.get_mut([i, j, k]), None);
        }
    }
    let k = 6;
    for j in 0..4 {
        for i in 0..2 {
            assert_eq!(tensor.get([i, j, k]), None);
            assert_eq!(tensor.get_mut([i, j, k]), None);
        }
    }
    for i in 0..48 {
        assert_eq!(tensor.data[i], 100);
    }
}

#[test]
fn dense_slice() {
    let mut tensor = Dense::new([2, 4, 6]);
    for i in 0..48 {
        tensor.data[i] = i;
    }
    for k in 0..6 {
        for j in 0..4 {
            assert_eq!(tensor.slice_one_line(&[j, k]), (8 * k + 2 * j..8 * k + 2 * j + 2).collect::<Vec<_>>());
            assert_eq!(tensor.slice_one_line_mut(&[j, k]), (8 * k + 2 * j..8 * k + 2 * j + 2).collect::<Vec<_>>());
        }
    }
}

#[test]
fn tensor_sparse() {
    let mut tensor = Sparse::new([2, 4, 6]);
    assert_eq!(tensor.size(), [2, 4, 6]);
    let mut val = 0;
    for k in 0..6 {
        for j in 0..4 {
            for i in 0..2 {
                assert_eq!(tensor.get([i, j, k]), None);
                assert_eq!(tensor.get_mut([i, j, k]), None);
                tensor.set([i, j, k], val);
                val += 1;
            }
        }
    }
    assert_eq!(tensor.data.len(), 48);
    tensor.data.values().enumerate().for_each(|(index, &value)| assert_eq!(index, value));
}
