use crate::data_types::{Dense, Sparse, TensorEitherOwned};

#[test]
fn dense_dense() {
    let mut a = Dense::new([2, 4, 6]);
    let mut value = 0;
    for k in 0..6 {
        for j in 0..4 {
            for i in 0..2 {
                a.set([i, j, k], value);
                value += 1;
            }
        }
    }
    let mut b = Dense::new([2, 4, 6]);
    for k in 0..6 {
        for j in 0..4 {
            for i in 0..2 {
                b.set([i, j, k], value);
                value += 1;
            }
        }
    }
    let mut result = TensorEitherOwned::from(a);
    result += b.into();
    assert_eq!(result.size(), [2, 4, 6]);
    let result = if let TensorEitherOwned::Dense(tensor) = result { tensor } else { unreachable!() };
    for (index, i) in result.data.iter().enumerate() {
        assert_eq!(*i, (48 + 2 * index) as i32);
    }
}

#[test]
fn dense_sparse() {
    let mut a = Dense::new([2, 4, 6]);
    let mut value = 0;
    for k in 0..6 {
        for j in 0..4 {
            for i in 0..2 {
                a.set([i, j, k], value);
                value += 1;
            }
        }
    }
    let mut b = Sparse::new([2, 4, 6]);
    for i in 0..=48 / 7 {
        let index = i * 7;
        let index = [index % 8 % 2, index % 8 / 2, index / 8];
        b.set(index, 100);
    }
    let mut result = TensorEitherOwned::from(a);
    result += b.into();
    let result = if let TensorEitherOwned::Dense(tensor) = result { tensor } else { unreachable!() };
    for i in 0..48 {
        if i % 7 == 0 {
            assert_eq!(result.data[i], 100 + i);
        } else {
            assert_eq!(result.data[i], i);
        }
    }
}

#[test]
fn sparse_dense() {
    let mut a = Sparse::new([2, 4, 6]);
    for i in 0..=48 / 7 {
        let index = i * 7;
        let index = [index % 8 % 2, index % 8 / 2, index / 8];
        a.set(index, 100);
    }
    let mut b = Dense::new([2, 4, 6]);
    let mut value = 0;
    for k in 0..6 {
        for j in 0..4 {
            for i in 0..2 {
                b.set([i, j, k], value);
                value += 1;
            }
        }
    }
    let mut result = TensorEitherOwned::from(a);
    result += b.into();
    let result = if let TensorEitherOwned::Dense(tensor) = result { tensor } else { unreachable!() };
    for i in 0..48 {
        if i % 7 == 0 {
            assert_eq!(result.data[i], 100 + i);
        } else {
            assert_eq!(result.data[i], i);
        }
    }
}

#[test]
fn sparse_sparse() {
    let mut a = Sparse::new([2, 4, 6]);
    for i in 0..48 / 3 {
        let index = i * 3;
        let index = [index % 8 % 2, index % 8 / 2, index / 8];
        a.set(index, 100);
    }
    let mut b = Sparse::new([2, 4, 6]);
    for i in 0..48 / 2 {
        let index = i * 2;
        let index = [index % 8 % 2, index % 8 / 2, index / 8];
        b.set(index, 50);
    }
    let mut result = TensorEitherOwned::from(a);
    result += b.into();
    for i in 0..48 {
        let index = [i % 8 % 2, i % 8 / 2, i / 8];
        let expect = match (i % 2, i % 3) {
            (0, 0) => Some(150),
            (_, 0) => Some(100),
            (0, _) => Some(50),
            _ => None,
        };
        assert_eq!(result.get(index), expect.as_ref());
    }
}
