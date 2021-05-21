use std::ops::AddAssign;

use num_traits::NumAssign;

use crate::data_types::{TensorEither, TensorEitherOwned};

#[cfg(test)]
mod tests;

impl<'a, T: NumAssign + Default + Clone, const N: usize> AddAssign<TensorEither<'a, T, N>> for TensorEitherOwned<T, N> {
    fn add_assign(&mut self, rhs: TensorEither<'a, T, N>) {
        assert_eq!(self.size(), rhs.size());
        match (self, rhs) {
            (TensorEitherOwned::Dense(this), TensorEither::Dense(rhs)) => {
                this.data.iter_mut().zip(rhs.data.iter()).for_each(|(this, rhs)| *this += rhs.clone());
            }
            (TensorEitherOwned::Dense(this), TensorEither::Sparse(rhs)) => {
                rhs.iter().for_each(|(k, v)| *this.get_mut(k).unwrap() += v.clone());
            }
            (this @ TensorEitherOwned::Sparse(_), TensorEither::Dense(rhs)) => {
                let mut rhs = rhs.clone().into_owned();
                if let TensorEitherOwned::Sparse(this) = this {
                    this.iter().for_each(|(k, v)| *rhs.get_mut(k).unwrap() += v.clone());
                }
                *this = TensorEitherOwned::Dense(rhs);
            }
            (TensorEitherOwned::Sparse(this), TensorEither::Sparse(rhs)) => {
                rhs.iter().for_each(|(k, v)| {
                    if let Some(this) = this.get_mut(k) {
                        *this += v.clone();
                    } else {
                        this.set(k, v.clone());
                    }
                });
            }
        };
    }
}
