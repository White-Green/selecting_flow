use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::rc::Rc;

use num_traits::NumAssign;
use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSlice, ParallelSliceMut};

use crate::compute_graph::constant::Constant;
use crate::compute_graph::{IntoComputeGraphNode, ShareCell};

mod tensor_operation;
#[cfg(test)]
mod tests;

#[derive(Debug, Clone)]
pub enum TensorEitherOwned<T, const N: usize> {
    Dense(Dense<T, N>),
    Sparse(Sparse<T, N>),
}

#[derive(Debug, Clone)]
pub enum TensorEither<'a, T: Clone, const N: usize> {
    Dense(Cow<'a, Dense<T, N>>),
    Sparse(Cow<'a, Sparse<T, N>>),
}

impl<T: Clone, const N: usize> TensorEitherOwned<T, N> {
    pub fn new_dense(value: Dense<T, N>) -> Self {
        value.into()
    }
    pub fn new_sparse(value: Sparse<T, N>) -> Self {
        value.into()
    }
    pub fn size(&self) -> [usize; N] {
        match self {
            TensorEitherOwned::Dense(tensor) => tensor.size,
            TensorEitherOwned::Sparse(tensor) => tensor.size,
        }
    }
    pub fn as_ref(&self) -> TensorEither<T, N> {
        match self {
            TensorEitherOwned::Dense(tensor) => tensor.into(),
            TensorEitherOwned::Sparse(tensor) => tensor.into(),
        }
    }
}

impl<T: Default + Clone, const N: usize> TensorEitherOwned<T, N> {
    pub fn get(&self, index: [usize; N]) -> Option<&T> {
        match self {
            TensorEitherOwned::Dense(tensor) => tensor.get(index),
            TensorEitherOwned::Sparse(tensor) => tensor.get(index),
        }
    }
    pub fn get_mut(&mut self, index: [usize; N]) -> Option<&mut T> {
        match self {
            TensorEitherOwned::Dense(tensor) => tensor.get_mut(index),
            TensorEitherOwned::Sparse(tensor) => tensor.get_mut(index),
        }
    }
}

impl<T, const N: usize> From<Dense<T, N>> for TensorEitherOwned<T, N> {
    fn from(value: Dense<T, N>) -> Self {
        TensorEitherOwned::Dense(value)
    }
}

impl<T, const N: usize> From<Sparse<T, N>> for TensorEitherOwned<T, N> {
    fn from(value: Sparse<T, N>) -> Self {
        TensorEitherOwned::Sparse(value)
    }
}

impl<'a, T: Clone, const N: usize> TensorEither<'a, T, N> {
    pub fn new_dense(value: Dense<T, N>) -> Self {
        value.into()
    }
    pub fn new_dense_ref(value: &'a Dense<T, N>) -> Self {
        value.into()
    }
    pub fn new_sparse(value: Sparse<T, N>) -> Self {
        value.into()
    }
    pub fn new_sparse_ref(value: &'a Sparse<T, N>) -> Self {
        value.into()
    }
    pub fn size(&self) -> [usize; N] {
        match self {
            TensorEither::Dense(tensor) => tensor.size,
            TensorEither::Sparse(tensor) => tensor.size,
        }
    }
    pub fn into_owned(self) -> TensorEitherOwned<T, N> {
        match self {
            TensorEither::Dense(tensor) => tensor.into_owned().into(),
            TensorEither::Sparse(tensor) => tensor.into_owned().into(),
        }
    }
    pub fn as_ref(&self) -> TensorEither<T, N> {
        match self {
            TensorEither::Dense(tensor) => tensor.as_ref().into(),
            TensorEither::Sparse(tensor) => tensor.as_ref().into(),
        }
    }
}

impl<'a, T: Default + Clone, const N: usize> TensorEither<'a, T, N> {
    pub fn get(&self, index: [usize; N]) -> Option<&T> {
        match self {
            TensorEither::Dense(tensor) => tensor.get(index),
            TensorEither::Sparse(tensor) => tensor.get(index),
        }
    }
}

impl<'a, T: Clone, const N: usize> From<Dense<T, N>> for TensorEither<'a, T, N> {
    fn from(value: Dense<T, N>) -> Self {
        TensorEither::Dense(Cow::Owned(value))
    }
}

impl<'a, T: Clone, const N: usize> From<&'a Dense<T, N>> for TensorEither<'a, T, N> {
    fn from(value: &'a Dense<T, N>) -> Self {
        TensorEither::Dense(Cow::Borrowed(value))
    }
}

impl<'a, T: Clone, const N: usize> From<Sparse<T, N>> for TensorEither<'a, T, N> {
    fn from(value: Sparse<T, N>) -> Self {
        TensorEither::Sparse(Cow::Owned(value))
    }
}

impl<'a, T: Clone, const N: usize> From<&'a Sparse<T, N>> for TensorEither<'a, T, N> {
    fn from(value: &'a Sparse<T, N>) -> Self {
        TensorEither::Sparse(Cow::Borrowed(value))
    }
}

impl<'a, T: Clone, const N: usize> From<TensorEitherOwned<T, N>> for TensorEither<'a, T, N> {
    fn from(value: TensorEitherOwned<T, N>) -> Self {
        match value {
            TensorEitherOwned::Dense(tensor) => tensor.into(),
            TensorEitherOwned::Sparse(tensor) => tensor.into(),
        }
    }
}

impl<'a, T: Clone, const N: usize> From<&'a TensorEitherOwned<T, N>> for TensorEither<'a, T, N> {
    fn from(value: &'a TensorEitherOwned<T, N>) -> Self {
        match value {
            TensorEitherOwned::Dense(tensor) => tensor.into(),
            TensorEitherOwned::Sparse(tensor) => tensor.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Dense<T, const N: usize> {
    size: [usize; N],
    weights: [usize; N],
    data: Vec<T>,
}

impl<T: Default, const N: usize> Dense<T, N> {
    pub fn new(size: [usize; N]) -> Self {
        let weights = size.iter().copied().enumerate().take(N - 1).fold([1; N], |mut state, (index, item)| {
            state[index + 1] = state[index] * item;
            state
        });
        let data = {
            let size = size.iter().copied().product::<usize>();
            let mut vec = Vec::with_capacity(size);
            vec.resize_with(size, T::default);
            vec
        };
        Self { size, weights, data }
    }

    pub fn size(&self) -> [usize; N] {
        self.size
    }

    pub fn get(&self, index: [usize; N]) -> Option<&T> {
        if index.iter().zip(self.size.iter()).all(|(&index, &size)| index < size) {
            let index: usize = index.iter().zip(self.weights.iter()).map(|(&index, &size)| index * size).sum();
            self.data.get(index)
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, index: [usize; N]) -> Option<&mut T> {
        if index.iter().zip(self.size.iter()).all(|(&index, &size)| index < size) {
            let index: usize = index.iter().zip(self.weights.iter()).map(|(&index, &size)| index * size).sum();
            self.data.get_mut(index)
        } else {
            None
        }
    }

    pub fn set(&mut self, index: [usize; N], value: T) {
        if index.iter().zip(self.size.iter()).all(|(&index, &size)| index < size) {
            let index: usize = index.iter().zip(self.weights.iter()).map(|(&index, &size)| index * size).sum();
            unsafe {
                *self.data.get_unchecked_mut(index) = value;
            }
        } else {
            panic!("index out of range");
        }
    }
}

impl<T, const N: usize> Dense<T, N> {
    pub fn as_all_slice(&self) -> &[T] {
        &self.data
    }

    pub fn as_all_slice_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn slice_one_line(&self, line_index: &[usize]) -> &[T] {
        assert_eq!(line_index.len(), N - 1);
        let start_index = Some(0).iter().chain(line_index.iter()).zip(self.weights.iter()).map(|(&index, &size)| index * size).sum();
        &self.data[start_index..start_index + self.size[0]]
    }
    pub fn slice_one_line_mut(&mut self, line_index: &[usize]) -> &mut [T] {
        assert_eq!(line_index.len(), N - 1);
        let start_index = Some(0).iter().chain(line_index.iter()).zip(self.weights.iter()).map(|(&index, &size)| index * size).sum();
        &mut self.data[start_index..start_index + self.size[0]]
    }

    pub fn partial_dimension(&self, dim: usize) -> impl '_ + ExactSizeIterator<Item = DensePartialDimension<'_, T, N>> {
        assert!(dim <= N);
        let &Dense { size, weights, ref data } = self;
        let chunk_size: usize = size[..dim].iter().product();
        data.chunks(chunk_size).map(move |data| DensePartialDimension { dim, size, weights, data })
    }

    pub fn partial_dimension_mut(&mut self, dim: usize) -> impl '_ + ExactSizeIterator<Item = DensePartialDimensionMut<'_, T, N>> {
        assert!(dim <= N);
        let &mut Dense { size, weights, ref mut data } = self;
        let chunk_size: usize = size[..dim].iter().product();
        data.chunks_mut(chunk_size).map(move |data| DensePartialDimensionMut { dim, size, weights, data })
    }
}

impl<T: Default + Send + Sync, const N: usize> Dense<T, N> {
    pub fn par_partial_dimension(&self, dim: usize) -> impl '_ + IndexedParallelIterator<Item = DensePartialDimension<'_, T, N>> {
        assert!(dim <= N);
        let &Dense { size, weights, ref data } = self;
        let chunk_size: usize = size[..dim].iter().product();
        data.par_chunks(chunk_size).map(move |data| DensePartialDimension { dim, size, weights, data })
    }

    pub fn par_partial_dimension_mut(&mut self, dim: usize) -> impl '_ + IndexedParallelIterator<Item = DensePartialDimensionMut<'_, T, N>> {
        assert!(dim <= N);
        let &mut Dense { size, weights, ref mut data } = self;
        let chunk_size: usize = size[..dim].iter().product();
        data.par_chunks_mut(chunk_size).map(move |data| DensePartialDimensionMut { dim, size, weights, data })
    }
}

pub struct DensePartialDimension<'a, T, const N: usize> {
    dim: usize,
    size: [usize; N],
    weights: [usize; N],
    data: &'a [T],
}

impl<'a, T, const N: usize> DensePartialDimension<'a, T, N> {
    pub fn size(&self) -> &[usize] {
        &self.size[..self.dim]
    }

    pub fn get(&self, index: &[usize]) -> Option<&T> {
        if index.len() == self.dim && index.iter().copied().zip(self.size.iter().copied()).all(|(index, size)| index < size) {
            let index: usize = index.iter().copied().zip(self.weights.iter().copied()).map(|(a, b)| a * b).sum();
            Some(&self.data[index])
        } else {
            None
        }
    }

    pub fn partial_dimension(&self, dim: usize) -> impl ExactSizeIterator<Item = DensePartialDimension<'_, T, N>> {
        assert!(dim <= self.dim);
        let &DensePartialDimension { size, weights, data, .. } = self;
        let chunk_size = self.size[..dim].iter().product();
        data.chunks(chunk_size).map(move |data| DensePartialDimension { dim, size, weights, data })
    }

    pub fn slice_one_line(&self, line_index: &[usize]) -> &[T] {
        assert_eq!(line_index.len(), self.dim - 1);
        let start_index = Some(0).iter().chain(line_index.iter()).zip(self.weights.iter()).map(|(&index, &size)| index * size).sum();
        &self.data[start_index..start_index + self.size[0]]
    }

    pub fn get_all_slice(&self) -> &[T] {
        self.data
    }
}

pub struct DensePartialDimensionMut<'a, T, const N: usize> {
    dim: usize,
    size: [usize; N],
    weights: [usize; N],
    data: &'a mut [T],
}

impl<'a, T, const N: usize> DensePartialDimensionMut<'a, T, N> {
    pub fn size(&self) -> &[usize] {
        &self.size[..self.dim]
    }

    pub fn get(&self, index: &[usize]) -> Option<&T> {
        if index.len() == self.dim && index.iter().copied().zip(self.size.iter().copied()).all(|(index, size)| index < size) {
            let index: usize = index.iter().copied().zip(self.weights.iter().copied()).map(|(a, b)| a * b).sum();
            Some(&self.data[index])
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, index: &[usize]) -> Option<&mut T> {
        if index.len() == self.dim && index.iter().copied().zip(self.size.iter().copied()).all(|(index, size)| index < size) {
            let index: usize = index.iter().copied().zip(self.weights.iter().copied()).map(|(a, b)| a * b).sum();
            Some(&mut self.data[index])
        } else {
            None
        }
    }

    pub fn partial_dimension(&self, dim: usize) -> impl ExactSizeIterator<Item = DensePartialDimension<'_, T, N>> {
        assert!(dim <= self.dim);
        let &DensePartialDimensionMut { size, weights, ref data, .. } = self;
        let chunk_size = self.size[..dim].iter().product();
        data.chunks(chunk_size).map(move |data| DensePartialDimension { dim, size, weights, data })
    }

    pub fn partial_dimension_mut(&mut self, dim: usize) -> impl ExactSizeIterator<Item = DensePartialDimensionMut<'_, T, N>> {
        assert!(dim <= self.dim);
        let &mut DensePartialDimensionMut { size, weights, ref mut data, .. } = self;
        let chunk_size = self.size[..dim].iter().product();
        data.chunks_mut(chunk_size).map(move |data| DensePartialDimensionMut { dim, size, weights, data })
    }

    pub fn slice_one_line(&self, line_index: &[usize]) -> &[T] {
        assert_eq!(line_index.len(), self.dim - 1);
        let start_index = Some(0).iter().chain(line_index.iter()).zip(self.weights.iter()).map(|(&index, &size)| index * size).sum();
        &self.data[start_index..start_index + self.size[0]]
    }

    pub fn slice_one_line_mut(&mut self, line_index: &[usize]) -> &mut [T] {
        assert_eq!(line_index.len(), self.dim - 1);
        let start_index = Some(0).iter().chain(line_index.iter()).zip(self.weights.iter()).map(|(&index, &size)| index * size).sum();
        &mut self.data[start_index..start_index + self.size[0]]
    }

    pub fn get_all_slice(&self) -> &[T] {
        self.data
    }

    pub fn get_all_slice_mut(&mut self) -> &mut [T] {
        self.data
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ArrayReverseOrder<T: Ord, const N: usize>([T; N]);

impl<T: Ord, const N: usize> PartialOrd for ArrayReverseOrder<T, N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        for (s, o) in self.0.iter().zip(other.0.iter()).rev() {
            match s.cmp(o) {
                Ordering::Equal => {}
                other => return Some(other),
            }
        }
        Some(Ordering::Equal)
    }
}

impl<T: Ord, const N: usize> Ord for ArrayReverseOrder<T, N> {
    fn cmp(&self, other: &Self) -> Ordering {
        for (s, o) in self.0.iter().zip(other.0.iter()).rev() {
            match s.cmp(o) {
                Ordering::Equal => {}
                other => return other,
            }
        }
        Ordering::Equal
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Sparse<T, const N: usize> {
    size: [usize; N],
    data: BTreeMap<ArrayReverseOrder<usize, N>, T>,
}

impl<T: Default, const N: usize> Sparse<T, N> {
    pub fn new(size: [usize; N]) -> Self {
        Self { size, data: BTreeMap::new() }
    }

    pub fn size(&self) -> [usize; N] {
        self.size
    }

    pub fn value_count(&self) -> usize {
        self.data.len()
    }

    pub fn get(&self, index: [usize; N]) -> Option<&T> {
        self.data.get(&ArrayReverseOrder(index))
    }

    pub fn get_mut(&mut self, index: [usize; N]) -> Option<&mut T> {
        self.data.get_mut(&ArrayReverseOrder(index))
    }

    pub fn set(&mut self, index: [usize; N], value: T) {
        if index.iter().zip(self.size.iter()).all(|(&index, &size)| index < size) {
            self.data.insert(ArrayReverseOrder(index), value);
        } else {
            panic!("index out of range")
        }
    }

    pub fn remove(&mut self, index: [usize; N]) {
        self.data.remove(&ArrayReverseOrder(index));
    }

    pub fn get_or_insert(&mut self, index: [usize; N], value: T) -> Option<&T> {
        if index.iter().zip(self.size.iter()).all(|(&index, &size)| index < size) {
            Some(self.data.entry(ArrayReverseOrder(index)).or_insert(value))
        } else {
            None
        }
    }

    pub fn get_or_insert_mut(&mut self, index: [usize; N], value: T) -> Option<&mut T> {
        if index.iter().zip(self.size.iter()).all(|(&index, &size)| index < size) {
            Some(self.data.entry(ArrayReverseOrder(index)).or_insert(value))
        } else {
            None
        }
    }
}

impl<T, const N: usize> Sparse<T, N> {
    pub fn iter(&self) -> impl Iterator<Item = ([usize; N], &T)> {
        self.data.iter().map(|(k, v)| (k.0, v))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = ([usize; N], &mut T)> {
        self.data.iter_mut().map(|(k, v)| (k.0, v))
    }
}

impl<T: NumAssign + Default + Clone, const N: usize> IntoComputeGraphNode<N> for Dense<T, N> {
    type ComputeGraphNode = Constant<Self>;

    fn into_node(self) -> Rc<ShareCell<Self::ComputeGraphNode>> {
        Rc::new(ShareCell::new(Constant::new(self)))
    }
}

impl<T: NumAssign + Default + Clone, const N: usize> IntoComputeGraphNode<N> for Sparse<T, N> {
    type ComputeGraphNode = Constant<Self>;

    fn into_node(self) -> Rc<ShareCell<Self::ComputeGraphNode>> {
        Rc::new(ShareCell::new(Constant::new(self)))
    }
}
