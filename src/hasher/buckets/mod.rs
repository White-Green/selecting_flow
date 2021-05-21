#![allow(dead_code)]
use itertools::Itertools;
use rand::{thread_rng, Rng};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::IntoParallelRefMutIterator;

use crate::data_types::{Dense, DensePartialDimension, DensePartialDimensionMut};

#[derive(Debug)]
pub(crate) struct Buckets<T> {
    data: Dense<T, 3>,
    length: Dense<usize, 2>,
}

impl<T: Default> Buckets<T> {
    pub(crate) fn new(width: usize, height: usize, max_len: usize) -> Self {
        Buckets {
            data: Dense::new([max_len, width, height]),
            length: Dense::new([width, height]),
        }
    }

    pub(crate) fn add_item(&mut self, x: usize, y: usize, item: T) {
        let index = self.length.get_mut([x, y]).unwrap();
        let i = if *index < self.data.size()[0] { *index } else { thread_rng().gen_range(0..self.data.size()[0]) };
        *index += 1;
        self.data.set([i, x, y], item);
    }

    pub(crate) fn clear(&mut self) {
        self.length.as_all_slice_mut().par_iter_mut().for_each(|v| *v = 0);
    }

    pub(crate) fn get_items(&self, x: usize, y: usize) -> &[T] {
        let len = self.length.get([x, y]).copied().unwrap().min(self.data.size()[0]);
        &self.data.slice_one_line(&[x, y])[..len]
    }

    pub(crate) fn partial_buckets(&self, dim: usize) -> impl ExactSizeIterator<Item = PartialDimBucket<'_, T>> {
        assert!(dim <= 2);
        self.data
            .partial_dimension(dim + 1)
            .zip_eq(self.length.partial_dimension(dim))
            .map(move |(data, len)| PartialDimBucket { dim, len, data })
    }

    pub(crate) fn partial_buckets_mut(&mut self, dim: usize) -> impl ExactSizeIterator<Item = PartialDimBucketMut<'_, T>> {
        assert!(dim <= 2);
        self.data
            .partial_dimension_mut(dim + 1)
            .zip_eq(self.length.partial_dimension_mut(dim))
            .map(move |(data, len)| PartialDimBucketMut { dim, len, data })
    }
}

impl<T: Default + Send + Sync> Buckets<T> {
    pub(crate) fn par_partial_buckets(&self, dim: usize) -> impl IndexedParallelIterator<Item = PartialDimBucket<'_, T>> {
        assert!(dim <= 2);
        self.data
            .par_partial_dimension(dim + 1)
            .zip_eq(self.length.par_partial_dimension(dim))
            .map(move |(data, len)| PartialDimBucket { dim, len, data })
    }

    pub(crate) fn par_partial_buckets_mut(&mut self, dim: usize) -> impl IndexedParallelIterator<Item = PartialDimBucketMut<'_, T>> {
        assert!(dim <= 2);
        self.data
            .par_partial_dimension_mut(dim + 1)
            .zip_eq(self.length.par_partial_dimension_mut(dim))
            .map(move |(data, len)| PartialDimBucketMut { dim, len, data })
    }
}

pub(crate) struct PartialDimBucket<'a, T> {
    dim: usize,
    len: DensePartialDimension<'a, usize, 2>,
    data: DensePartialDimension<'a, T, 3>,
}

impl<'a, T> PartialDimBucket<'a, T> {
    pub(crate) fn get_items(&self, index: &[usize]) -> &[T] {
        assert_eq!(index.len(), self.dim);
        let len = self.len.get(index).copied().unwrap().min(self.data.size()[0]);
        &self.data.slice_one_line(index)[..len]
    }
}

pub(crate) struct PartialDimBucketMut<'a, T> {
    dim: usize,
    len: DensePartialDimensionMut<'a, usize, 2>,
    data: DensePartialDimensionMut<'a, T, 3>,
}

impl<'a, T> PartialDimBucketMut<'a, T> {
    pub(crate) fn get_items(&self, index: &[usize]) -> &[T] {
        assert_eq!(index.len(), self.dim);
        let len = self.len.get(index).copied().unwrap().min(self.data.size()[0]);
        &self.data.slice_one_line(index)[..len]
    }

    pub(crate) fn add_item(&mut self, index: &[usize], item: T) {
        assert_eq!(index.len(), self.dim);
        let item_index = self.len.get_mut(index).unwrap();
        let i = if *item_index < self.data.size()[0] {
            *item_index
        } else {
            thread_rng().gen_range(0..self.data.size()[0])
        };
        *item_index += 1;
        let mut i = [i; 3];
        i[1..1 + index.len()].clone_from_slice(index);
        *self.data.get_mut(&i[..1 + index.len()]).unwrap() = item;
    }
}
