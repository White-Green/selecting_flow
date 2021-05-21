use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;

use num_traits::{NumAssign, One};

use crate::data_types::{Dense, TensorEither, TensorEitherOwned};

pub mod activation;
pub mod constant;
pub mod fully_connected_layer;
pub mod input_box;

pub(crate) type ShareCell<T> = RefCell<T>;

pub(crate) fn read<'a, T: 'a + ?Sized>(cell: &'a impl AsRef<RefCell<T>>) -> impl 'a + Deref<Target = T> {
    cell.as_ref().borrow()
}

pub(crate) fn write<'a, T: 'a + ?Sized>(cell: &'a impl AsRef<RefCell<T>>) -> impl 'a + DerefMut<Target = T> {
    cell.as_ref().borrow_mut()
}

pub struct ValueWithSequence<T> {
    pub(crate) value: T,
    pub(crate) sequence: NonZeroUsize,
}

impl<T> ValueWithSequence<T> {
    pub(crate) fn new(value: T, sequence: NonZeroUsize) -> Self {
        Self { value, sequence }
    }
}

pub struct DynNode {
    node: Rc<ShareCell<dyn ComputeGraphNode>>,
    generation: usize,
}

impl DynNode {
    pub(crate) fn new<N: 'static + ComputeGraphNode>(node: Rc<ShareCell<N>>, generation: usize) -> Self {
        DynNode {
            node: node as Rc<ShareCell<dyn ComputeGraphNode>>,
            generation,
        }
    }
}

impl PartialEq for DynNode {
    fn eq(&self, other: &Self) -> bool {
        self.generation.eq(&other.generation) && Rc::ptr_eq(&self.node, &other.node)
    }
}

impl Eq for DynNode {}

impl PartialOrd for DynNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DynNode {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.generation.cmp(&other.generation) {
            Ordering::Equal => Rc::as_ptr(&self.node).cmp(&Rc::as_ptr(&other.node)),
            other => other,
        }
    }
}

impl Hash for DynNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(Rc::as_ptr(&self.node) as *const () as usize);
        state.write_usize(self.generation);
    }
}

pub trait ComputeGraphNode {
    fn generation(&self) -> usize;
    fn prev_nodes(&self, next_list: &mut Vec<DynNode>);
    fn clear_gradient(&mut self);
    fn apply_back_propagation(&mut self);
}

pub trait ExactDimensionComputeGraphNode<const N: usize>: ComputeGraphNode {
    type Item: NumAssign + Default + Clone;
    fn output_shape(&self) -> [usize; N];
    fn get_output(&mut self) -> ValueWithSequence<TensorEither<Self::Item, N>>;
    fn add_gradient(&mut self, tensor: TensorEither<Self::Item, N>);
}

pub(crate) trait IntoComputeGraphNode<const N: usize>
where
    <Self::ComputeGraphNode as ExactDimensionComputeGraphNode<N>>::Item: NumAssign + Default + Clone,
{
    type ComputeGraphNode: ExactDimensionComputeGraphNode<N>;
    fn into_node(self) -> Rc<ShareCell<Self::ComputeGraphNode>>;
}

pub struct GraphNode<T: ExactDimensionComputeGraphNode<N>, const N: usize>(Rc<ShareCell<T>>)
where
    T::Item: NumAssign + Default + Clone;

impl<T: ExactDimensionComputeGraphNode<N>, const N: usize> Clone for GraphNode<T, N> {
    fn clone(&self) -> Self {
        GraphNode(Rc::clone(&self.0))
    }
}

impl<G: ExactDimensionComputeGraphNode<N>, const N: usize> GraphNode<G, N>
where
    G::Item: NumAssign + Default + Clone,
{
    pub(crate) fn new(item: G) -> Self {
        GraphNode(Rc::new(ShareCell::new(item)))
    }

    pub fn get_output_value(&mut self) -> TensorEitherOwned<G::Item, N> {
        let mut node = write(&self.0);
        let ValueWithSequence { value, sequence: _ } = node.get_output();
        value.into_owned()
    }

    pub fn clear_gradient_once(&mut self) {
        write(&self.0).clear_gradient();
    }

    pub fn clear_gradient_all(&mut self) {
        let mut vec = Vec::new();
        let mut q = VecDeque::new();
        let mut cleared = HashSet::new();
        let mut node = write(&self.0);
        node.clear_gradient();
        node.prev_nodes(&mut vec);
        q.extend(vec.drain(..));
        while let Some(node) = q.pop_front() {
            if cleared.contains(&node) {
                continue;
            }
            {
                let mut node = write(&node.node);
                node.clear_gradient();
                node.prev_nodes(&mut vec);
                q.extend(vec.drain(..));
            }
            cleared.insert(node);
        }
    }

    pub fn back_propagate_all(&mut self) {
        let mut vec = Vec::new();
        let mut q = BinaryHeap::new();
        let mut cleared = HashSet::new();
        let mut node = write(&self.0);
        let ValueWithSequence { value: output, .. } = node.get_output();
        let mut gradient = Dense::new(output.size());
        gradient.as_all_slice_mut().iter_mut().for_each(|v| *v = G::Item::one());
        node.add_gradient(gradient.into());
        node.apply_back_propagation();
        node.prev_nodes(&mut vec);
        q.extend(vec.drain(..));
        while let Some(node) = q.pop() {
            if cleared.contains(&node) {
                continue;
            }
            {
                let mut node = write(&node.node);
                node.apply_back_propagation();
                node.prev_nodes(&mut vec);
                q.extend(vec.drain(..));
            }
            cleared.insert(node);
        }
    }
}

impl<G: ExactDimensionComputeGraphNode<N>, const N: usize> IntoComputeGraphNode<N> for GraphNode<G, N>
where
    G::Item: NumAssign + Default + Clone,
{
    type ComputeGraphNode = G;

    fn into_node(self) -> Rc<ShareCell<Self::ComputeGraphNode>> {
        self.0
    }
}
