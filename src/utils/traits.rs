//! Shared trait and trait boundaries

use ann_search_rs::utils::dist::SimdDistance;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use std::fmt::Debug;
use std::iter::Sum;

/// Core float trait for EVoC
pub trait EvocFloat:
    Float + FromPrimitive + ToPrimitive + Send + Sync + Debug + Sum + Default + SimdDistance + 'static
{
}

impl<T> EvocFloat for T where
    T: Float
        + FromPrimitive
        + ToPrimitive
        + Send
        + Sync
        + Debug
        + Sum
        + Default
        + SimdDistance
        + 'static
{
}
