use num::Float;

use crate::fmod;

pub type Activation = fn(fmod) -> fmod;
pub type Gradient = fn(fmod) -> fmod;

pub struct NeuronActivation {
    pub activation: Activation,
    pub gradient: Gradient,
}

#[inline(always)]
fn sigmoid<F: Float>(f: F) -> F {
    use std::f64::consts::E;
    let e = F::from(E).unwrap();
    F::one() / (F::one() + e.powf(-f))
}
#[inline(always)]
fn sigmoid_grad<F: Float>(f: F) -> F {
    f * (F::one() - f)
}
#[inline(always)]
fn linear<F: Float>(f: F) -> F {
    f
}
#[inline(always)]
fn linear_grad<F: Float>(_f: F) -> F {
    F::one()
}

#[inline(always)]
fn relu<F: Float>(f: F) -> F {
    if f < F::zero() {
        F::zero()
    } else {
        f
    }
}
#[inline(always)]
fn relu_grad<F: Float>(f: F) -> F {
    if f < F::zero() {
        F::zero()
    } else {
        F::one()
    }
}

impl NeuronActivation {
    pub fn sigmoid() -> NeuronActivation {
        NeuronActivation {
            activation: sigmoid,
            gradient: sigmoid_grad,
        }
    }
    pub fn linear() -> NeuronActivation {
        NeuronActivation {
            activation: linear,
            gradient: linear_grad,
        }
    }
    pub fn relu() -> NeuronActivation {
        NeuronActivation {
            activation: relu,
            gradient: relu_grad,
        }
    }
}
