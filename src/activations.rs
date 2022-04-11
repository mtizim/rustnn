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

// branchless
#[inline(always)]
fn relu(f: fmod) -> fmod {
    (f.abs() + f) / 2.0
}
#[inline(always)]
fn relu_grad(f: fmod) -> fmod {
    let sgn = f.signum();
    (sgn.abs() + sgn) / 2.0
}
#[inline(always)]
fn tanh<F: Float>(f: F) -> F {
    f.tanh()
}
#[inline(always)]
fn tanh_grad<F: Float>(f: F) -> F {
    F::one() - f.powi(2)
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
    pub fn tanh() -> NeuronActivation {
        NeuronActivation {
            activation: tanh,
            gradient: tanh_grad,
        }
    }
}
