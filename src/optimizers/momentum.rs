use crate::fmod;
use ndarray::{Array1, Array2};

use super::optimizers::*;

#[inline(always)]
fn momentum_memoizer(
    params: &Vec<fmod>,
    mem: &OptimizerMemory,
    weight_grad: &Vec<Array2<fmod>>,
    bias_grad: &Vec<Array1<fmod>>,
) -> OptimizerMemory {
    let epsilon = params[0];
    let momentum = params[1];

    if mem.len() != 0 {
        let memweight: Vec<Array2<fmod>> = (&(mem[0]).0)
            .into_iter()
            .zip(weight_grad.into_iter())
            .map(|(w_mem, w_grad)| momentum * w_mem + epsilon * w_grad)
            .collect();
        let membias: Vec<Array1<fmod>> = (&(mem[0]).1)
            .into_iter()
            .zip(bias_grad.into_iter())
            .map(|(b_mem, b_grad)| momentum * b_mem + epsilon * b_grad)
            .collect();

        vec![(memweight, membias)]
    } else {
        let memweight: Vec<Array2<fmod>> = weight_grad
            .into_iter()
            .map(|w_grad| epsilon * w_grad)
            .collect();
        let membias: Vec<Array1<fmod>> = bias_grad
            .into_iter()
            .map(|b_grad| epsilon * b_grad)
            .collect();

        vec![(memweight, membias)]
    }
}
#[inline(always)]
fn momentum_bias_update(
    _params: &Vec<fmod>,
    mem: &OptimizerMemory,
    _bias_grad: &Vec<Array1<fmod>>,
) -> BiasUpdate {
    mem[0].1.to_owned()
}
#[inline(always)]
fn momentum_weight_update(
    _params: &Vec<fmod>,
    mem: &OptimizerMemory,
    _weight_grad: &Vec<Array2<fmod>>,
) -> WeightUpdate {
    mem[0].0.to_owned()
}

impl Optimizer {
    pub fn momentum(eps: fmod, decay: fmod) -> Optimizer {
        Optimizer {
            parameters: vec![eps, decay],
            memoizer: momentum_memoizer,
            memory: vec![],
            bias_update_rule: momentum_bias_update,
            weight_update_rule: momentum_weight_update,
        }
    }
}
