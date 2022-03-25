use ndarray::{arr1, arr2, Array1, Array2};

use crate::fmod;

use super::optimizers::*;

#[inline(always)]
fn sgd_memoizer(
    _params: &Vec<fmod>,
    _mem: &OptimizerMemory,
    _weight_grad: &Vec<Array2<fmod>>,
    _bias_grad: &Vec<Array1<fmod>>,
) -> OptimizerMemory {
    vec![]
}
#[inline(always)]
fn sgd_bias_update(
    params: &Vec<fmod>,
    _mem: &OptimizerMemory,
    bias_grad: &Vec<Array1<fmod>>,
) -> Vec<Array1<fmod>> {
    let eps = params[0];
    bias_grad
        .into_iter()
        .map(|layergrad| eps * layergrad)
        .collect()
}
#[inline(always)]
fn sgd_weight_update(
    params: &Vec<fmod>,
    _mem: &OptimizerMemory,
    weight_grad: &Vec<Array2<fmod>>,
) -> Vec<Array2<fmod>> {
    let eps = params[0];
    weight_grad
        .into_iter()
        .map(|layergrad| eps * layergrad)
        .collect()
}

impl Optimizer {
    pub fn sgd(eps: fmod) -> Optimizer {
        Optimizer {
            parameters: vec![eps],
            memoizer: sgd_memoizer,
            memory: vec![],
            bias_update_rule: sgd_bias_update,
            weight_update_rule: sgd_weight_update,
        }
    }
}
