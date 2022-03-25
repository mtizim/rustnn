use crate::{fmod, Array2};
use ndarray::Array1;

pub type OptimizerMemory = Vec<(Vec<Array2<fmod>>, Vec<Array1<fmod>>)>;
// parameters, memory, weight_gradients[layer], bias_gradients[layer]
pub type Memoizer = fn(
    &Vec<fmod>,
    &OptimizerMemory,
    &Vec<Array2<fmod>>,
    &Vec<Array1<fmod>>, //
) -> OptimizerMemory;

// parameters, memory, grads
pub type WeightUpdateRule = fn(
    &Vec<fmod>,
    &OptimizerMemory,
    &Vec<Array2<fmod>>, //
) -> Vec<Array2<fmod>>;
pub type BiasUpdateRule = fn(
    &Vec<fmod>,
    &OptimizerMemory,
    &Vec<Array1<fmod>>, //
) -> Vec<Array1<fmod>>;

pub struct Optimizer {
    pub parameters: Vec<fmod>,
    pub memoizer: Memoizer,
    pub memory: OptimizerMemory,
    pub bias_update_rule: BiasUpdateRule,
    pub weight_update_rule: WeightUpdateRule,
}

impl Optimizer {
    pub fn memoize(
        &mut self,
        weight_gradients: &Vec<Array2<fmod>>,
        bias_gradients: &Vec<Array1<fmod>>,
    ) {
        let memoizer = self.memoizer;
        let memory = memoizer(
            &self.parameters,
            &self.memory,
            weight_gradients,
            bias_gradients,
        );
        self.memory = memory;
    }

    pub fn get_weight_deltas(
        &self,
        activations: &Array1<fmod>,
        gradients: &Array1<fmod>,
        errors: &Array1<fmod>,
    ) -> Array1<fmod> {
        activations.to_owned()
    }
    pub fn get_bias_deltas(
        &self,
        activations: &Array1<fmod>,
        gradients: &Array1<fmod>,
        errors: &Array1<fmod>,
    ) -> Array1<fmod> {
        activations.to_owned()
    }
}
