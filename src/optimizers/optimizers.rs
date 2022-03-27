use crate::{fmod, mlpmodel::Gradients, Array2};
use ndarray::Array1;

pub type WeightUpdate = Vec<Array2<fmod>>;
pub type BiasUpdate = Vec<Array1<fmod>>;
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
) -> WeightUpdate;
pub type BiasUpdateRule = fn(
    &Vec<fmod>,
    &OptimizerMemory,
    &Vec<Array1<fmod>>, //
) -> BiasUpdate;

pub struct Optimizer {
    pub parameters: Vec<fmod>,
    pub memoizer: Memoizer,
    pub memory: OptimizerMemory,
    pub bias_update_rule: BiasUpdateRule,
    pub weight_update_rule: WeightUpdateRule,
}

impl Optimizer {
    fn memoize(
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

    fn get_weight_update(&self, gradients: &Vec<Array2<fmod>>) -> WeightUpdate {
        let rule = self.weight_update_rule;
        rule(&self.parameters, &self.memory, gradients)
    }
    fn get_bias_update(&self, gradients: &Vec<Array1<fmod>>) -> BiasUpdate {
        let rule = self.bias_update_rule;
        rule(&self.parameters, &self.memory, gradients)
    }

    pub fn get_updates(&mut self, gradients: &Gradients) -> (WeightUpdate, BiasUpdate) {
        self.memoize(&gradients.0, &gradients.1);
        (
            self.get_weight_update(&gradients.0),
            self.get_bias_update(&gradients.1),
        )
    }
}
