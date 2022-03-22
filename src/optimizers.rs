use crate::Array2;
use ndarray::Array1;

type OptimizerMemory = Vec<(Array2<f64>, Array1<f64>)>;
// memory, activations, gradients, errors -> memory
pub type Memoizer =
    fn(&OptimizerMemory, &Vec<Array1<f64>>, &Vec<Array1<f64>>, &Vec<Array1<f64>>) -> OptimizerMemory;

// memory, layer
pub type WeightUpdateRule = fn(OptimizerMemory, usize) -> Array1<f64>;
pub type BiasUpdateRule = fn(OptimizerMemory, usize) -> Array1<f64>;

pub struct Optimizer {
    pub memoizer: Memoizer,
    pub memory: OptimizerMemory,
    pub bias_update_rule: BiasUpdateRule,
    pub weight_update_rule: WeightUpdateRule,
}

impl Optimizer {
    pub fn memoize(
        &mut self,
        activations: &Vec<Array1<f64>>,
        gradients: &Vec<Array1<f64>>,
        errors: &Vec<Array1<f64>>,
    )  {
        let memoizer = self.memoizer;
        let memory = memoizer(&self.memory,activations,gradients,errors);
        self.memory = memory;
    }
}

