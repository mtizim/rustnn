#![allow(dead_code)]
use crate::activations::*;
use crate::datastructs::*;
use crate::helpers::*;
use crate::mlpmodel::MultilayerPerceptron;
use crate::optimizers::optimizers::Optimizer;
use ndarray::{Array, Array1, Array2};
use ndarray_linalg::{flatten, into_col};

use prgrs::Prgrs;

mod activations;
mod datastructs;
mod helpers;
mod mlpmodel;
mod optimizers;

#[allow(non_camel_case_types)]
type fmod = f64;

fn main() {
    // This is just copypasted code, faster to write than a for loop:
    // multimodal 1 hidden
    if false {
        {
            // sigmoid, linear, tanh, relu
            {
                let data_train = helpers::read_data_reg("data/multimodal-large-training.csv", 0);
                let data_test = helpers::read_data_reg("data/multimodal-large-test.csv", 0);
                let shape = vec![1, 16, 1];
                let activations = vec![NeuronActivation::sigmoid(), NeuronActivation::sigmoid()];

                let mut mlp = MultilayerPerceptron::new(
                    shape,
                    activations,
                    Optimizer::rmsprop(0.001, 0.99),
                    false,
                );

                let n = 1000;
                for _ in Prgrs::new(0..n, n) {
                    mlp.train_epoch(&data_train.ref_repr(), 32);
                }
                let yhats = mlp.predict(&data_test.x);

                let f1 = mse(&yhats, &data_test.y);
                println!("multimodal 1hid sigmoid: {:?}", f1);
            }
            {
                let data_train = helpers::read_data_reg("data/multimodal-large-training.csv", 0);
                let data_test = helpers::read_data_reg("data/multimodal-large-test.csv", 0);
                let shape = vec![1, 16, 1];
                let activations = vec![NeuronActivation::linear(), NeuronActivation::linear()];

                let mut mlp = MultilayerPerceptron::new(
                    shape,
                    activations,
                    Optimizer::rmsprop(0.001, 0.99),
                    false,
                );

                let n = 1000;
                for _ in Prgrs::new(0..n, n) {
                    mlp.train_epoch(&data_train.ref_repr(), 32);
                }
                let yhats = mlp.predict(&data_test.x);
                let f1 = mse(&yhats, &data_test.y);
                println!("multimodal 1hid linear: {:?}", f1);
            }
            {
                let data_train = helpers::read_data_reg("data/multimodal-large-training.csv", 0);
                let data_test = helpers::read_data_reg("data/multimodal-large-test.csv", 0);
                let shape = vec![1, 16, 1];
                let activations = vec![NeuronActivation::tanh(), NeuronActivation::tanh()];

                let mut mlp = MultilayerPerceptron::new(
                    shape,
                    activations,
                    Optimizer::rmsprop(0.001, 0.99),
                    false,
                );

                let n = 1000;
                for _ in Prgrs::new(0..n, n) {
                    mlp.train_epoch(&data_train.ref_repr(), 32);
                }
                let yhats = mlp.predict(&data_test.x);
                let f1 = mse(&yhats, &data_test.y);
                println!("multimodal 1hid tanh: {:?}", f1);
            }
            {
                let data_train = helpers::read_data_reg("data/multimodal-large-training.csv", 0);
                let data_test = helpers::read_data_reg("data/multimodal-large-test.csv", 0);
                let shape = vec![1, 16, 1];
                let activations = vec![NeuronActivation::relu(), NeuronActivation::relu()];

                let mut mlp = MultilayerPerceptron::new(
                    shape,
                    activations,
                    Optimizer::rmsprop(0.001, 0.99),
                    false,
                );

                let n = 1000;
                for _ in Prgrs::new(0..n, n) {
                    mlp.train_epoch(&data_train.ref_repr(), 32);
                }
                let yhats = mlp.predict(&data_test.x);
                let f1 = mse(&yhats, &data_test.y);
                println!("multimodal 1hid relu: {:?}", f1);
            }
        }
        // multimodal 2 hidden
        {
            // sigmoid, linear, tanh, relu
            {
                let data_train = helpers::read_data_reg("data/multimodal-large-training.csv", 0);
                let data_test = helpers::read_data_reg("data/multimodal-large-test.csv", 0);
                let shape = vec![1, 16, 16, 1];
                let activations = vec![
                    NeuronActivation::sigmoid(),
                    NeuronActivation::sigmoid(),
                    NeuronActivation::sigmoid(),
                ];

                let mut mlp = MultilayerPerceptron::new(
                    shape,
                    activations,
                    Optimizer::rmsprop(0.001, 0.99),
                    false,
                );

                let n = 1000;
                for _ in Prgrs::new(0..n, n) {
                    mlp.train_epoch(&data_train.ref_repr(), 32);
                }
                let yhats = mlp.predict(&data_test.x);
                let f1 = mse(&yhats, &data_test.y);
                println!("multimodal 2hid sigmoid: {:?}", f1);
            }
            {
                let data_train = helpers::read_data_reg("data/multimodal-large-training.csv", 0);
                let data_test = helpers::read_data_reg("data/multimodal-large-test.csv", 0);
                let shape = vec![1, 16, 16, 1];
                let activations = vec![
                    NeuronActivation::linear(),
                    NeuronActivation::linear(),
                    NeuronActivation::linear(),
                ];

                let mut mlp = MultilayerPerceptron::new(
                    shape,
                    activations,
                    Optimizer::rmsprop(0.001, 0.99),
                    false,
                );

                let n = 1000;
                for _ in Prgrs::new(0..n, n) {
                    mlp.train_epoch(&data_train.ref_repr(), 32);
                }
                let yhats = mlp.predict(&data_test.x);
                let f1 = mse(&yhats, &data_test.y);
                println!("multimodal 2hid linear: {:?}", f1);
            }
            {
                let data_train = helpers::read_data_reg("data/multimodal-large-training.csv", 0);
                let data_test = helpers::read_data_reg("data/multimodal-large-test.csv", 0);
                let shape = vec![1, 16, 16, 1];
                let activations = vec![
                    NeuronActivation::tanh(),
                    NeuronActivation::tanh(),
                    NeuronActivation::tanh(),
                ];

                let mut mlp = MultilayerPerceptron::new(
                    shape,
                    activations,
                    Optimizer::rmsprop(0.001, 0.99),
                    false,
                );

                let n = 1000;
                for _ in Prgrs::new(0..n, n) {
                    mlp.train_epoch(&data_train.ref_repr(), 32);
                }
                let yhats = mlp.predict(&data_test.x);
                let f1 = mse(&yhats, &data_test.y);
                println!("multimodal 2hid tanh: {:?}", f1);
            }
            {
                let data_train = helpers::read_data_reg("data/multimodal-large-training.csv", 0);
                let data_test = helpers::read_data_reg("data/multimodal-large-test.csv", 0);
                let shape = vec![1, 16, 16, 1];
                let activations = vec![
                    NeuronActivation::relu(),
                    NeuronActivation::relu(),
                    NeuronActivation::relu(),
                ];

                let mut mlp = MultilayerPerceptron::new(
                    shape,
                    activations,
                    Optimizer::rmsprop(0.001, 0.99),
                    false,
                );

                let n = 1000;
                for _ in Prgrs::new(0..n, n) {
                    mlp.train_epoch(&data_train.ref_repr(), 32);
                }
                let yhats = mlp.predict(&data_test.x);
                let f1 = mse(&yhats, &data_test.y);
                println!("multimodal 2hid relu: {:?}", f1);
            }
        }
        // multimodal 3 hidden
        {
            // sigmoid, linear, tanh, relu
            {
                let data_train = helpers::read_data_reg("data/multimodal-large-training.csv", 0);
                let data_test = helpers::read_data_reg("data/multimodal-large-test.csv", 0);
                let shape = vec![1, 16, 16, 16, 1];
                let activations = vec![
                    NeuronActivation::sigmoid(),
                    NeuronActivation::sigmoid(),
                    NeuronActivation::sigmoid(),
                    NeuronActivation::sigmoid(),
                ];

                let mut mlp = MultilayerPerceptron::new(
                    shape,
                    activations,
                    Optimizer::rmsprop(0.001, 0.99),
                    false,
                );

                let n = 1000;
                for _ in Prgrs::new(0..n, n) {
                    mlp.train_epoch(&data_train.ref_repr(), 32);
                }
                let yhats = mlp.predict(&data_test.x);
                let f1 = mse(&yhats, &data_test.y);
                println!("multimodal 3hid sigmoid: {:?}", f1);
            }
            {
                let data_train = helpers::read_data_reg("data/multimodal-large-training.csv", 0);
                let data_test = helpers::read_data_reg("data/multimodal-large-test.csv", 0);
                let shape = vec![1, 16, 16, 16, 1];
                let activations = vec![
                    NeuronActivation::linear(),
                    NeuronActivation::linear(),
                    NeuronActivation::linear(),
                    NeuronActivation::linear(),
                ];
                let mut mlp = MultilayerPerceptron::new(
                    shape,
                    activations,
                    Optimizer::rmsprop(0.001, 0.99),
                    false,
                );

                let n = 1000;
                for _ in Prgrs::new(0..n, n) {
                    mlp.train_epoch(&data_train.ref_repr(), 32);
                }
                let yhats = mlp.predict(&data_test.x);
                let f1 = mse(&yhats, &data_test.y);
                println!("multimodal 3hid linear: {:?}", f1);
            }
            {
                let data_train = helpers::read_data_reg("data/multimodal-large-training.csv", 0);
                let data_test = helpers::read_data_reg("data/multimodal-large-test.csv", 0);
                let shape = vec![1, 16, 16, 16, 1];
                let activations = vec![
                    NeuronActivation::tanh(),
                    NeuronActivation::tanh(),
                    NeuronActivation::tanh(),
                    NeuronActivation::tanh(),
                ];
                let mut mlp = MultilayerPerceptron::new(
                    shape,
                    activations,
                    Optimizer::rmsprop(0.001, 0.99),
                    false,
                );

                let n = 1000;
                for _ in Prgrs::new(0..n, n) {
                    mlp.train_epoch(&data_train.ref_repr(), 32);
                }
                let yhats = mlp.predict(&data_test.x);
                let f1 = mse(&yhats, &data_test.y);
                println!("multimodal 3hid tanh: {:?}", f1);
            }
            {
                let data_train = helpers::read_data_reg("data/multimodal-large-training.csv", 0);
                let data_test = helpers::read_data_reg("data/multimodal-large-test.csv", 0);
                let shape = vec![1, 16, 16, 16, 1];
                let activations = vec![
                    NeuronActivation::relu(),
                    NeuronActivation::relu(),
                    NeuronActivation::relu(),
                    NeuronActivation::relu(),
                ];
                let mut mlp = MultilayerPerceptron::new(
                    shape,
                    activations,
                    Optimizer::rmsprop(0.001, 0.99),
                    false,
                );

                let n = 1000;
                for _ in Prgrs::new(0..n, n) {
                    mlp.train_epoch(&data_train.ref_repr(), 32);
                }
                let yhats = mlp.predict(&data_test.x);
                let f1 = mse(&yhats, &data_test.y);
                println!("multimodal 3hid relu: {:?}", f1);
            }
        }
    }
    {
        // 1 - 2 hidden relu
        // 2 - 2 hidden sigmoid
        {
            // steps large 1
            {
                let data_train = helpers::read_data_reg("data/steps-large-training.csv", 1);
                let data_test = helpers::read_data_reg("data/steps-large-test.csv", 1);
                let shape = vec![1, 16, 16, 1];
                let activations = vec![
                    NeuronActivation::relu(),
                    NeuronActivation::relu(),
                    NeuronActivation::relu(),
                ];
                let mut mlp = MultilayerPerceptron::new(
                    shape,
                    activations,
                    Optimizer::rmsprop(0.001, 0.99),
                    false,
                );

                let n = 1000;
                for _ in Prgrs::new(0..n, n) {
                    mlp.train_epoch(&data_train.ref_repr(), 32);
                }
                let yhats = mlp.predict(&data_test.x);
                let f1 = mse(&yhats, &data_test.y);
                println!("steps large 3hid relu: {:?}", f1);
            }
            // steps large 2
            {
                let data_train = helpers::read_data_reg("data/steps-large-training.csv", 1);
                let data_test = helpers::read_data_reg("data/steps-large-test.csv", 1);
                let shape = vec![1, 16, 16, 1];
                let activations = vec![
                    NeuronActivation::sigmoid(),
                    NeuronActivation::sigmoid(),
                    NeuronActivation::sigmoid(),
                ];
                let mut mlp = MultilayerPerceptron::new(
                    shape,
                    activations,
                    Optimizer::rmsprop(0.001, 0.99),
                    false,
                );

                let n = 1000;
                for _ in Prgrs::new(0..n, n) {
                    mlp.train_epoch(&data_train.ref_repr(), 32);
                }
                let yhats = mlp.predict(&data_test.x);
                let f1 = mse(&yhats, &data_test.y);
                println!("steps large 3hid sigmoid: {:?}", f1);
            }
        }

        // rings3 regular 1
        // rings3 regular 2
        {
            {
                let data_train = helpers::read_data_r("data/rings3-regular-training.csv", 0, 3);
                let data_test = helpers::read_data_r("data/rings3-regular-test.csv", 0, 3);
                let shape = vec![3, 16, 16, 3];
                let activations = vec![
                    NeuronActivation::relu(),
                    NeuronActivation::relu(),
                    NeuronActivation::relu(),
                ];
                let mut mlp = MultilayerPerceptron::new(
                    shape,
                    activations,
                    Optimizer::rmsprop(0.001, 0.99),
                    true,
                );

                let n = 1000;
                for _ in Prgrs::new(0..n, n) {
                    mlp.train_epoch(&data_train.ref_repr(), 32);
                }
                let yhats = mlp.predict(&data_test.x);
                let f1 = f1(&yhats, &data_test.y, 3);
                println!("rings3 3hid relu: {:?}", f1);
            }
            {
                let data_train = helpers::read_data_r("data/rings3-regular-training.csv", 0, 3);
                let data_test = helpers::read_data_r("data/rings3-regular-test.csv", 0, 3);
                let shape = vec![3, 16, 16, 3];
                let activations = vec![
                    NeuronActivation::sigmoid(),
                    NeuronActivation::sigmoid(),
                    NeuronActivation::sigmoid(),
                ];
                let mut mlp = MultilayerPerceptron::new(
                    shape,
                    activations,
                    Optimizer::rmsprop(0.001, 0.99),
                    true,
                );

                let n = 1000;
                for _ in Prgrs::new(0..n, n) {
                    mlp.train_epoch(&data_train.ref_repr(), 32);
                }
                let yhats = mlp.predict(&data_test.x);
                let f1 = f1(&yhats, &data_test.y, 3);
                println!("rings3 3hid sigmoid: {:?}", f1);
            }
        }

        // rings5 regular 1
        // rings5 regular 2
        {
            // steps large 1
            {
                let data_train = helpers::read_data_r("data/rings5-regular-training.csv", 0, 5);
                let data_test = helpers::read_data_r("data/rings5-regular-test.csv", 0, 5);
                let shape = vec![3, 16, 16, 5];
                let activations = vec![
                    NeuronActivation::relu(),
                    NeuronActivation::relu(),
                    NeuronActivation::relu(),
                ];
                let mut mlp = MultilayerPerceptron::new(
                    shape,
                    activations,
                    Optimizer::rmsprop(0.001, 0.99),
                    true,
                );

                let n = 1000;
                for _ in Prgrs::new(0..n, n) {
                    mlp.train_epoch(&data_train.ref_repr(), 32);
                }
                let yhats = mlp.predict(&data_test.x);
                let f1 = f1(&yhats, &data_test.y, 5);
                println!("rings5 3hid relu: {:?}", f1);
            }
            // steps large 2
            {
                let data_train = helpers::read_data_r("data/rings5-regular-training.csv", 0, 5);
                let data_test = helpers::read_data_r("data/rings5-regular-test.csv", 0, 5);
                let shape = vec![3, 16, 16, 5];
                let activations = vec![
                    NeuronActivation::sigmoid(),
                    NeuronActivation::sigmoid(),
                    NeuronActivation::sigmoid(),
                ];
                let mut mlp = MultilayerPerceptron::new(
                    shape,
                    activations,
                    Optimizer::rmsprop(0.001, 0.99),
                    true,
                );

                let n = 1000;
                for _ in Prgrs::new(0..n, n) {
                    mlp.train_epoch(&data_train.ref_repr(), 32);
                }
                let yhats = mlp.predict(&data_test.x);
                let f1 = f1(&yhats, &data_test.y, 5);
                println!("rings5 3hid sigmoid: {:?}", f1);
            }
        }
    }
}

// OUTPUTS:
// [#################] (100%)
// multimodal 1hid sigmoid: 5350.473958102715
// [#################] (100%)
// multimodal 1hid linear: 4433.7955207233945
// [#################] (100%)
// multimodal 1hid tanh: 5343.132230095669
// [#################] (100%)
// multimodal 1hid relu: 4394.690082389213
// [#################] (100%)
// multimodal 2hid sigmoid: 5350.461373197856
// [#################] (100%)
// multimodal 2hid linear: 4433.982381981636
// [#################] (100%)
// multimodal 2hid tanh: 5282.115256402388
// [#################] (100%)
// multimodal 2hid relu: 4352.981228847567
// [#################] (100%)
// multimodal 3hid sigmoid: 5350.461291892413
// [#################] (100%)
// multimodal 3hid linear: 4434.508231247277
// [#################] (100%)
// multimodal 3hid tanh: 5319.794491544793
// [#################] (100%)
// multimodal 3hid relu: 7305.861128832839
//          The MSE is high because multimodal is not linear, and
//      it is not constrained to sigmoid/tanh ranges
//          It seems that tanh and sigmoid learn faster than relu,
//      but relu learns better, because its gradient doesn't vanish,
//      in the same way.
//
// [###########] (100%)
// steps large 3hid relu: 3642.05234309117
// [###########] (100%)
// steps large 3hid sigmoid: 7362.998112897957
//          the step function is out of range for a sigmoid activation,
//          and a 3 layer relu is better suited for prediction.
// [###########] (100%)
// rings3 3hid relu: [0.25870648, 0.6074498, 0.5508475]
// [###########] (100%)
// rings3 3hid sigmoid: [0.8037383, 0.775045, 0.6851727]
// [###########] (100%)
// rings5 3hid relu: [0.52840906, 0.6574395, 0.58779764, 0.5382114, 0.61290324]
// [###########] (100%)
// rings5 3hid sigmoid: [0.0, 0.7469512, 0.85148513, 0.88796103, 0.8271605]
//          relu does a good enough job, but the sigmoid is just better suited
//          for a classification task
