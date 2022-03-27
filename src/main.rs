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
    // square large
    {
        {
            // Test data has a bigger range than train data, so it can't be predicted with a low mse
            let data_train = helpers::read_data("data/square-large-training.csv", 1);
            let data_test = helpers::read_data("data/square-large-training.csv", 1);
            let shape = vec![1, 5, 5, 1];
            let activations = vec![
                NeuronActivation::sigmoid(),
                NeuronActivation::linear(),
                NeuronActivation::linear(),
            ];
            let mut mlp = MultilayerPerceptron::new(shape, activations, Optimizer::sgd(0.0001));

            let n = 1000;
            for _ in Prgrs::new(0..n, n) {
                mlp.train_epoch(&data_train.ref_repr(), 32);
            }
            let yhats = mlp.predict(&data_test.x);
            let mse = mse(&yhats, &data_test.y);
            println!("MSE (square large sgd): {mse}");
        }
        {
            let data_train = helpers::read_data("data/square-large-training.csv", 1);
            let data_test = helpers::read_data("data/square-large-training.csv", 1);
            let shape = vec![1, 5, 5, 1];
            let activations = vec![
                NeuronActivation::sigmoid(),
                NeuronActivation::linear(),
                NeuronActivation::linear(),
            ];
            let mut mlp =
                MultilayerPerceptron::new(shape, activations, Optimizer::momentum(0.0001, 0.9));

            let n = 1000;
            for _ in Prgrs::new(0..n, n) {
                mlp.train_epoch(&data_train.ref_repr(), 32);
            }
            let yhats = mlp.predict(&data_test.x);
            let mse = mse(&yhats, &data_test.y);
            println!("MSE (square large momentum): {mse}");
        }
        {
            let data_train = helpers::read_data("data/square-large-training.csv", 1);
            let data_test = helpers::read_data("data/square-large-training.csv", 1);
            let shape = vec![1, 5, 5, 1];
            let activations = vec![
                NeuronActivation::sigmoid(),
                NeuronActivation::linear(),
                NeuronActivation::linear(),
            ];
            let mut mlp =
                MultilayerPerceptron::new(shape, activations, Optimizer::rmsprop(0.1, 0.99));

            let n = 1000;
            for _ in Prgrs::new(0..n, n) {
                mlp.train_epoch(&data_train.ref_repr(), 32);
            }
            let yhats = mlp.predict(&data_test.x);
            let mse = mse(&yhats, &data_test.y);
            println!("MSE (square large rmsprop): {mse}");
        }
    }
    // steps large
    {
        {
            let data_train = helpers::read_data("data/steps-large-training.csv", 1);
            let data_test = helpers::read_data("data/steps-large-training.csv", 1);
            let shape = vec![1, 15, 15, 15, 15, 1];
            let activations = vec![
                NeuronActivation::sigmoid(),
                NeuronActivation::sigmoid(),
                NeuronActivation::linear(),
                NeuronActivation::linear(),
                NeuronActivation::linear(),
            ];
            let mut mlp = MultilayerPerceptron::new(shape, activations, Optimizer::sgd(0.0001));

            let n = 1000;
            for _ in Prgrs::new(0..n, n) {
                mlp.train_epoch(&data_train.ref_repr(), 32);
            }
            let yhats = mlp.predict(&data_test.x);
            let mse = mse(&yhats, &data_test.y);
            println!("MSE (steps large sgd): {mse}");
        }
        {
            let data_train = helpers::read_data("data/steps-large-training.csv", 1);
            let data_test = helpers::read_data("data/steps-large-training.csv", 1);
            let shape = vec![1, 15, 15, 1];
            let activations = vec![
                NeuronActivation::sigmoid(),
                NeuronActivation::linear(),
                NeuronActivation::linear(),
            ];
            let mut mlp =
                MultilayerPerceptron::new(shape, activations, Optimizer::momentum(0.004, 0.9));

            let n = 1000;
            for _ in Prgrs::new(0..n, n) {
                mlp.train_epoch(&data_train.ref_repr(), 32);
            }
            let yhats = mlp.predict(&data_test.x);
            let mse = mse(&yhats, &data_test.y);
            println!("MSE (steps large momentum): {mse}");
        }
        {
            let data_train = helpers::read_data("data/steps-large-training.csv", 1);
            let data_test = helpers::read_data("data/steps-large-training.csv", 1);
            let shape = vec![1, 15, 15, 15, 15, 1];
            let activations = vec![
                NeuronActivation::sigmoid(),
                NeuronActivation::sigmoid(),
                NeuronActivation::linear(),
                NeuronActivation::linear(),
                NeuronActivation::linear(),
            ];
            let mut mlp =
                MultilayerPerceptron::new(shape, activations, Optimizer::rmsprop(0.01, 0.99));

            let n = 10000;
            for _ in Prgrs::new(0..n, n) {
                mlp.train_epoch(&data_train.ref_repr(), 32);
            }
            let yhats = mlp.predict(&data_test.x);
            let mse = mse(&yhats, &data_test.y);
            println!("MSE (steps large rmsprop): {mse}");
        }
    }
    // multimodal large
    {
        {
            let data_train = helpers::read_data("data/multimodal-large-training.csv", 0);
            let data_test = helpers::read_data("data/multimodal-large-test.csv", 0);
            let shape = vec![1, 15, 15, 15, 15, 1];
            let activations = vec![
                NeuronActivation::sigmoid(),
                NeuronActivation::sigmoid(),
                NeuronActivation::linear(),
                NeuronActivation::linear(),
                NeuronActivation::linear(),
            ];
            let mut mlp = MultilayerPerceptron::new(shape, activations, Optimizer::sgd(0.0001));

            let n = 1000;
            for _ in Prgrs::new(0..n, n) {
                mlp.train_epoch(&data_train.ref_repr(), 32);
            }
            let yhats = mlp.predict(&data_test.x);
            let mse = mse(&yhats, &data_test.y);
            println!("MSE (multimodal large sgd): {mse}");
        }
        {
            let data_train = helpers::read_data("data/multimodal-large-training.csv", 0);
            let data_test = helpers::read_data("data/multimodal-large-test.csv", 0);
            let shape = vec![1, 15, 15, 15, 15, 1];
            let activations = vec![
                NeuronActivation::sigmoid(),
                NeuronActivation::sigmoid(),
                NeuronActivation::linear(),
                NeuronActivation::linear(),
                NeuronActivation::linear(),
            ];
            let mut mlp =
                MultilayerPerceptron::new(shape, activations, Optimizer::momentum(0.0001, 0.9));

            let n = 1000;
            for _ in Prgrs::new(0..n, n) {
                mlp.train_epoch(&data_train.ref_repr(), 32);
            }
            let yhats = mlp.predict(&data_test.x);
            let mse = mse(&yhats, &data_test.y);
            println!("MSE (multimodal large momentum): {mse}");
        }
        {
            let data_train = helpers::read_data("data/multimodal-large-training.csv", 0);
            let data_test = helpers::read_data("data/multimodal-large-test.csv", 0);
            let shape = vec![1, 15, 15, 15, 15, 1];
            let activations = vec![
                NeuronActivation::sigmoid(),
                NeuronActivation::sigmoid(),
                NeuronActivation::linear(),
                NeuronActivation::linear(),
                NeuronActivation::linear(),
            ];
            let mut mlp =
                MultilayerPerceptron::new(shape, activations, Optimizer::rmsprop(0.01, 0.99));

            let n = 1000;
            for _ in Prgrs::new(0..n, n) {
                mlp.train_epoch(&data_train.ref_repr(), 32);
            }
            let yhats = mlp.predict(&data_test.x);
            let mse = mse(&yhats, &data_test.y);
            println!("MSE (multimodal large rmsprop): {mse}");
        }
    }
}

// OUTPUT: (note that some training times differ)
// [#############] (100%)
// MSE (square large sgd): 3.331925435962874
// [#############] (100%)
// MSE (square large momentum): 0.7833461236300092
// [#############] (100%)
// MSE (square large rmsprop): 131.96570713797897
//
// [#############] (100%)
// MSE (steps large sgd): 23.942884295272517
// [#############] (100%)
// MSE (steps large momentum): 6.696773729199052
// [#############] (100%)
// MSE (steps large rmsprop): 4.286258947173477
//
// [#############] (100%)
// MSE (multimodal large sgd): 2.2324397166955285
// [#############] (100%)
// MSE (multimodal large momentum): 1.6712454605684703
// [#############] (100%)
// MSE (multimodal large rmsprop): 2.0997478106784477
