use crate::activations::*;
use crate::datastructs::*;
use crate::helpers::*;
use crate::mlpmodel::MultilayerPerceptron;
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
    {
        let mut convergence_data: Vec<(u32, f64)> = Vec::new();

        let data_train = helpers::read_data("data/square-simple-training.csv", 0);
        let data_test = helpers::read_data("data/square-simple-test.csv", 0);
        let shape = vec![1, 5, 5, 1];
        let activations = vec![
            NeuronActivation::sigmoid(),
            NeuronActivation::linear(),
            NeuronActivation::linear(),
        ];
        let mut mlp = MultilayerPerceptron::new(shape, activations, 0.0002);

        let n = 5000;
        for i in Prgrs::new(0..n, n) {
            mlp.train_epoch(&data_train.ref_repr(), 32);

            if i % 100 == 0 {
                let mse = mse(&mlp.predict(&data_test.x), &data_test.y);
                convergence_data.push((u32::try_from(i).unwrap(), mse));
            }
        }
        let yhats = mlp.predict(&data_test.x);
        let mse = mse(&yhats, &data_test.y);
        println!("MSE (square simple): {mse}");
    }
}
