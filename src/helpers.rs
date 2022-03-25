use crate::datastructs::Data;
use crate::fmod;
use core::iter::zip;
use ndarray::arr1;
use ndarray::Array1;

pub fn read_data<P: AsRef<std::path::Path>>(path: P, startidx: usize) -> Data {
    let mut reader = csv::Reader::from_path(path).expect("Path existence");

    let mut xs = Vec::new();
    let mut ys = Vec::new();

    for result in reader.records() {
        let record = result.expect("a CSV record");

        let cx = (&record[startidx]).parse().expect("Formatting");
        let cy = (&record[startidx + 1]).parse().expect("Formatting");
        xs.push(arr1(&[cx]));
        ys.push(arr1(&[cy]));
    }
    Data { x: xs, y: ys }
}

pub fn mse(yhats: &Vec<Array1<fmod>>, ys: &Vec<Array1<fmod>>) -> fmod {
    //! assumes 1-element vectors for now
    let errs_squared: Vec<fmod> = zip(yhats, ys).map(|(yhat, y)| (yhat - y)[0]).collect();

    errs_squared
        .into_iter()
        .map(|a| a.powi(2))
        .reduce(|a, b| a + b)
        .unwrap()
        / yhats.len() as fmod
}
