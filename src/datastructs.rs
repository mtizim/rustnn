use ndarray::Array1;

pub struct Data {
    pub x: Vec<Array1<f64>>,
    pub y: Vec<Array1<f64>>,
}
impl Data {
    pub fn ref_repr(&self) -> DataRef {
        DataRef {
            x: &self.x,
            y: &self.y,
        }
    }
}
pub struct DataRef<'a> {
    pub x: &'a Vec<Array1<f64>>,
    pub y: &'a Vec<Array1<f64>>,
}
pub struct Batch<'a> {
    pub x: &'a [Array1<f64>],
    pub y: &'a [Array1<f64>],
}
