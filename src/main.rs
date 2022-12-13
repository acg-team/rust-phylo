use nalgebra::{dmatrix, DMatrix};

mod njmat;
mod tree;

// TODO(naijulejshaja): Remove this check to enable dead code warnings.
#[allow(dead_code)]
#[derive(Debug)]
enum NJError {
    UnknownError(&'static str),
}

type Result<T> = std::result::Result<T, NJError>;
type Mat = DMatrix<f32>;

fn main() -> Result<()> {
    let distances = dmatrix![
        0.0, 5.0, 9.0, 9.0, 8.0;
        5.0, 0.0, 10.0, 10.0, 9.0;
        9.0, 10.0, 0.0, 8.0, 7.0;
        9.0, 10.0, 8.0, 0.0, 3.0;
        8.0, 9.0, 7.0, 3.0, 0.0];
    let nj_distances = njmat::NJMat {
        idx: (0..5).collect(),
        distances,
    };
    println!("{:?}", tree::build_nj_tree(nj_distances)?);
    Ok(())
}

#[cfg(test)]
mod tests;
