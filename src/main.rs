use anyhow::Error;

extern crate bio;

mod io;
mod njmat;
mod sequences;
mod tree;

// TODO(naijulejshaja): Remove this check to enable dead code warnings.
// #[allow(dead_code)]
// #[derive(Debug)]
// enum NJError {
//     UnknownError(&'static str),
// }

type Result<T> = std::result::Result<T, Error>;

fn main() -> Result<()> {
    let sequences = io::read_sequences_from_file("./data/sequences_DNA2_unaligned.fasta").unwrap();
    let _alphabet = sequences::get_sequence_type(&sequences);

    let nj_distances = sequences::compute_distance_matrix(&sequences);
    println!("{:?}", tree::build_nj_tree(nj_distances)?);
    Ok(())
}

#[cfg(test)]
mod tests;
