use anyhow::Error;

use crate::sequences::SequenceType;
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

mod parsimony_alignment;

fn main() -> Result<()> {
    let sequences = io::read_sequences_from_file("./data/sequences_DNA3_2seqs.fasta").unwrap();
    let _alphabet = sequences::get_sequence_alphabet(&sequences);
    let setsx = sequences::get_parsimony_sets(&sequences[0], &SequenceType::DNA);
    let setsy = sequences::get_parsimony_sets(&sequences[1], &SequenceType::DNA);
    println!("{:?}", &setsx);
    println!("{:?}", &setsy);

    println!("{:}", &sequences[0]);
    println!("{:}", &sequences[1]);
    parsimony_alignment::pars_align(&setsx, &setsy);

    let nj_distances = sequences::compute_distance_matrix(&sequences);
    println!("{:?}", tree::build_nj_tree(nj_distances)?);
    Ok(())
}

#[cfg(test)]
mod tests;
