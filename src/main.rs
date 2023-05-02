#[macro_use]
extern crate assert_float_eq;

use anyhow::Error;

mod io;
mod njmat;
mod sequences;
mod tree;
mod substitution_models;
mod parsimony_alignment;

use crate::substitution_models::SubstitutionModel;

type Result<T> = std::result::Result<T, Error>;
type Result2<T, E> = std::result::Result<T, E>;


fn main() -> Result<()> {
    let sequences = io::read_sequences_from_file("./data/sequences_protein1.fasta").unwrap();
    let sequence_type = sequences::get_sequence_type(&sequences);

    let nj_distances = sequences::compute_distance_matrix(&sequences);
    let tree = tree::build_nj_tree(nj_distances)?;

    let (alignment, scores) = parsimony_alignment::pars_align_on_tree(1.0, 2.0, 0.5, &tree, &sequences, &sequence_type);
    let msa = parsimony_alignment::compile_alignment(&tree, &sequences, &alignment, None);

    io::write_sequences_to_file(&msa, "msa.fasta")?;
    println!("Alignment scores are {:?}", scores);

    io::read_newick_from_string(&String::from("(((A:1.0,B:1.0)E:2.0,C:1.0)F:1.0,D:0.0)G:2.0;"))?;

    let wag = substitution_models::ProteinSubstModel::new("WAG").unwrap();
    println!("{:?}", wag.get_rate(b'A', b'A'));

    Ok(())
}

#[cfg(test)]
mod tests;