use std::fmt::Display;
use std::path::Path;

use std::fs;

use crate::alignment::Sequences;
use crate::alphabets::{dna_alphabet, protein_alphabet, Alphabet};
use crate::io::read_sequences_from_file;
use crate::likelihood::{ModelSearchCost, TreeSearchCost};
use crate::phylo_info::PhyloInfoBuilder as PIB;
use crate::pip_model::{PIPCost, PIPCostBuilder as PIPCB, PIPModel};
use crate::substitution_models::{
    dna_models::*, protein_models::*, QMatrix, QMatrixFactory, SubstModel, SubstitutionCost,
    SubstitutionCostBuilder as SCB,
};
use crate::tree;
use crate::tree::tree_parser::from_newick;

#[cfg(test)]
fn search_costs_equal_template<C: ModelSearchCost + TreeSearchCost>(cost: C) {
    assert_eq!(ModelSearchCost::cost(&cost), TreeSearchCost::cost(&cost));
}

#[cfg(test)]
fn test_subst_model<Q: QMatrix + QMatrixFactory + Clone + PartialEq + Display + 'static>(
    alpha: Alphabet,
    freqs: &[f64],
    params: &[f64],
) -> SubstitutionCost<Q> {
    // https://molevolworkshop.github.io/faculty/huelsenbeck/pdf/WoodsHoleHandout.pdf
    let fldr = Path::new("./data");
    let records =
        read_sequences_from_file(&fldr.join("Huelsenbeck_example_long_DNA.fasta")).unwrap();
    let seqs = Sequences::with_alphabet(records.clone(), alpha);
    let tree = tree!(&fs::read_to_string(fldr.join("Huelsenbeck_example.newick")).unwrap());
    let info = PIB::build_from_objects(seqs, tree).unwrap();
    let model = SubstModel::<Q>::new(freqs, params).unwrap();
    SCB::new(model, info).build().unwrap()
}

#[test]
fn dna_search_costs_equal() {
    search_costs_equal_template(test_subst_model::<JC69>(dna_alphabet(), &[], &[]));
    search_costs_equal_template(test_subst_model::<K80>(dna_alphabet(), &[], &[2.0]));
    search_costs_equal_template(test_subst_model::<HKY>(
        dna_alphabet(),
        &[0.22, 0.26, 0.33, 0.19],
        &[0.5],
    ));
    search_costs_equal_template(test_subst_model::<TN93>(
        dna_alphabet(),
        &[0.22, 0.26, 0.33, 0.19],
        &[0.5970915, 0.2940435, 0.00135],
    ));
    search_costs_equal_template(test_subst_model::<GTR>(
        dna_alphabet(),
        &[0.1, 0.3, 0.4, 0.2],
        &[5.0, 1.0, 1.0, 1.0, 1.0, 5.0],
    ));
}

#[test]
fn protein_search_costs_equal() {
    search_costs_equal_template(test_subst_model::<WAG>(protein_alphabet(), &[], &[]));
    search_costs_equal_template(test_subst_model::<HIVB>(protein_alphabet(), &[], &[]));
    search_costs_equal_template(test_subst_model::<BLOSUM>(protein_alphabet(), &[], &[]));
    let freqs = &[1.0 / 20.0; 20];
    search_costs_equal_template(test_subst_model::<WAG>(protein_alphabet(), freqs, &[]));
    search_costs_equal_template(test_subst_model::<HIVB>(protein_alphabet(), freqs, &[]));
    search_costs_equal_template(test_subst_model::<BLOSUM>(protein_alphabet(), freqs, &[]));
}

#[cfg(test)]
fn test_pip_model<Q: QMatrix + QMatrixFactory + Clone + PartialEq + Display + 'static>(
    alpha: Alphabet,
    freqs: &[f64],
    params: &[f64],
) -> PIPCost<Q> {
    // https://molevolworkshop.github.io/faculty/huelsenbeck/pdf/WoodsHoleHandout.pdf

    let fldr = Path::new("./data");
    let records =
        read_sequences_from_file(&fldr.join("Huelsenbeck_example_long_DNA.fasta")).unwrap();
    let seqs = Sequences::with_alphabet(records.clone(), alpha);
    let tree = tree!(&fs::read_to_string(fldr.join("Huelsenbeck_example.newick")).unwrap());
    let info = PIB::build_from_objects(seqs, tree).unwrap();
    let model = PIPModel::<Q>::new(freqs, params).unwrap();
    PIPCB::new(model, info).build().unwrap()
}

#[test]
fn dna_pip_search_costs_equal() {
    search_costs_equal_template(test_pip_model::<JC69>(dna_alphabet(), &[], &[1.2, 0.5]));
    search_costs_equal_template(test_pip_model::<K80>(dna_alphabet(), &[], &[1.2, 0.5, 2.0]));
    search_costs_equal_template(test_pip_model::<HKY>(
        dna_alphabet(),
        &[0.22, 0.26, 0.33, 0.19],
        &[1.2, 0.5, 0.5],
    ));
    search_costs_equal_template(test_pip_model::<TN93>(
        dna_alphabet(),
        &[0.22, 0.26, 0.33, 0.19],
        &[1.2, 0.5, 0.5970915, 0.2940435, 0.00135],
    ));
    search_costs_equal_template(test_pip_model::<GTR>(
        dna_alphabet(),
        &[0.1, 0.3, 0.4, 0.2],
        &[1.2, 0.5, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0],
    ));
}

#[test]
fn protein_pip_search_costs_equal() {
    search_costs_equal_template(test_pip_model::<WAG>(protein_alphabet(), &[], &[1.2, 0.5]));
    search_costs_equal_template(test_pip_model::<HIVB>(protein_alphabet(), &[], &[1.2, 0.5]));
    search_costs_equal_template(test_pip_model::<BLOSUM>(
        protein_alphabet(),
        &[],
        &[1.2, 0.5],
    ));
    let freqs = &[1.0 / 20.0; 20];
    search_costs_equal_template(test_pip_model::<WAG>(
        protein_alphabet(),
        freqs,
        &[1.2, 0.5],
    ));
    search_costs_equal_template(test_pip_model::<HIVB>(
        protein_alphabet(),
        freqs,
        &[1.2, 0.5],
    ));
    search_costs_equal_template(test_pip_model::<BLOSUM>(
        protein_alphabet(),
        freqs,
        &[1.2, 0.5],
    ));
}
