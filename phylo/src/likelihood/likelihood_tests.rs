use crate::likelihood::setup_dna_likelihood;
use crate::phylo_info::PhyloInfo;
use crate::tree::{NodeIdx::Leaf as L, Tree};
use approx::assert_relative_eq;
use bio::io::fasta::Record;

#[cfg(test)]
fn setup_simple_phylo_info(blen_i: f64, blen_j: f64) -> PhyloInfo {
    let sequences = vec![
        Record::with_attrs("A0", None, b"A"),
        Record::with_attrs("B1", None, b"A"),
    ];
    let mut tree = Tree::new(&sequences);
    tree.add_parent(0, L(0), L(1), blen_i, blen_j);
    tree.complete = true;
    tree.create_postorder();
    tree.create_preorder();
    PhyloInfo { tree, sequences }
}

#[test]
fn dna_simple_likelihood() {
    let info = setup_simple_phylo_info(1.0, 1.0);
    let mut likelihood = setup_dna_likelihood(&info, "JC69".to_string(), vec![], false).unwrap();
    assert_relative_eq!(
        likelihood.compute_log_likelihood(),
        -2.5832498829317445,
        epsilon = 1e-6
    );
    let info = setup_simple_phylo_info(1.0, 2.0);
    let mut likelihood = setup_dna_likelihood(&info, "JC69".to_string(), vec![], false).unwrap();
    assert_relative_eq!(
        likelihood.compute_log_likelihood(),
        -2.719098272533848,
        epsilon = 1e-6
    );
}

#[cfg(test)]
fn setup_cb_example_phylo_info() -> PhyloInfo {
    use crate::tree::tree_parser;
    let sequences = vec![
        Record::with_attrs("one", None, b"C"),
        Record::with_attrs("two", None, b"A"),
        Record::with_attrs("three", None, b"T"),
        Record::with_attrs("four", None, b"G"),
    ];
    let newick = "((one:2,two:2):1,(three:1,four:1):2);".to_string();
    let tree = tree_parser::from_newick_string(&newick)
        .unwrap()
        .pop()
        .unwrap();
    PhyloInfo { tree, sequences }
}

#[test]
fn dna_cb_example_likelihood() {
    let info = setup_cb_example_phylo_info();
    let mut likelihood = setup_dna_likelihood(
        &info,
        "tn93".to_string(),
        vec![0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135],
        false,
    )
    .unwrap();
    assert_relative_eq!(
        likelihood.compute_log_likelihood(),
        -17.1035117087,
        epsilon = 1e-6
    );
}

#[cfg(test)]
fn setup_mol_evo_example_phylo_info() -> PhyloInfo {
    use crate::tree::tree_parser;
    let sequences = vec![
        Record::with_attrs("one", None, b"T"),
        Record::with_attrs("two", None, b"C"),
        Record::with_attrs("three", None, b"A"),
        Record::with_attrs("four", None, b"C"),
        Record::with_attrs("five", None, b"C"),
    ];
    let newick = "(((one:0.2,two:0.2):0.1,three:0.2):0.1,(four:0.2,five:0.2):0.1);".to_string();
    let tree = tree_parser::from_newick_string(&newick)
        .unwrap()
        .pop()
        .unwrap();
    PhyloInfo { tree, sequences }
}

#[test]
fn dna_mol_evo_example_likelihood() {
    let info = setup_mol_evo_example_phylo_info();
    let mut likelihood = setup_dna_likelihood(&info, "k80".to_string(), vec![], true).unwrap();
    assert_relative_eq!(
        likelihood.compute_log_likelihood(),
        -7.581408,
        epsilon = 1e-6
    );
}
