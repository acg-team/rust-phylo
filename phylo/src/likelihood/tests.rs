use std::fmt::Display;
use std::path::Path;

use approx::assert_relative_eq;
use bio::io::fasta::Record;

use crate::alignment::Sequences;
use crate::alphabets::Alphabet;
use crate::evolutionary_models::EvoModel;
use crate::io::read_sequences_from_file;
use crate::likelihood::PhyloCostFunction;
use crate::phylo_info::{PhyloInfo, PhyloInfoBuilder as PIB};
use crate::pip_model::PIPModel;
use crate::substitution_models::{
    dna_models::*, protein_models::*, FreqVector, QMatrix, SubstMatrix, SubstModel,
};
use crate::tree::{tree_parser::from_newick, Tree};
use crate::{frequencies, record_wo_desc as record, tree};

#[cfg(test)]
fn setup_simple_phylo_info(blen_i: f64, blen_j: f64) -> PhyloInfo {
    let sequences = Sequences::new(vec![record!("A0", b"A"), record!("B1", b"A")]);
    let tree = tree!(format!("((A0:{},B1:{}):1.0);", blen_i, blen_j).as_str());
    PIB::build_from_objects(sequences, tree).unwrap()
}

#[test]
fn dna_simple_likelihood() {
    let info = &setup_simple_phylo_info(1.0, 1.0);
    let jc69 = SubstModel::<JC69>::new(&[], &[]).unwrap();
    assert_relative_eq!(jc69.cost(info, false), -2.5832498829317445, epsilon = 1e-6);
    let jc69 = SubstModel::<JC69>::new(&[], &[]).unwrap();
    let info = &setup_simple_phylo_info(1.0, 2.0);
    assert_relative_eq!(jc69.cost(info, false), -2.719098272533848, epsilon = 1e-6);
}

#[cfg(test)]
fn change_logl_on_freq_change_template<Q: QMatrix + PartialEq + Display>(
    freqs: &[f64],
    params: &[f64],
) {
    // likelihood should change when frequencies are changed in models with free freqs
    let info = setup_cb_example_phylo_info();
    let mut model: SubstModel<Q> = SubstModel::<Q>::new(freqs, params).unwrap();
    let logl = model.cost(&info, false);
    model.set_freqs(frequencies!(&[0.1, 0.2, 0.3, 0.4]));
    assert_ne!(logl, model.cost(&info, true));
}

#[test]
fn change_logl_on_freq_change() {
    change_logl_on_freq_change_template::<HKY>(&[0.22, 0.26, 0.33, 0.19], &[0.5]);
    change_logl_on_freq_change_template::<TN93>(&[0.22, 0.26, 0.33, 0.19], &[0.5]);
    change_logl_on_freq_change_template::<GTR>(&[0.22, 0.26, 0.33, 0.19], &[0.5]);
}

#[cfg(test)]
fn same_logl_on_freq_change_template<Q: QMatrix + PartialEq + Display>(
    freqs: &[f64],
    params: &[f64],
) {
    // likelihood should change when frequencies are changed in models with free freqs
    let info = setup_cb_example_phylo_info();
    let mut model: SubstModel<Q> = SubstModel::<Q>::new(freqs, params).unwrap();
    let logl = model.cost(&info, false);
    model.set_freqs(frequencies!(&[0.1, 0.2, 0.3, 0.4]));
    assert_eq!(logl, model.cost(&info, true));
}

#[test]
fn same_logl_on_freq_change() {
    same_logl_on_freq_change_template::<JC69>(&[0.22, 0.26, 0.33, 0.19], &[2.0]);
    same_logl_on_freq_change_template::<K80>(&[0.22, 0.26, 0.33, 0.19], &[]);
}

#[cfg(test)]
fn change_logl_on_param_change_template<T: QMatrix + PartialEq + Display>(
    freqs: &[f64],
    params: &[f64],
) {
    // likelihood should change when frequencies are changed in models with free freqs
    let info = setup_cb_example_phylo_info();
    let mut model: SubstModel<T> = SubstModel::<T>::new(freqs, params).unwrap();
    let logl = model.cost(&info, false);
    model.set_param(0, 100.0);
    assert_ne!(logl, model.cost(&info, true));
}

#[test]
fn change_logl_on_param_change() {
    change_logl_on_param_change_template::<K80>(&[], &[2.0]);
    change_logl_on_param_change_template::<HKY>(&[0.22, 0.26, 0.33, 0.19], &[0.5]);
    change_logl_on_param_change_template::<TN93>(
        &[0.22, 0.26, 0.33, 0.19],
        &[0.5970915, 0.2940435, 0.00135],
    );
    change_logl_on_param_change_template::<GTR>(
        &[0.1, 0.3, 0.4, 0.2],
        &[5.0, 1.0, 1.0, 1.0, 1.0, 5.0],
    );
}

#[test]
fn same_likelihood_on_param_change() {
    // likelihood should not change when parameters are changed for jc69
    let info = setup_cb_example_phylo_info();
    let mut model = SubstModel::<JC69>::new(&[], &[]).unwrap();
    let logl = model.cost(&info, false);
    model.set_param(0, 100.0);
    assert_eq!(logl, model.cost(&info, true));
}

#[cfg(test)]
fn dna_gaps_as_ambigs_template<Q: QMatrix + Display>(freqs: &[f64], params: &[f64]) {
    let tree = tree!("((one:2,two:2):1,(three:1,four:1):2);");
    let sequences = Sequences::new(vec![
        record!("one", b"CCCCCCXX"),
        record!("two", b"XXAAAAAA"),
        record!("three", b"TTTNNTTT"),
        record!("four", b"GNGGGGNG"),
    ]);
    let info_ambig = &PIB::build_from_objects(sequences, tree.clone()).unwrap();
    let sequences = Sequences::new(vec![
        record!("one", b"CCCCCC--"),
        record!("two", b"--AAAAAA"),
        record!("three", b"TTT--TTT"),
        record!("four", b"G-GGGG-G"),
    ]);
    let info_gaps = &PIB::build_from_objects(sequences, tree.clone()).unwrap();

    let model = SubstModel::<Q>::new(freqs, params).unwrap();
    assert_eq!(model.cost(info_ambig, true), model.cost(info_gaps, true));
}

#[test]
fn dna_gaps_as_ambigs() {
    dna_gaps_as_ambigs_template::<JC69>(&[], &[]);
    dna_gaps_as_ambigs_template::<K80>(&[], &[]);
    dna_gaps_as_ambigs_template::<HKY>(&[0.22, 0.26, 0.33, 0.19], &[0.5]);
    dna_gaps_as_ambigs_template::<TN93>(
        &[0.22, 0.26, 0.33, 0.19],
        &[0.5970915, 0.2940435, 0.00135],
    );
    dna_gaps_as_ambigs_template::<GTR>(&[0.1, 0.3, 0.4, 0.2], &[5.0, 1.0, 1.0, 1.0, 1.0, 5.0]);
}

#[cfg(test)]
fn setup_phylo_info_single_leaf() -> PhyloInfo {
    let sequences = Sequences::new(vec![record!("A0", b"AAAAAA")]);
    let tree = Tree::new(&sequences).unwrap();
    PIB::build_from_objects(sequences, tree).unwrap()
}

#[test]
fn dna_likelihood_one_node() {
    let info = &setup_phylo_info_single_leaf();
    let model = SubstModel::<JC69>::new(&[], &[]).unwrap();
    assert!(model.cost(info, false) < 0.0);
}

#[cfg(test)]
fn setup_cb_example_phylo_info() -> PhyloInfo {
    let sequences = Sequences::new(vec![
        record!("one", b"C"),
        record!("two", b"A"),
        record!("three", b"T"),
        record!("four", b"G"),
    ]);
    let newick = "((one:2,two:2):1,(three:1,four:1):2);".to_string();
    PIB::build_from_objects(sequences, tree!(&newick)).unwrap()
}

#[test]
fn dna_cb_example_likelihood() {
    let info = &setup_cb_example_phylo_info();
    let mut model =
        SubstModel::<TN93>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5970915, 0.2940435, 0.00135])
            .unwrap();
    model.qmatrix.q = SubstMatrix::from_row_slice(
        4,
        4,
        &[
            -0.15594579,
            0.15524379,
            0.00044550000000000004,
            0.0002565,
            0.13136013,
            -0.13206213,
            0.00044550000000000004,
            0.0002565,
            0.000297,
            0.000351,
            -0.056516265,
            0.055868265,
            0.000297,
            0.000351,
            0.097034355,
            -0.097682355,
        ],
    );
    assert_relative_eq!(model.cost(info, false), -17.1035117087, epsilon = 1e-6);
}

#[cfg(test)]
fn setup_mol_evo_example_phylo_info() -> PhyloInfo {
    let sequences = Sequences::new(vec![
        record!("one", b"T"),
        record!("two", b"C"),
        record!("three", b"A"),
        record!("four", b"C"),
        record!("five", b"C"),
    ]);
    let newick = "(((one:0.2,two:0.2):0.1,three:0.2):0.1,(four:0.2,five:0.2):0.1);".to_string();
    PIB::build_from_objects(sequences, tree!(&newick)).unwrap()
}

#[test]
fn dna_mol_evo_example_likelihood() {
    let info = &setup_mol_evo_example_phylo_info();
    let model = SubstModel::<K80>::new(&[], &[]).unwrap();
    assert_relative_eq!(model.cost(info, false), -7.581408, epsilon = 1e-6);
}

#[cfg(test)]
fn dna_ambig_example_logl_template<Q: QMatrix + Display>(freqs: &[f64], params: &[f64]) {
    // Checks that likelihoods for different ambiguous characters are the same
    let fldr = Path::new("./data");
    let info_w_x = &PIB::with_attrs(
        fldr.join("ambiguous_example.fasta"),
        fldr.join("ambiguous_example.newick"),
    )
    .build()
    .unwrap();

    let info_w_n = &PIB::with_attrs(
        fldr.join("ambiguous_example_N.fasta"),
        fldr.join("ambiguous_example.newick"),
    )
    .build()
    .unwrap();

    let model = SubstModel::<Q>::new(freqs, params).unwrap();
    assert_relative_eq!(model.cost(info_w_x, true), model.cost(info_w_n, true));
}

#[test]
fn dna_ambig_example_likelihood() {
    dna_ambig_example_logl_template::<JC69>(&[], &[]);
    dna_ambig_example_logl_template::<K80>(&[], &[]);
    dna_ambig_example_logl_template::<HKY>(&[0.22, 0.26, 0.33, 0.19], &[0.5]);
    dna_ambig_example_logl_template::<TN93>(
        &[0.22, 0.26, 0.33, 0.19],
        &[0.5970915, 0.2940435, 0.00135],
    );
    dna_ambig_example_logl_template::<GTR>(&[0.1, 0.3, 0.4, 0.2], &[5.0, 1.0, 1.0, 1.0, 1.0, 5.0]);
}

#[test]
fn dna_ambig_example_likelihood_k80() {
    // Checks the exact value for k80
    let fldr = Path::new("./data");
    let info_w_x = &PIB::with_attrs(
        fldr.join("ambiguous_example.fasta"),
        fldr.join("ambiguous_example.newick"),
    )
    .build()
    .unwrap();
    let k80 = SubstModel::<K80>::new(&[], &[2.0, 1.0]).unwrap();
    assert_relative_eq!(
        k80.cost(info_w_x, true),
        -137.24280493914029,
        epsilon = 1e-6
    );
}

#[test]
fn dna_huelsenbeck_example_likelihood() {
    // https://molevolworkshop.github.io/faculty/huelsenbeck/pdf/WoodsHoleHandout.pdf
    let fldr = Path::new("./data");
    let info = &PIB::with_attrs(
        fldr.join("Huelsenbeck_example_long_DNA.fasta"),
        fldr.join("Huelsenbeck_example.newick"),
    )
    .build()
    .unwrap();
    let hky = SubstModel::<HKY>::new(&[0.1, 0.3, 0.4, 0.2], &[5.0]).unwrap();
    assert_relative_eq!(hky.cost(info, true), -216.234734, epsilon = 1e-3);
    let gtr_as_hky =
        SubstModel::<GTR>::new(&[0.1, 0.3, 0.4, 0.2], &[5.0, 1.0, 1.0, 1.0, 1.0, 5.0]).unwrap();
    assert_relative_eq!(gtr_as_hky.cost(info, true), -216.234734, epsilon = 1e-3);
}

#[cfg(test)]
fn protein_example_logl_template<Q: QMatrix + Display>(
    params: &[f64],
    expected_llik: f64,
    epsilon: f64,
) {
    let fldr = Path::new("./data/phyml_protein_example");
    let info = &PIB::with_attrs(fldr.join("nogap_seqs.fasta"), fldr.join("true_tree.newick"))
        .build()
        .unwrap();
    let model = SubstModel::<Q>::new(&[], params).unwrap();
    assert_relative_eq!(model.cost(info, false), expected_llik, epsilon = epsilon);
}

#[test]
fn protein_example_likelihood() {
    protein_example_logl_template::<WAG>(&[], -4505.736814460457, 1e-3);
    protein_example_logl_template::<HIVB>(&[], -4407.989226397638, 1e-5);
    protein_example_logl_template::<BLOSUM>(&[], -4587.71053, 1e-5);
}

#[cfg(test)]
fn simple_reroot_info(alphabet: &Alphabet) -> (PhyloInfo, PhyloInfo) {
    let sequences = Sequences::with_alphabet(
        vec![
            record!("A", b"CTATATATACIJL"),
            record!("B", b"ATATATATAAIHL"),
            record!("C", b"TTATATATATIJL"),
        ],
        alphabet.clone(),
    );
    let info = PIB::build_from_objects(sequences.clone(), tree!("((A:2.0,B:2.0):1.0,C:2.0):0.0;"))
        .unwrap();
    let info_rerooted =
        PIB::build_from_objects(sequences, tree!("(A:1.0,(B:2.0,C:3.0):1.0):0.0;")).unwrap();
    (info, info_rerooted)
}

#[cfg(test)]
fn logl_revers_template<Q: QMatrix + Display>(freqs: &[f64], params: &[f64], epsilon: f64) {
    let model = SubstModel::<Q>::new(freqs, params).unwrap();
    let (info, info_rerooted) = &simple_reroot_info(model.qmatrix.alphabet());
    assert_relative_eq!(
        model.cost(info, false),
        model.cost(info_rerooted, true),
        epsilon = epsilon,
    );
}

#[test]
fn dna_logl_reversibility() {
    logl_revers_template::<JC69>(&[], &[], 1e-10);
    logl_revers_template::<K80>(&[], &[], 1e-10);
    logl_revers_template::<HKY>(&[0.22, 0.26, 0.33, 0.19], &[0.5], 1e-10);
    logl_revers_template::<TN93>(
        &[0.22, 0.26, 0.33, 0.19],
        &[0.5970915, 0.2940435, 0.00135],
        1e-10,
    );
    logl_revers_template::<GTR>(
        &[0.22, 0.26, 0.33, 0.19],
        &[5.0, 1.0, 1.0, 1.0, 1.0, 5.0],
        1e-10,
    );
}

#[test]
fn protein_logl_reversibility() {
    logl_revers_template::<WAG>(&[], &[], 1e-8);
    logl_revers_template::<HIVB>(&[], &[], 1e-8);
    logl_revers_template::<BLOSUM>(&[], &[], 1e-3);
}

fn huelsenbeck_reversibility_template<Q: QMatrix + Display>(freqs: &[f64], params: &[f64]) {
    // https://molevolworkshop.github.io/faculty/huelsenbeck/pdf/WoodsHoleHandout.pdf
    let fldr = Path::new("./data");
    let info = &PIB::with_attrs(
        fldr.join("Huelsenbeck_example_long_DNA.fasta"),
        fldr.join("Huelsenbeck_example.newick"),
    )
    .build()
    .unwrap();
    let info_rerooted = &PIB::with_attrs(
        fldr.join("Huelsenbeck_example_long_DNA.fasta"),
        fldr.join("Huelsenbeck_example_reroot.newick"),
    )
    .build()
    .unwrap();
    let model = SubstModel::<Q>::new(freqs, params).unwrap();
    assert_relative_eq!(
        model.cost(info, false),
        model.cost(info_rerooted, true),
        epsilon = 1e-10,
    );
}

#[test]
fn huelsenbeck_reversibility() {
    huelsenbeck_reversibility_template::<JC69>(&[], &[]);
    huelsenbeck_reversibility_template::<K80>(&[], &[0.5]);
    huelsenbeck_reversibility_template::<HKY>(&[0.22, 0.26, 0.33, 0.19], &[0.5]);
    huelsenbeck_reversibility_template::<TN93>(
        &[0.22, 0.26, 0.33, 0.19],
        &[0.5970915, 0.2940435, 0.00135],
    );
    huelsenbeck_reversibility_template::<GTR>(
        &[0.1, 0.3, 0.4, 0.2],
        &[5.0, 1.0, 1.0, 1.0, 1.0, 5.0],
    );
}

#[test]
fn pip_likelihood_correct_after_reset() {
    let tree1 = tree!("(((A:1.0,B:1.0)E:2.0,(C:1.0,D:1.0)F:2.0)G:3.0);");
    let tree2 = tree!("(((A:2.0,B:2.0)E:4.0,(C:2.0,D:2.0)F:4.0)G:6.0);");
    let sequences = Sequences::new(vec![
        record!("A", b"P"),
        record!("B", b"P"),
        record!("C", b"P"),
        record!("D", b"P"),
    ]);
    let info1 = PIB::build_from_objects(sequences.clone(), tree1).unwrap();
    let info2 = PIB::build_from_objects(sequences, tree2).unwrap();

    let pip_wag = PIPModel::<WAG>::new(&[], &[50.0, 0.1]).unwrap();
    let c1 = pip_wag.cost(&info1, false);
    assert_relative_eq!(c1, -1004.2260753055999, epsilon = 1e-5);
    assert_relative_eq!(c1, pip_wag.cost(&info1, true), epsilon = 1e-5);

    let c2 = pip_wag.cost(&info2, true);
    assert_ne!(c1, c2);
    assert_relative_eq!(c2, -1425.1290016747846, epsilon = 1e-5);
    assert_relative_eq!(c2, pip_wag.cost(&info2, true), epsilon = 1e-5);
}

#[cfg(test)]
fn logl_correct_after_reset_template<Q: QMatrix + Display>(llik1: f64, llik2: f64) {
    let tree1 = tree!("(((A:1.0,B:1.0)E:2.0,(C:1.0,D:1.0)F:2.0)G:3.0);");
    let tree2 = tree!("(((A:2.0,B:2.0)E:4.0,(C:2.0,D:2.0)F:4.0)G:6.0);");
    let sequences = Sequences::new(vec![
        record!("A", b"P"),
        record!("B", b"P"),
        record!("C", b"P"),
        record!("D", b"P"),
    ]);
    let info1 = PIB::build_from_objects(sequences.clone(), tree1).unwrap();
    let info2 = PIB::build_from_objects(sequences, tree2).unwrap();

    let model = SubstModel::<Q>::new(&[], &[]).unwrap();
    let c1 = model.cost(&info1, false);
    assert_relative_eq!(c1, llik1, epsilon = 1e-5);
    assert_relative_eq!(c1, model.cost(&info1, true), epsilon = 1e-5);

    let c2 = model.cost(&info2, true);
    assert_ne!(c1, c2);
    assert_relative_eq!(c2, llik2, epsilon = 1e-5);
    assert_relative_eq!(c2, model.cost(&info2, true), epsilon = 1e-5);
}

#[test]
fn protein_logl_correct_after_reset() {
    logl_correct_after_reset_template::<WAG>(-7.488595394504073, -10.206456536551775);
    logl_correct_after_reset_template::<HIVB>(-7.231597482410509, -9.865559545434952);
    logl_correct_after_reset_template::<BLOSUM>(-7.4408154253528975, -10.33187874481282);
}

fn one_site_one_char_template<Q: QMatrix + Display>(freqs: &[f64], params: &[f64]) {
    // This used to fail on leaf data creation when some of the sequences were empty
    let model = SubstModel::<Q>::new(freqs, params).unwrap();
    let sequences = Sequences::with_alphabet(
        vec![
            record!("one", b"C"),
            record!("two", b"-"),
            record!("three", b"-"),
            record!("four", b"-"),
        ],
        model.qmatrix.alphabet().clone(),
    );
    let tree = tree!("((one:2,two:2):1,(three:1,four:1):2);");
    let info = PIB::build_from_objects(sequences, tree).unwrap();

    let logl = model.cost(&info, false);
    assert_ne!(logl, f64::NEG_INFINITY);
    assert!(logl < 0.0);
}

#[test]
fn dna_one_site_one_char() {
    one_site_one_char_template::<JC69>(&[], &[]);
    one_site_one_char_template::<K80>(&[], &[]);
    one_site_one_char_template::<HKY>(&[0.22, 0.26, 0.33, 0.19], &[0.5]);
    one_site_one_char_template::<TN93>(&[0.22, 0.26, 0.33, 0.19], &[0.5970915, 0.2940435, 0.00135]);
    one_site_one_char_template::<GTR>(&[0.1, 0.3, 0.4, 0.2], &[5.0, 1.0, 1.0, 1.0, 1.0, 5.0]);
}

#[test]
fn protein_one_site_one_char() {
    one_site_one_char_template::<WAG>(&[], &[]);
    one_site_one_char_template::<HIVB>(&[], &[]);
    one_site_one_char_template::<BLOSUM>(&[], &[]);
}

#[test]
fn hiv_subset_valid_subst_likelihood() {
    let fldr = Path::new("./data/real_examples/");
    let alignment = fldr.join("HIV-1_env_DNA_mafft_alignment_subset.fasta");
    let info = PIB::new(alignment).build().unwrap();
    let gtr =
        SubstModel::<GTR>::new(&[0.25, 0.25, 0.25, 0.25], &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
    let logl = gtr.cost(&info, false);
    assert_ne!(logl, f64::NEG_INFINITY);
    assert!(logl < 0.0);
}

#[test]
fn hiv_subset_valid_pip_likelihood() {
    let fldr = Path::new("./data/real_examples/");
    let alignment = fldr.join("HIV-1_env_DNA_mafft_alignment_subset.fasta");
    let info = PIB::new(alignment).build().unwrap();
    let pip = PIPModel::<GTR>::new(
        &[0.25, 0.25, 0.25, 0.25],
        &[0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let logl = pip.cost(&info, false);
    assert_ne!(logl, f64::NEG_INFINITY);
    assert!(logl < 0.0);
}

#[test]
fn dna_gaps_against_phyml() {
    let newick =
        "(C:0.06465432,D:27.43128366,(A:0.00000001,B:0.00000001)0.000000:0.08716381);".to_string();
    let sequences = Sequences::new(
        read_sequences_from_file(&Path::new("./data/").join("sequences_DNA1.fasta")).unwrap(),
    );
    let info = PIB::build_from_objects(sequences, tree!(&newick)).unwrap();
    let jc69 = SubstModel::<JC69>::new(&[], &[]).unwrap();

    // Compare against value from PhyML
    assert_relative_eq!(jc69.cost(&info, false), -9.70406054783923);
}

#[test]
fn dna_single_char_gaps_against_phyml() {
    let jc69 = SubstModel::<JC69>::new(&[], &[]).unwrap();
    let tree =
        tree!("(C:0.06465432,D:27.43128366,(A:0.00000001,B:0.00000001)0.000000:0.08716381);");

    let sequences = Sequences::new(vec![
        record!("A", b"A"),
        record!("B", b"A"),
        record!("C", b"A"),
        record!("D", b"T"),
    ]);
    let info = PIB::build_from_objects(sequences, tree.clone()).unwrap();
    assert_relative_eq!(jc69.cost(&info, true), -2.920437792326963); // Compare against PhyML

    let sequences = Sequences::new(vec![
        record!("A", b"X"),
        record!("B", b"X"),
        record!("C", b"X"),
        record!("D", b"X"),
    ]);
    let info = PIB::build_from_objects(sequences, tree.clone()).unwrap();
    assert_relative_eq!(jc69.cost(&info, true), 0.0, epsilon = 1e-5); // Compare against PhyML

    let sequences = Sequences::new(vec![
        record!("A", b"X"),
        record!("B", b"X"),
        record!("C", b"X"),
        record!("D", b"T"),
    ]);
    let info = PIB::build_from_objects(sequences, tree.clone()).unwrap();
    assert_relative_eq!(jc69.cost(&info, true), -1.38629, epsilon = 1e-5); // Compare against PhyML

    let sequences = Sequences::new(vec![
        record!("A", b"-"),
        record!("B", b"-"),
        record!("C", b"A"),
        record!("D", b"T"),
    ]);
    let info = PIB::build_from_objects(sequences, tree.clone()).unwrap();
    assert_relative_eq!(jc69.cost(&info, true), -2.77259, epsilon = 1e-5); // Compare against PhyML

    let sequences = Sequences::new(vec![
        record!("A", b"-"),
        record!("B", b"A"),
        record!("C", b"A"),
        record!("D", b"T"),
    ]);
    let info = PIB::build_from_objects(sequences, tree).unwrap();
    assert_relative_eq!(jc69.cost(&info, true), -2.92044, epsilon = 1e-5); // Compare against PhyML
}

#[test]
fn dna_ambig_chars_against_phyml() {
    let jc69 = SubstModel::<JC69>::new(&[], &[]).unwrap();
    let tree = tree!("(C:0.06465432,D:27.43128366,(A:0.00000001,B:0.00000001)0.0:0.08716381);");
    let sequences = Sequences::new(vec![
        record!("A", b"B"),
        record!("B", b"A"),
        record!("C", b"A"),
        record!("D", b"T"),
    ]);
    let info = PIB::build_from_objects(sequences, tree).unwrap();
    assert_relative_eq!(jc69.cost(&info, true), -21.28936836, epsilon = 1e-7);
}

#[test]
fn dna_x_simple_fully_likely() {
    let jc69 = SubstModel::<JC69>::new(&[], &[]).unwrap();
    let tree = tree!("(A:0.05,B:0.0005):0.0;");
    let sequences = Sequences::new(vec![record!("A", b"X"), record!("B", b"X")]);
    let info = PIB::build_from_objects(sequences, tree).unwrap();
    assert_relative_eq!(jc69.cost(&info, true), 0.0, epsilon = 1e-5);
}

#[cfg(test)]
fn x_fully_likely_template<Q: QMatrix + Display>(freqs: &[f64], params: &[f64]) {
    let model = SubstModel::<Q>::new(freqs, params).unwrap();
    let tree = tree!("(((A:2.0,B:2.0)E:4.0,(C:2.0,D:2.0)F:4.0)G:6.0);");
    let sequences = Sequences::with_alphabet(
        vec![
            record!("A", b"X"),
            record!("B", b"X"),
            record!("C", b"X"),
            record!("D", b"X"),
        ],
        model.qmatrix.alphabet().clone(),
    );
    let info = PIB::build_from_objects(sequences, tree).unwrap();
    assert_relative_eq!(model.cost(&info, true), 0.0, epsilon = 1e-5);
}

#[test]
fn dna_x_fully_likely() {
    x_fully_likely_template::<JC69>(&[], &[]);
    x_fully_likely_template::<K80>(&[], &[]);
    x_fully_likely_template::<HKY>(&[0.22, 0.26, 0.33, 0.19], &[0.5]);
    x_fully_likely_template::<TN93>(&[0.22, 0.26, 0.33, 0.19], &[0.5970915, 0.2940435, 0.00135]);
    x_fully_likely_template::<GTR>(&[0.1, 0.3, 0.4, 0.2], &[5.0, 1.0, 1.0, 1.0, 1.0, 5.0]);
}

#[test]
fn protein_x_fully_likely() {
    x_fully_likely_template::<WAG>(&[], &[]);
    x_fully_likely_template::<HIVB>(&[], &[]);
    x_fully_likely_template::<BLOSUM>(&[], &[]);
}
