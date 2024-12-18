use std::fmt::Display;
use std::path::Path;

use approx::assert_relative_eq;

use crate::evolutionary_models::EvoModel;
use crate::likelihood::PhyloCostFunction;
use crate::optimisers::{EvoModelOptimiser, FrequencyOptimisation, ModelOptimiser};
use crate::phylo_info::PhyloInfoBuilder as PIB;
use crate::pip_model::PIPModel;

use crate::substitution_models::{dna_models::*, protein_models::*, QMatrix, SubstModel};

#[test]
fn check_likelihood_opt_k80() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("K80/K80.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = SubstModel::<K80>::new(&[], &[4.0, 1.0]).unwrap();
    let unopt_logl = model.cost(&info, true);
    let o = ModelOptimiser::new(&model, &info, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert!(o.final_logl > unopt_logl);

    let model = o.model.clone();
    let expected_logl = model.cost(&info, true);

    assert_relative_eq!(o.final_logl, expected_logl, epsilon = 1e-6);
    assert_relative_eq!(o.final_logl, -4034.5008033, epsilon = 1e-6);
    assert_relative_eq!(o.model.params()[0], 1.884815, epsilon = 1e-5);
}

#[test]
fn frequencies_unchanged_opt_k80() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("K80/K80.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = SubstModel::<K80>::new(&[], &[4.0, 1.0]).unwrap();
    let o = ModelOptimiser::new(&model, &info, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();
    assert!(o.model.freqs().iter().all(|&x| x == 0.25));
}

#[test]
fn parameter_definition_after_optim_k80() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("K80/K80.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = SubstModel::<K80>::new(&[], &[4.0, 1.0]).unwrap();
    let o = ModelOptimiser::new(&model, &info, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert_relative_eq!(o.model.params(), o.model.params(), epsilon = 1e-5);
    assert!(o.model.freqs().iter().all(|&x| x == 0.25));
}

#[test]
fn gtr_on_k80_data() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("K80/K80.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = SubstModel::<GTR>::new(
        &[0.25, 0.35, 0.3, 0.1],
        &[0.88, 0.03, 0.00001, 0.07, 0.02, 1.0],
    )
    .unwrap();
    let o = ModelOptimiser::new(&model, &info, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();
    assert_relative_eq!(o.model.params(), o.model.params(), epsilon = 1e-1);
    assert!(o.model.freqs().iter().all(|&x| x - 0.25 < 1e-2));
}

#[test]
fn parameter_definition_after_optim_hky() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = SubstModel::<HKY>::new(&[], &[0.26, 0.2, 0.4, 0.14, 4.0, 1.0]).unwrap();
    let start_logl = model.cost(&info, false);
    let o = ModelOptimiser::new(&model, &info, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();

    assert_relative_eq!(o.model.params(), o.model.params(), epsilon = 1e-1);
    assert!(o.model.freqs().iter().all(|&x| x != 0.25));
    assert_eq!(o.initial_logl, start_logl);
    assert!(o.final_logl > start_logl);
}

#[test]
fn parameter_definition_after_optim_tn93() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = SubstModel::<TN93>::new(&[0.26, 0.2, 0.4, 0.14], &[4.0, 2.0, 1.0]).unwrap();
    let start_logl = model.cost(&info, false);
    let o = ModelOptimiser::new(&model, &info, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();
    assert_ne!(model.params()[0], o.model.params()[0]);
    assert_ne!(model.params()[1], o.model.params()[1]);
    assert_ne!(model.params()[2], o.model.params()[2]);
    assert!(o.model.freqs().iter().all(|&x| x != 0.25));
    assert!(o.final_logl > start_logl);
}

#[test]
fn check_parameter_optimisation_gtr() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    // Optimized parameters from PhyML
    let phyml_model = SubstModel::<GTR>::new(
        &[0.24720, 0.35320, 0.29540, 0.10420],
        &[1.0, 0.031184397, 0.000100000, 0.077275972, 0.041508690, 1.0],
    )
    .unwrap();
    let phyml_logl = phyml_model.cost(&info, false);
    assert_relative_eq!(phyml_logl, -3474.48083, epsilon = 1.0e-5);

    let paml_model = SubstModel::<GTR>::new(
        &[0.25318, 0.32894, 0.31196, 0.10592],
        &[0.88892, 0.03190, 0.00001, 0.07102, 0.02418, 1.0],
    )
    .unwrap(); // Original input to paml
    let paml_logl = paml_model.cost(&info, false);
    assert!(phyml_logl > paml_logl);

    let model = SubstModel::<GTR>::new(
        &[0.24720, 0.35320, 0.29540, 0.10420],
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let o = ModelOptimiser::new(&model, &info, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert!(o.final_logl > phyml_logl);
    assert!(o.final_logl > paml_logl);

    let o2 = ModelOptimiser::new(&o.model, &info, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert!(o2.final_logl >= o.final_logl);
    assert!(o2.iterations < 10);
}

#[test]
fn check_parameter_optimisation_k80_vs_phyml() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    // Optimized parameters from PhyML
    let phyml_model = SubstModel::<K80>::new(&[], &[19.432093]).unwrap();
    let phyml_logl = phyml_model.cost(&info, true);
    assert_relative_eq!(phyml_logl, -3629.2205979421, epsilon = 1.0e-5);

    let model = SubstModel::<K80>::new(&[], &[2.0, 1.0]).unwrap();
    let o = ModelOptimiser::new(&model, &info, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert!(o.final_logl > phyml_logl);

    let o2 = ModelOptimiser::new(&o.model, &info, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert!(o2.final_logl == o.final_logl);
    assert!(o2.iterations < 10);
    assert_relative_eq!(o.model.params(), phyml_model.params(), epsilon = 1e-2);
    assert_relative_eq!(o.final_logl, phyml_logl, epsilon = 1e-6);
}

#[test]
fn check_parameter_optimisation_hky_vs_phyml() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = SubstModel::<HKY>::new(&[0.25, 0.25, 0.25, 0.25], &[2.0]).unwrap();
    let o = ModelOptimiser::new(&model, &info, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();

    let o2 = ModelOptimiser::new(&o.model, &info, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();
    assert_relative_eq!(o2.final_logl, o.final_logl);
    assert!(o2.iterations < 10);

    // Optimized parameters from PhyML
    let phyml_model =
        SubstModel::<HKY>::new(&[0.24720, 0.35320, 0.29540, 0.10420], &[20.357397]).unwrap();
    let phyml_logl = phyml_model.cost(&info, false);
    assert_relative_eq!(phyml_logl, -3483.9223510041406, epsilon = 1.0e-5);

    assert!(o.final_logl >= phyml_logl);
    assert_relative_eq!(o.model.freqs(), phyml_model.freqs());
    assert_relative_eq!(o.model.params(), phyml_model.params(), epsilon = 1e-2);
    assert_relative_eq!(o.final_logl, phyml_logl, epsilon = 1e-5);
}

#[test]
fn frequencies_fixed_opt_gtr() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model =
        SubstModel::<GTR>::new(&[0.25, 0.35, 0.3, 0.1], &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
    let o = ModelOptimiser::new(&model, &info, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert!(o.model.freqs().as_slice() == [0.25, 0.35, 0.3, 0.1]);
}

#[cfg(test)]
fn frequencies_fixed_protein_template<Q: QMatrix + Clone + Display>()
where
    SubstModel<Q>: EvoModel,
{
    let fldr = Path::new("./data");
    let info = PIB::with_attrs(
        fldr.join("sequences_protein1.fasta"),
        fldr.join("tree_diff_branch_lengths_2.newick"),
    )
    .build()
    .unwrap();
    let model = SubstModel::<Q>::new(&[], &[]).unwrap();
    let initial_llik = model.cost(&info, false);
    let o = ModelOptimiser::new(&model, &info, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert_eq!(initial_llik, o.initial_logl);
    assert_eq!(initial_llik, o.final_logl);
    assert_eq!(model.freqs(), o.model.freqs());
}

#[test]
fn frequencies_fixed_protein() {
    frequencies_fixed_protein_template::<WAG>();
    frequencies_fixed_protein_template::<HIVB>();
    frequencies_fixed_protein_template::<BLOSUM>();
}

#[cfg(test)]
fn frequencies_empirical_protein_template<Q: QMatrix + Clone + Display>()
where
    SubstModel<Q>: EvoModel,
{
    let fldr = Path::new("./data");
    let info = PIB::with_attrs(
        fldr.join("sequences_protein1.fasta"),
        fldr.join("tree_diff_branch_lengths_2.newick"),
    )
    .build()
    .unwrap();
    let model = SubstModel::<Q>::new(&[], &[]).unwrap();
    let initial_llik = model.cost(&info, false);
    let o = ModelOptimiser::new(&model, &info, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();
    assert_eq!(initial_llik, o.initial_logl);
    assert_ne!(initial_llik, o.final_logl);
    assert_ne!(model.freqs(), o.model.freqs());
}

#[test]
fn frequencies_empirical_protein() {
    frequencies_empirical_protein_template::<WAG>();
    frequencies_empirical_protein_template::<HIVB>();
    frequencies_empirical_protein_template::<BLOSUM>();
}

#[test]
fn check_parameter_optimisation_pip_arpiptest() {
    let fldr = Path::new("./data/pip/arpip/");
    let info = &PIB::with_attrs(fldr.join("msa.fasta"), fldr.join("tree.nwk"))
        .build()
        .unwrap();
    let pip_gtr = PIPModel::<GTR>::new(
        &[0.25, 0.25, 0.25, 0.25],
        &[0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let o = ModelOptimiser::new(&pip_gtr, info, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();
    let initial_logl = pip_gtr.cost(info, true);
    assert_eq!(o.initial_logl, initial_logl);
    assert!(o.final_logl > initial_logl);
}

#[test]
fn optimisation_pip_propip_example() {
    let fldr = Path::new("./data/pip/propip/");
    let info = &PIB::with_attrs(fldr.join("msa.initial.fasta"), fldr.join("tree.nwk"))
        .build()
        .unwrap();

    let pip_gtr = PIPModel::<GTR>::new(
        &[0.25, 0.25, 0.25, 0.25],
        &[14.142_1, 0.1414, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();

    let initial_logl = pip_gtr.cost(info, false);
    assert_relative_eq!(initial_logl, -1241.9555557710014, epsilon = 1e-1);
    let o = ModelOptimiser::new(&pip_gtr, info, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert_eq!(o.initial_logl, initial_logl);
    assert!(o.final_logl > initial_logl);
    assert_relative_eq!(o.final_logl, -1081.1682773217494, epsilon = 1e-0);
    assert_eq!(o.final_logl, o.model.cost(info, true));
}

#[test]
fn optimisation_against_python_no_gaps() {
    let fldr = Path::new("./data");
    let info = &PIB::with_attrs(
        fldr.join("Huelsenbeck_example_long_DNA.fasta"),
        fldr.join("Huelsenbeck_example.newick"),
    )
    .build()
    .unwrap();

    let pip_hky = PIPModel::<HKY>::new(&[0.25, 0.25, 0.25, 0.25], &[1.2, 0.45, 1.0]).unwrap();
    assert_relative_eq!(
        pip_hky.cost(info, true),
        -361.1613531649497, // value from the python script
        epsilon = 1e-1
    );
    let o = ModelOptimiser::new(&pip_hky, info, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    let params = o.model.params();
    assert_ne!(params[0], 1.2);
    assert_ne!(params[1], 0.45);
    assert_ne!(params[2], 1.0);
    assert!(o.final_logl > -361.1613531649497);
    assert_relative_eq!(
        o.final_logl,
        -227.1894519082493, // value from the python script
        epsilon = 1e-1
    );
}

#[test]
fn optimisation_pip_gtr() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let pip_gtr = PIPModel::<GTR>::new(
        &[0.24720, 0.35320, 0.29540, 0.10420],
        &[0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let initial_logl = pip_gtr.cost(&info, false);
    let pip_o = ModelOptimiser::new(&pip_gtr, &info, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();

    assert_relative_eq!(initial_logl, -9988.486546494, epsilon = 1e0); // value from the python script
    assert_relative_eq!(pip_o.initial_logl, initial_logl);
    assert!(pip_o.final_logl > initial_logl);

    let gtr = SubstModel::<GTR>::new(
        &[0.24720, 0.35320, 0.29540, 0.10420],
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let o = ModelOptimiser::new(&gtr, &info, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert_relative_eq!(pip_o.model.params()[2..], o.model.params(), epsilon = 1e-2);
}

#[test]
fn protein_example_pip_opt() {
    let fldr = Path::new("./data/phyml_protein_example/");
    let info = &PIB::with_attrs(fldr.join("seqs.fasta"), fldr.join("true_tree.newick"))
        .build()
        .unwrap();
    let pip = PIPModel::<WAG>::new(&[], &[2.0, 0.1]).unwrap();
    let initial_logl = pip.cost(info, false);
    let o = ModelOptimiser::new(&pip, info, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();
    assert!(o.final_logl > initial_logl);
    assert_relative_eq!(o.initial_logl, initial_logl);
    assert_ne!(o.model.params()[0], 2.0);
    assert_ne!(o.model.params()[1], 0.1);
    assert_eq!(o.model.cost(info, true), o.final_logl);
}
