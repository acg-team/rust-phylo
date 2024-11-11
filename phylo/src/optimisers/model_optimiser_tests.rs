use rstest::rstest;

use std::path::PathBuf;

use approx::assert_relative_eq;

use crate::evolutionary_models::{
    DNAModelType::*,
    EvoModel,
    ProteinModelType::{self, *},
};
use crate::likelihood::PhyloCostFunction;
use crate::optimisers::{EvoModelOptimiser, FrequencyOptimisation, ModelOptimiser};
use crate::phylo_info::PhyloInfoBuilder;
use crate::pip_model::PIPModel;
use crate::substitution_models::{DNASubstModel, ProteinSubstModel};

#[test]
fn check_likelihood_opt_k80() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sim/K80/K80.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
    )
    .build()
    .unwrap();
    let model = DNASubstModel::new(K80, &[4.0, 1.0]).unwrap();
    let unopt_logl = model.cost(&info, false);
    let o = ModelOptimiser::new(&model, &info, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert!(o.final_logl > unopt_logl);

    let model = o.model.clone();
    let expected_logl = model.cost(&info, false);

    assert_relative_eq!(o.final_logl, expected_logl, epsilon = 1e-6);
    assert_relative_eq!(o.final_logl, -4034.5008033, epsilon = 1e-6);
    assert_relative_eq!(o.model.params.rtc, 1.884815, epsilon = 1e-5);
    assert_relative_eq!(o.model.params.rta, 1.0, epsilon = 1e-5);
}

#[test]
fn frequencies_unchanged_opt_k80() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sim/K80/K80.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
    )
    .build()
    .unwrap();
    let model = DNASubstModel::new(K80, &[4.0, 1.0]).unwrap();
    let o = ModelOptimiser::new(&model, &info, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();
    assert!(o.model.params.pi.iter().all(|&x| x == 0.25));
}

#[test]
fn parameter_definition_after_optim_k80() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sim/K80/K80.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
    )
    .build()
    .unwrap();
    let model = DNASubstModel::new(K80, &[4.0, 1.0]).unwrap();
    let o = ModelOptimiser::new(&model, &info, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    let params = &o.model.params;
    assert_eq!(params.rtc, params.rag);
    assert_eq!(params.rta, params.rtg);
    assert_eq!(params.rta, params.rca);
    assert_eq!(params.rta, params.rcg);
    assert!(params.pi.iter().all(|&x| x == 0.25));
}

#[test]
fn gtr_on_k80_data() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sim/K80/K80.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
    )
    .build()
    .unwrap();
    let model = DNASubstModel::new(
        GTR,
        &[0.25, 0.35, 0.3, 0.1, 0.88, 0.03, 0.00001, 0.07, 0.02, 1.0],
    )
    .unwrap();
    let o = ModelOptimiser::new(&model, &info, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();
    let params = &o.model.params;
    assert_relative_eq!(params.rta, params.rtg, epsilon = 1e-1);
    assert_relative_eq!(params.rta, params.rca, epsilon = 1e-1);
    assert_relative_eq!(params.rta, params.rcg, epsilon = 1e-1);
    assert!(params.pi.iter().all(|&x| x - 0.25 < 1e-2));
}

#[test]
fn parameter_definition_after_optim_hky() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sim/GTR/gtr.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
    )
    .build()
    .unwrap();
    let model = DNASubstModel::new(HKY, &[0.26, 0.2, 0.4, 0.14, 4.0, 1.0]).unwrap();
    let start_logl = model.cost(&info, false);
    let o = ModelOptimiser::new(&model, &info, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();
    let params = &o.model.params;
    assert_eq!(params.rtc, params.rag);
    assert_eq!(params.rta, params.rtg);
    assert_eq!(params.rta, params.rca);
    assert_eq!(params.rta, params.rcg);
    assert!(params.pi.iter().all(|&x| x != 0.25));
    assert_eq!(o.initial_logl, start_logl);
    assert!(o.final_logl > start_logl);
}

#[test]
fn parameter_definition_after_optim_tn93() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sim/GTR/gtr.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
    )
    .build()
    .unwrap();
    let model = DNASubstModel::new(TN93, &[0.26, 0.2, 0.4, 0.14, 4.0, 2.0, 1.0]).unwrap();
    let start_logl = model.cost(&info, false);
    let o = ModelOptimiser::new(&model, &info, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();
    let params = &o.model.params;
    assert_ne!(params.rtc, params.rag);
    assert_eq!(params.rta, params.rtg);
    assert_eq!(params.rta, params.rca);
    assert_eq!(params.rta, params.rcg);
    assert!(params.pi.iter().all(|&x| x != 0.25));
    assert!(o.final_logl > start_logl);
}

#[test]
fn check_parameter_optimisation_gtr() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sim/GTR/gtr.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
    )
    .build()
    .unwrap();
    // Optimized parameters from PhyML
    let phyml_model = DNASubstModel::new(
        GTR,
        &[
            0.24720,
            0.35320,
            0.29540,
            0.10420,
            1.0,
            0.031184397,
            0.000100000,
            0.077275972,
            0.041508690,
            1.0,
        ],
    )
    .unwrap();
    let phyml_logl = phyml_model.cost(&info, false);
    assert_relative_eq!(phyml_logl, -3474.48083, epsilon = 1.0e-5);

    let paml_model = DNASubstModel::new(
        GTR,
        &[
            0.25318, 0.32894, 0.31196, 0.10592, 0.88892, 0.03190, 0.00001, 0.07102, 0.02418, 1.0,
        ],
    )
    .unwrap(); // Original input to paml
    let paml_logl = paml_model.cost(&info, false);
    assert!(phyml_logl > paml_logl);

    let model = DNASubstModel::new(
        GTR,
        &[
            0.24720, 0.35320, 0.29540, 0.10420, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
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
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sim/GTR/gtr.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
    )
    .build()
    .unwrap();
    // Optimized parameters from PhyML
    let phyml_model = DNASubstModel::new(K80, &[19.432093, 1.0]).unwrap();
    let phyml_logl = phyml_model.cost(&info, false);
    assert_relative_eq!(phyml_logl, -3629.2205979421, epsilon = 1.0e-5);

    let model = DNASubstModel::new(K80, &[2.0, 1.0]).unwrap();
    let o = ModelOptimiser::new(&model, &info, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert!(o.final_logl > phyml_logl);

    let o2 = ModelOptimiser::new(&o.model, &info, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert!(o2.final_logl == o.final_logl);
    assert!(o2.iterations < 10);

    assert_relative_eq!(o.model.params.rtc, phyml_model.params.rtc, epsilon = 1e-2);
    assert_relative_eq!(o.model.params.rta, phyml_model.params.rta, epsilon = 1e-5);
    assert_relative_eq!(o.model.params.rtg, phyml_model.params.rtg, epsilon = 1e-5);
    assert_relative_eq!(o.model.params.rca, phyml_model.params.rca, epsilon = 1e-5);
    assert_relative_eq!(o.model.params.rcg, phyml_model.params.rcg, epsilon = 1e-5);
    assert_relative_eq!(o.model.params.rag, phyml_model.params.rag, epsilon = 1e-2);
    assert_relative_eq!(o.final_logl, phyml_logl, epsilon = 1e-6);
}

#[test]
fn check_parameter_optimisation_hky_vs_phyml() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sim/GTR/gtr.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
    )
    .build()
    .unwrap();

    let model = DNASubstModel::new(HKY, &[0.25, 0.25, 0.25, 0.25, 2.0, 1.0]).unwrap();
    let o = ModelOptimiser::new(&model, &info, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();

    let o2 = ModelOptimiser::new(&o.model, &info, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();
    assert!(o2.final_logl == o.final_logl);
    assert!(o2.iterations < 10);

    // Optimized parameters from PhyML
    let phyml_model =
        DNASubstModel::new(HKY, &[0.24720, 0.35320, 0.29540, 0.10420, 20.357397, 1.0]).unwrap();
    let phyml_logl = phyml_model.cost(&info, false);
    assert_relative_eq!(phyml_logl, -3483.9223510041406, epsilon = 1.0e-5);

    assert!(o.final_logl >= phyml_logl);
    assert_relative_eq!(o.model.freqs(), phyml_model.freqs());
    assert_relative_eq!(o.model.params.rtc, phyml_model.params.rtc, epsilon = 1e-2);
    assert_relative_eq!(o.model.params.rta, phyml_model.params.rta, epsilon = 1e-5);
    assert_relative_eq!(o.model.params.rtg, phyml_model.params.rtg, epsilon = 1e-5);
    assert_relative_eq!(o.model.params.rca, phyml_model.params.rca, epsilon = 1e-5);
    assert_relative_eq!(o.model.params.rcg, phyml_model.params.rcg, epsilon = 1e-5);
    assert_relative_eq!(o.model.params.rag, phyml_model.params.rag, epsilon = 1e-2);
    assert_relative_eq!(o.final_logl, phyml_logl, epsilon = 1e-5);
}

#[test]
fn frequencies_fixed_opt_gtr() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sim/GTR/gtr.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
    )
    .build()
    .unwrap();
    let model =
        DNASubstModel::new(GTR, &[0.25, 0.35, 0.3, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
    let o = ModelOptimiser::new(&model, &info, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert!(o.model.params.pi.as_slice() == [0.25, 0.35, 0.3, 0.1]);
}

#[rstest]
#[case::wag(WAG)]
#[case::blosum(BLOSUM)]
#[case::hivb(HIVB)]
fn frequencies_fixed_protein(#[case] model_type: ProteinModelType) {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sequences_protein1.fasta"),
        PathBuf::from("./data/tree_diff_branch_lengths_2.newick"),
    )
    .build()
    .unwrap();
    let model = ProteinSubstModel::new(model_type, &[]).unwrap();
    let initial_llik = model.cost(&info, false);
    let o = ModelOptimiser::new(&model, &info, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert_eq!(initial_llik, o.initial_logl);
    assert_eq!(initial_llik, o.final_logl);
    assert_eq!(model.freqs(), o.model.freqs());
}

#[rstest]
#[case::wag(WAG)]
#[case::blosum(BLOSUM)]
#[case::hivb(HIVB)]
fn frequencies_empirical_protein(#[case] model_type: ProteinModelType) {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sequences_protein1.fasta"),
        PathBuf::from("./data/tree_diff_branch_lengths_2.newick"),
    )
    .build()
    .unwrap();
    let model = ProteinSubstModel::new(model_type, &[]).unwrap();
    let initial_llik = model.cost(&info, false);
    let o = ModelOptimiser::new(&model, &info, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();
    assert_eq!(initial_llik, o.initial_logl);
    assert_ne!(initial_llik, o.final_logl);
    assert_ne!(model.freqs(), o.model.freqs());
}

#[test]
fn check_parameter_optimisation_pip_arpiptest() {
    let info = &PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/pip/arpip/msa.fasta"),
        PathBuf::from("./data/pip/arpip/tree.nwk"),
    )
    .build()
    .unwrap();

    let pip_gtr = PIPModel::<DNASubstModel>::new(
        GTR,
        &[
            0.1, 0.1, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
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
    let info = &PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/pip/propip/msa.initial.fasta"),
        PathBuf::from("./data/pip/propip/tree.nwk"),
    )
    .build()
    .unwrap();

    let pip_gtr = PIPModel::<DNASubstModel>::new(
        GTR,
        &[
            14.142_1, 0.1414, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
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
    let info = &PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example.newick"),
    )
    .build()
    .unwrap();

    let pip_hky =
        PIPModel::<DNASubstModel>::new(HKY, &[1.2, 0.45, 0.25, 0.25, 0.25, 0.25, 1.0]).unwrap();
    assert_relative_eq!(
        pip_hky.cost(info, false),
        -361.1613531649497, // value from the python script
        epsilon = 1e-1
    );
    let o = ModelOptimiser::new(&pip_hky, info, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    let params = &o.model.params.subst_model.params;
    assert_eq!(params.rtc, params.rag);
    assert_eq!(params.rca, params.rta);
    assert_eq!(params.rca, params.rcg);
    assert_eq!(params.rca, params.rtg);
    assert_ne!(o.model.params.mu, 1.2);
    assert_ne!(o.model.params.lambda, 0.45);
    assert!(o.final_logl > -361.1613531649497);
    assert_relative_eq!(
        o.final_logl,
        -227.1894519082493, // value from the python script
        epsilon = 1e-1
    );
}

#[test]
fn optimisation_pip_gtr() {
    let info = &PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sim/GTR/gtr.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
    )
    .build()
    .unwrap();
    let pip_gtr = PIPModel::<DNASubstModel>::new(
        GTR,
        &[
            0.1, 0.1, 0.24720, 0.35320, 0.29540, 0.10420, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
    )
    .unwrap();
    let initial_logl = pip_gtr.cost(info, false);
    let pip_o = ModelOptimiser::new(&pip_gtr, info, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();

    assert_relative_eq!(initial_logl, -9988.486546494, epsilon = 1e0); // value from the python script
    assert_relative_eq!(pip_o.initial_logl, initial_logl);
    assert!(pip_o.final_logl > initial_logl);

    let gtr = DNASubstModel::new(
        GTR,
        &[
            0.24720, 0.35320, 0.29540, 0.10420, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
    )
    .unwrap();
    let o = ModelOptimiser::new(&gtr, info, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();

    let pip_subst_params = pip_o.model.params.subst_model.params;
    let subst_params = o.model.params;
    assert_relative_eq!(pip_subst_params.rtc, subst_params.rtc, epsilon = 1e-4);
    assert_relative_eq!(pip_subst_params.rta, subst_params.rta, epsilon = 1e-4);
    assert_relative_eq!(pip_subst_params.rtg, subst_params.rtg, epsilon = 1e-4);
    assert_relative_eq!(pip_subst_params.rca, subst_params.rca, epsilon = 1e-4);
    assert_relative_eq!(pip_subst_params.rcg, subst_params.rcg, epsilon = 1e-4);
    assert_relative_eq!(pip_subst_params.rag, subst_params.rag, epsilon = 1e-4);
}

#[test]
fn protein_example_pip_opt() {
    let info = &PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/phyml_protein_example/nogap_seqs.fasta"),
        PathBuf::from("./data/phyml_protein_example/tree.newick"),
    )
    .build()
    .unwrap();
    let pip = PIPModel::<ProteinSubstModel>::new(WAG, &[2.0, 0.1]).unwrap();
    let initial_logl = pip.cost(info, false);
    let o = ModelOptimiser::new(&pip, info, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();
    assert!(o.final_logl > initial_logl);
    assert_relative_eq!(o.initial_logl, initial_logl);
    assert_ne!(o.model.params.lambda, 2.0);
    assert_ne!(o.model.params.mu, 0.1);
    assert_eq!(o.model.cost(info, true), o.final_logl);
}
