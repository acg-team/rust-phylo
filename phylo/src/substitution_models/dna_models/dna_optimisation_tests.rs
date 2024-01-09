use std::path::PathBuf;

use approx::assert_relative_eq;

use crate::{
    evolutionary_models::EvolutionaryModelInfo,
    phylo_info::phyloinfo_from_files,
    substitution_models::{
        dna_models::{
            gtr::{self, GTRModelOptimiser},
            DNALikelihoodCost, DNAModelOptimiser, DNASubstParams,
        },
        FreqVector, SubstitutionModelInfo,
    },
};

#[test]
fn check_parameter_optimisation_gtr() {
    let info = phyloinfo_from_files(
        PathBuf::from("./data/sim/GTR/gtr.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
    )
    .unwrap();
    let likelihood = DNALikelihoodCost { info: &info };
    let phyml_params = DNASubstParams {
        pi: FreqVector::from_column_slice(&[0.24720, 0.35320, 0.29540, 0.10420]),
        rtc: 1.0,
        rta: 0.031184397,
        rtg: 0.000100000,
        rca: 0.077275972,
        rcg: 0.041508690,
        rag: 1.0,
    }; // Optimized parameters from PhyML
    let model = gtr::gtr(phyml_params);
    let mut tmp_info = SubstitutionModelInfo::new(likelihood.info, &model).unwrap();
    let phyml_logl = likelihood.compute_log_likelihood(&model, &mut tmp_info);
    assert_relative_eq!(phyml_logl, -3474.48083, epsilon = 1.0e-5);

    let paml_params = DNASubstParams {
        pi: FreqVector::from_column_slice(&[0.25318, 0.32894, 0.31196, 0.10592]),
        rtc: 0.88892,
        rta: 0.03190,
        rtg: 0.00001,
        rca: 0.07102,
        rcg: 0.02418,
        rag: 1.0,
    }; // Original input to paml
    let model = gtr::gtr(paml_params);
    let mut tmp_info = SubstitutionModelInfo::new(likelihood.info, &model).unwrap();
    let paml_logl = likelihood.compute_log_likelihood(&model, &mut tmp_info);
    assert!(phyml_logl > paml_logl);

    let params = DNASubstParams {
        pi: FreqVector::from_column_slice(&[0.24720, 0.35320, 0.29540, 0.10420]),
        rtc: 1.0,
        rta: 1.0,
        rtg: 1.0,
        rca: 1.0,
        rcg: 1.0,
        rag: 1.0,
    };
    let model = gtr::gtr(params);
    let (_, _, logl) = GTRModelOptimiser::new(&likelihood, &model)
        .optimise_parameters()
        .unwrap();
    assert!(logl > phyml_logl);
    assert!(logl > paml_logl);
}
