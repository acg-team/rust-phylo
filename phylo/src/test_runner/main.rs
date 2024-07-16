use std::path::PathBuf;

use approx::assert_relative_eq;

use phylo::likelihood::LikelihoodCostFunction;
use phylo::phylo_info::{GapHandling, PhyloInfo};
use phylo::substitution_models::dna_models::{
    dna_model_optimiser::DNAModelOptimiser, DNALikelihoodCost,
};
use phylo::substitution_models::dna_models::{gtr, DNASubstParams};
use phylo::substitution_models::FreqVector;

fn main() {
    let info = PhyloInfo::from_files(
        PathBuf::from("./data/sim/GTR/gtr.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
        &GapHandling::Ambiguous,
    )
    .unwrap();
    let likelihood = DNALikelihoodCost { info: &info };
    let phyml_params = DNASubstParams {
        pi: frequencies!(&[0.24720, 0.35320, 0.29540, 0.10420]),
        rtc: 1.0,
        rta: 0.031184397,
        rtg: 0.000100000,
        rca: 0.077275972,
        rcg: 0.041508690,
        rag: 1.0,
    }; // Optimized parameters from PhyML
    let model = gtr::gtr(phyml_params);
    let phyml_logl = LikelihoodCostFunction::compute_log_likelihood(&likelihood, &model);
    assert_relative_eq!(phyml_logl, -3474.48083, epsilon = 1.0e-5);

    let paml_params = DNASubstParams {
        pi: frequencies!(&[0.25318, 0.32894, 0.31196, 0.10592]),
        rtc: 0.88892,
        rta: 0.03190,
        rtg: 0.00001,
        rca: 0.07102,
        rcg: 0.02418,
        rag: 1.0,
    }; // Original input to paml
    let model = gtr::gtr(paml_params);
    let paml_logl = LikelihoodCostFunction::compute_log_likelihood(&likelihood, &model);
    assert!(phyml_logl > paml_logl);

    let params = DNASubstParams {
        pi: frequencies!(&[0.24720, 0.35320, 0.29540, 0.10420]),
        rtc: 1.0,
        rta: 1.0,
        rtg: 1.0,
        rca: 1.0,
        rcg: 1.0,
        rag: 1.0,
    };
    let (_, opt_params, logl) = DNAModelOptimiser::new(&likelihood)
        .optimise_gtr_parameters(&params)
        .unwrap();
    assert!(logl > phyml_logl);
    assert!(logl > paml_logl);

    let (iters, _, double_opt_logl) = DNAModelOptimiser::new(&likelihood)
        .optimise_gtr_parameters(&opt_params)
        .unwrap();
    assert!(double_opt_logl >= logl);
    assert!(iters < 10);
}
