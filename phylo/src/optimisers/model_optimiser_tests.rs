use std::path::Path;

use approx::assert_relative_eq;

use crate::evolutionary_models::{EvoModel, FrequencyOptimisation};
use crate::frequencies;
use crate::likelihood::ModelSearchCost;
use crate::optimisers::ModelOptimiser;
use crate::phylo_info::PhyloInfoBuilder as PIB;
use crate::pip_model::{PIPCostBuilder, PIPModel};
use crate::substitution_models::{
    dna_models::*, protein_models::*, FreqVector, QMatrix, QMatrixMaker, SubstModel,
    SubstitutionCostBuilder as SCB,
};

#[test]
fn likelihood_improves_k80() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("K80/K80.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = SubstModel::<K80>::new(&[], &[4.0, 1.0]);

    let c = SCB::new(model, info.clone()).build().unwrap();
    let unopt_logl = c.cost();
    let o = ModelOptimiser::new(c, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert!(o.final_cost > unopt_logl);

    let model = SubstModel::<K80>::new(o.cost.freqs().into(), o.cost.params());
    let c2 = SCB::new(model, info).build().unwrap();
    assert_relative_eq!(o.initial_cost, unopt_logl);
    assert_relative_eq!(o.final_cost, o.cost.cost());
    assert_relative_eq!(o.final_cost, c2.cost());
    assert_relative_eq!(o.final_cost, -4034.500803, epsilon = 1e-5);
    assert_relative_eq!(o.cost.params()[0], 1.884815, epsilon = 1e-5);
}

#[test]
fn frequencies_unchanged_k80() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("K80/K80.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = SubstModel::<K80>::new(&[], &[4.0, 1.0]);
    let c = SCB::new(model.clone(), info).build().unwrap();
    let initial_logl = c.cost();
    let o = ModelOptimiser::new(c, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();
    assert_eq!(initial_logl, o.initial_cost);
    assert_eq!(o.cost.cost(), o.final_cost);
    assert_ne!(o.cost.model.params(), model.params());
    assert_relative_eq!(o.cost.freqs(), &frequencies!(&[0.25; 4]));
    assert_relative_eq!(o.cost.freqs(), model.freqs());
}

#[test]
fn parameter_change_k80() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("K80/K80.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = SubstModel::<K80>::new(&[], &[4.0, 1.0]);
    let c = SCB::new(model.clone(), info).build().unwrap();
    let initial_logl = c.cost();
    let o = ModelOptimiser::new(c, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert_eq!(initial_logl, o.initial_cost);
    assert_eq!(o.cost.cost(), o.final_cost);
    assert_ne!(o.cost.model.params(), model.params());
    assert_relative_eq!(o.cost.freqs(), &frequencies!(&[0.25; 4]));
    assert_relative_eq!(o.cost.freqs(), model.freqs());
}

#[test]
fn gtr_on_k80_data() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("K80/K80.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = SubstModel::<GTR>::new(&[0.25, 0.35, 0.3, 0.1], &[0.88, 0.03, 0.00001, 0.07, 0.02]);
    let c = SCB::new(model.clone(), info.clone()).build().unwrap();
    let o_gtr = ModelOptimiser::new(c, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();

    let model = SubstModel::<K80>::new(&[], &[4.0, 1.0]);
    let c = SCB::new(model.clone(), info.clone()).build().unwrap();
    let o_k80 = ModelOptimiser::new(c, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();

    assert!(o_gtr.final_cost >= o_k80.final_cost);
    assert_relative_eq!(
        o_gtr.cost.freqs(),
        &frequencies!(&[0.25; 4]),
        epsilon = 1e-1
    );
}

#[cfg(test)]
fn improved_logl_fixed_freqs_template<Q: QMatrix + QMatrixMaker>() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = SubstModel::<Q>::new(&[], &[]);

    let c = SCB::new(model.clone(), info).build().unwrap();
    let initial_logl = c.cost();
    let o = ModelOptimiser::new(c, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();

    assert_eq!(initial_logl, o.initial_cost);
    assert_eq!(o.cost.cost(), o.final_cost);
    assert!(o.final_cost > initial_logl);
    assert_ne!(o.cost.model.params(), model.params());
    assert_eq!(o.cost.freqs(), model.freqs());
}

#[test]
fn improved_logl_fixed_freqs() {
    improved_logl_fixed_freqs_template::<K80>();
    improved_logl_fixed_freqs_template::<HKY>();
    improved_logl_fixed_freqs_template::<TN93>();
    improved_logl_fixed_freqs_template::<GTR>();
}

#[cfg(test)]
fn improved_logl_empirical_freqs_template<Q: QMatrix + QMatrixMaker>() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = SubstModel::<Q>::new(&[], &[]);

    let c = SCB::new(model.clone(), info).build().unwrap();
    let initial_logl = c.cost();
    let o = ModelOptimiser::new(c, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();

    assert_eq!(initial_logl, o.initial_cost);
    assert_eq!(o.cost.cost(), o.final_cost);
    assert!(o.final_cost > initial_logl);
    assert_ne!(o.cost.model.params(), model.params());
    assert_ne!(o.cost.freqs(), model.freqs());
    assert_eq!(o.cost.freqs(), &o.cost.empirical_freqs());
}

#[test]
fn improved_logl_empirical_freqs() {
    improved_logl_empirical_freqs_template::<HKY>();
    improved_logl_empirical_freqs_template::<TN93>();
    improved_logl_empirical_freqs_template::<GTR>();
}

#[test]
fn gtr_vs_phyml() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    // Optimized parameters from PhyML
    let phyml_model = SubstModel::<GTR>::new(
        &[0.24720, 0.35320, 0.29540, 0.10420],
        &[1.0, 0.031184397, 0.000100000, 0.077275972, 0.041508690, 1.0],
    );
    let phyml_logl = SCB::new(phyml_model, info.clone()).build().unwrap().cost();
    assert_relative_eq!(phyml_logl, -3474.48083, epsilon = 1.0e-5);

    let paml_model = SubstModel::<GTR>::new(
        &[0.25318, 0.32894, 0.31196, 0.10592],
        &[0.88892, 0.03190, 0.00001, 0.07102, 0.02418, 1.0],
    ); // Original input to paml
    let paml_logl = SCB::new(paml_model, info.clone()).build().unwrap().cost();
    assert!(phyml_logl > paml_logl);

    let model = SubstModel::<GTR>::new(
        &[0.24720, 0.35320, 0.29540, 0.10420],
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    );
    let c = SCB::new(model, info.clone()).build().unwrap();
    let o = ModelOptimiser::new(c, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert!(o.final_cost > phyml_logl);
    assert!(o.final_cost > paml_logl);

    let o2 = ModelOptimiser::new(o.cost, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert!(o2.final_cost >= o.final_cost);
    assert!(o2.iterations < 10);
}

#[test]
fn k80_vs_phyml() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    // Optimized parameters from PhyML
    let phyml_model = SubstModel::<K80>::new(&[], &[19.432093]);
    let phyml_logl = SCB::new(phyml_model.clone(), info.clone())
        .build()
        .unwrap()
        .cost();
    assert_relative_eq!(phyml_logl, -3629.2205979421, epsilon = 1.0e-5);

    let model = SubstModel::<K80>::new(&[], &[2.0, 1.0]);
    let o = ModelOptimiser::new(
        SCB::new(model, info.clone()).build().unwrap(),
        FrequencyOptimisation::Fixed,
    )
    .run()
    .unwrap();
    assert!(o.final_cost > phyml_logl);

    let o2 = ModelOptimiser::new(o.cost.clone(), FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert_relative_eq!(o2.final_cost, o.final_cost, epsilon = 1e-10);
    assert!(o2.final_cost >= o.final_cost);
    assert!(o2.iterations < 10);
    assert_relative_eq!(o.cost.params(), phyml_model.params(), epsilon = 1e-2);
    assert_relative_eq!(o.final_cost, phyml_logl, epsilon = 1e-6);
}

#[test]
fn hky_vs_phyml() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = SubstModel::<HKY>::new(&[0.25, 0.25, 0.25, 0.25], &[2.0]);
    let o = ModelOptimiser::new(
        SCB::new(model, info.clone()).build().unwrap(),
        FrequencyOptimisation::Empirical,
    )
    .run()
    .unwrap();

    let o2 = ModelOptimiser::new(o.cost.clone(), FrequencyOptimisation::Empirical)
        .run()
        .unwrap();
    assert_relative_eq!(o2.final_cost, o.final_cost);
    assert!(o2.iterations < 10);

    // Optimized parameters from PhyML
    let phyml_model = SubstModel::<HKY>::new(&[0.24720, 0.35320, 0.29540, 0.10420], &[20.357397]);
    let phyml_logl = SCB::new(phyml_model.clone(), info).build().unwrap().cost();
    assert_relative_eq!(phyml_logl, -3483.9223510041406, epsilon = 1.0e-5);

    assert!(o.final_cost >= phyml_logl);
    assert_relative_eq!(o.cost.freqs(), phyml_model.freqs());
    assert_relative_eq!(o.cost.params(), phyml_model.params(), epsilon = 1e-2);
    assert_relative_eq!(o.final_cost, phyml_logl, epsilon = 1e-5);
    assert_eq!(o.final_cost, o.cost.cost());
}

#[test]
fn frequencies_fixed_opt_gtr() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = SubstModel::<GTR>::new(&[0.25, 0.35, 0.3, 0.1], &[1.0; 5]);
    let o = ModelOptimiser::new(
        SCB::new(model, info.clone()).build().unwrap(),
        FrequencyOptimisation::Fixed,
    )
    .run()
    .unwrap();
    assert_eq!(o.cost.freqs(), &frequencies!(&[0.25, 0.35, 0.3, 0.1]));
    assert_ne!(o.cost.params(), &[1.0; 5]);
    assert_eq!(o.final_cost, o.cost.cost());
}

#[cfg(test)]
fn frequencies_fixed_protein_template<Q: QMatrix + QMatrixMaker>() {
    let fldr = Path::new("./data");
    let info = PIB::with_attrs(
        fldr.join("sequences_protein1.fasta"),
        fldr.join("tree_diff_branch_lengths_2.newick"),
    )
    .build()
    .unwrap();
    let model = SubstModel::<Q>::new(&[], &[]);
    let c = SCB::new(model.clone(), info).build().unwrap();

    let initial_llik = c.cost();
    let o = ModelOptimiser::new(c, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert_eq!(initial_llik, o.initial_cost);
    assert_eq!(initial_llik, o.final_cost);
    assert_eq!(model.freqs(), o.cost.freqs());
    assert_eq!(o.final_cost, o.cost.cost());
}

#[test]
fn frequencies_fixed_protein() {
    frequencies_fixed_protein_template::<WAG>();
    frequencies_fixed_protein_template::<HIVB>();
    frequencies_fixed_protein_template::<BLOSUM>();
}

#[cfg(test)]
fn frequencies_empirical_protein_template<Q: QMatrix + QMatrixMaker>() {
    let fldr = Path::new("./data");
    let info = PIB::with_attrs(
        fldr.join("sequences_protein1.fasta"),
        fldr.join("tree_diff_branch_lengths_2.newick"),
    )
    .build()
    .unwrap();
    let model = SubstModel::<Q>::new(&[], &[]);
    let c = SCB::new(model.clone(), info).build().unwrap();

    let initial_llik = c.cost();
    let o = ModelOptimiser::new(c, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();
    assert_eq!(initial_llik, o.initial_cost);
    assert_ne!(initial_llik, o.final_cost);
    assert_ne!(model.freqs(), o.cost.freqs());
}

#[test]
fn frequencies_empirical_protein() {
    frequencies_empirical_protein_template::<WAG>();
    frequencies_empirical_protein_template::<HIVB>();
    frequencies_empirical_protein_template::<BLOSUM>();
}

#[test]
fn arpip_example() {
    let fldr = Path::new("./data/pip/arpip/");
    let info = PIB::with_attrs(fldr.join("msa.fasta"), fldr.join("tree.nwk"))
        .build()
        .unwrap();
    let pip_gtr = PIPModel::<GTR>::new(
        &[0.25, 0.25, 0.25, 0.25],
        &[0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    );
    let c = PIPCostBuilder::new(pip_gtr.clone(), info.clone())
        .build()
        .unwrap();
    let initial_logl = c.cost();

    let o = ModelOptimiser::new(c, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();
    assert_eq!(o.initial_cost, initial_logl);
    assert_relative_eq!(o.initial_cost, -212.3260420492571, epsilon = 1e-6); // value from the python script

    assert!(o.final_cost > initial_logl);
    assert_ne!(o.cost.params(), pip_gtr.params());
    assert_ne!(o.cost.freqs(), pip_gtr.freqs());
    assert_eq!(o.final_cost, o.cost.cost());

    let pip_gtr = PIPModel::<GTR>::new(o.cost.freqs().as_slice(), o.cost.params());
    let c = PIPCostBuilder::new(pip_gtr, info.clone()).build().unwrap();
    assert_eq!(o.final_cost, c.cost());

    assert_relative_eq!(o.final_cost, -161.70972212811094, epsilon = 1e-6); // value from python script
}

#[test]
fn pip_propip_example() {
    let fldr = Path::new("./data/pip/propip/");
    let info = PIB::with_attrs(fldr.join("msa.initial.fasta"), fldr.join("tree.nwk"))
        .build()
        .unwrap();

    let pip_gtr = PIPModel::<GTR>::new(
        &[0.25, 0.25, 0.25, 0.25],
        &[14.142_1, 0.1414, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    );
    let c = PIPCostBuilder::new(pip_gtr.clone(), info).build().unwrap();
    let initial_logl = c.cost();

    assert_relative_eq!(initial_logl, -1241.99424826303, epsilon = 1e-8); // value from the python script
    assert_relative_eq!(initial_logl, -1241.9944955187807, epsilon = 1e-3); // value from ProPIP
    let o = ModelOptimiser::new(c, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert_eq!(o.initial_cost, initial_logl);
    assert!(o.final_cost > initial_logl);
    assert_relative_eq!(o.final_cost, -1081.7242569614295, epsilon = 1e-8); // value from the python script
    assert_eq!(o.final_cost, o.cost.cost());
}

#[test]
fn pip_vs_python_no_gaps() {
    let fldr = Path::new("./data");
    let info = PIB::with_attrs(
        fldr.join("Huelsenbeck_example_long_DNA.fasta"),
        fldr.join("Huelsenbeck_example.newick"),
    )
    .build()
    .unwrap();

    let pip_hky = PIPModel::<HKY>::new(&[0.25; 4], &[1.2, 0.45, 1.0]);

    let c = PIPCostBuilder::new(pip_hky, info).build().unwrap();
    assert_relative_eq!(c.cost(), -361.18634412281443, epsilon = 1e-7); // value from the python script
    let o = ModelOptimiser::new(c, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    let params = o.cost.params();
    assert_ne!(params[0], 1.2);
    assert_ne!(params[1], 0.45);
    assert_ne!(params[2], 1.0);
    assert!(o.final_cost > -361.18634412281443);
    assert_relative_eq!(o.final_cost, -227.0848511231626, epsilon = 1e-7); // value from the python script
    assert_eq!(o.final_cost, o.cost.cost());
}

#[test]
fn pip_gtr_optimisation() {
    // Check that pip parameter optimisation produces expected results
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let pip_gtr = PIPModel::<GTR>::new(
        &[0.24720, 0.35320, 0.29540, 0.10420],
        &[0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    );
    let c = PIPCostBuilder::new(pip_gtr, info.clone()).build().unwrap();

    let initial_logl = c.cost();
    let pip_o = ModelOptimiser::new(c, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    let pip_gtr = PIPModel::<GTR>::new(pip_o.cost.freqs().as_slice(), pip_o.cost.params());

    let c = PIPCostBuilder::new(pip_gtr, info.clone()).build().unwrap();
    assert_eq!(pip_o.final_cost, c.cost());

    assert_relative_eq!(initial_logl, -9988.840775519875, epsilon = 1e-5); // value from the python script
    assert_relative_eq!(pip_o.initial_cost, initial_logl);
    assert!(pip_o.final_cost > initial_logl);
    assert_relative_eq!(pip_o.final_cost, -3481.828364024475, epsilon = 1e-5); // value from the python script
    assert_eq!(pip_o.final_cost, pip_o.cost.cost());
}

#[test]
fn pip_gtr_vs_gtr_params() {
    // Compare pip gtr parameter optimisation vs the original gtr model
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let pip_gtr = PIPModel::<GTR>::new(
        &[0.24720, 0.35320, 0.29540, 0.10420],
        &[0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    );
    let c = PIPCostBuilder::new(pip_gtr, info.clone()).build().unwrap();
    let pip_o = ModelOptimiser::new(c, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();

    let gtr = SubstModel::<GTR>::new(
        &[0.24720, 0.35320, 0.29540, 0.10420],
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    );
    let c = SCB::new(gtr, info).build().unwrap();
    let o = ModelOptimiser::new(c, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert_relative_eq!(pip_o.cost.params()[2..], o.cost.params(), epsilon = 1e-2);
}

#[test]
fn pip_protein_example() {
    let fldr = Path::new("./data/phyml_protein_example/");
    let info = PIB::with_attrs(fldr.join("seqs.fasta"), fldr.join("example_tree.newick"))
        .build()
        .unwrap();
    let pip = PIPModel::<WAG>::new(&[], &[2.0, 0.1]);
    let c = PIPCostBuilder::new(pip, info).build().unwrap();
    let initial_logl = c.cost();
    let o = ModelOptimiser::new(c, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();
    assert!(o.final_cost > initial_logl);
    assert_relative_eq!(o.initial_cost, initial_logl);
    assert_ne!(o.cost.params()[0], 2.0);
    assert_ne!(o.cost.params()[1], 0.1);
    assert_eq!(o.cost.cost(), o.final_cost);
}
