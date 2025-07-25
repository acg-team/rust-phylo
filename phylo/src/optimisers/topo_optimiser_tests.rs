use std::path::Path;

use approx::assert_relative_eq;

use crate::alignment::{Alignment, Sequences};
use crate::evolutionary_models::FrequencyOptimisation::Empirical;
use crate::likelihood::TreeSearchCost;
use crate::optimisers::{
    BranchOptimiser, ModelOptimiser, PhyloOptimisationResult, TopologyOptimiser,
};
use crate::parsimony::{scoring::ModelScoringBuilder, BasicParsimonyCost, DolloParsimonyCost};
use crate::phylo_info::{PhyloInfo, PhyloInfoBuilder as PIB};
use crate::pip_model::{PIPCost, PIPCostBuilder as PIPCB, PIPModel};
use crate::substitution_models::{
    dna_models::*, protein_models::*, QMatrix, SubstModel, SubstitutionCost,
    SubstitutionCostBuilder as SCB,
};
use crate::{record_wo_desc as record, tree};

// Macros for tests where precomputed results can be used to speed up local testing
macro_rules! define_optimise_trees {
    ($($fn_name:ident: { model = $model:ident, cost = $cost:ident, builder = $builder:ident }),* $(,)?) => {
        $(
            #[cfg(not(feature = "precomputed-test-results"))]
            fn $fn_name<Q: QMatrix + Send>(
                seq_file: &std::path::Path,
                _: &std::path::Path,
                model: $model<Q>,
            ) -> PhyloOptimisationResult<$cost<Q>> {
                let start_info = PIB::new(seq_file).build().unwrap();
                let cost = $builder::new(model, start_info).build().unwrap();
                TopologyOptimiser::new(cost).run().unwrap()
            }

            #[cfg(feature = "precomputed-test-results")]
            fn $fn_name<Q: QMatrix + Send>(
                seq_file: &std::path::Path,
                tree_file: &std::path::Path,
                model: $model<Q>,
            ) -> PhyloOptimisationResult<$cost<Q>> {
                let start_info = PIB::new(seq_file).build().unwrap();

                if let Ok(precomputed) =
                    PIB::with_attrs(seq_file, tree_file).build()
                {
                    let initial_cost = $builder::new(model.clone(), start_info).build().unwrap();
                    let final_cost = $builder::new(model, precomputed.clone()).build().unwrap();

                    PhyloOptimisationResult {
                        initial_cost: initial_cost.cost(),
                        final_cost: final_cost.cost(),
                        iterations: 0,
                        cost: final_cost.clone(),
                    }
                } else {
                    let cost = $builder::new(model.clone(), start_info).build().unwrap();
                    let res = TopologyOptimiser::new(cost).run().unwrap();
                    assert!(crate::io::write_newick_to_file(
                        &[res.cost.tree().clone()],
                        tree_file
                    )
                    .is_ok());
                    res
                }
            }
        )*
    };
}

define_optimise_trees!(
    optimise_tree: { model = SubstModel, cost = SubstitutionCost, builder = SCB },
    optimise_tree_pip: { model = PIPModel, cost = PIPCost, builder = PIPCB },
);

#[test]
fn k80_simple() {
    // Check that optimisation on k80 data improves k80 likelihood when starting from a given tree
    let tree = tree!("(((A:1.0,B:1.0)E:2.0,(C:1.0,D:1.0)F:2.0)G:3.0);");
    let msa = Alignment::from_aligned(
        Sequences::new(vec![
            record!("A", b"CTATATATAC"),
            record!("B", b"ATATATATAA"),
            record!("C", b"TTATATATAT"),
            record!("D", b"TTATATATAT"),
        ]),
        &tree,
    )
    .unwrap();
    let info = PhyloInfo { msa, tree };

    let k80 = SubstModel::<K80>::new(&[], &[4.0, 1.0]);
    let c = SCB::new(k80.clone(), info).build().unwrap();
    let unopt_logl = c.cost();
    let o = TopologyOptimiser::new(c).run().unwrap();

    assert!(o.final_cost >= unopt_logl);
    assert_eq!(o.initial_cost, unopt_logl);
    assert_eq!(o.final_cost, o.cost.cost());

    let c = SCB::new(k80, o.cost.info.clone()).build().unwrap();
    assert_relative_eq!(c.cost(), o.final_cost);
    assert_relative_eq!(c.cost(), o.cost.cost());
}

#[test]
fn k80_sim_data_from_given() {
    // Check that optimisation on k80 data improves k80 likelihood when starting from a given tree
    let fldr = Path::new("./data/sim/K80");
    let info = PIB::with_attrs(fldr.join("K80.fasta"), fldr.join("../tree.newick"))
        .build()
        .unwrap();
    let k80 = SubstModel::<K80>::new(&[], &[4.0, 1.0]);
    let c = SCB::new(k80.clone(), info).build().unwrap();
    let unopt_logl = c.cost();
    let o = TopologyOptimiser::new(c).run().unwrap();

    assert!(o.final_cost >= unopt_logl);
    assert_eq!(o.initial_cost, unopt_logl);
    assert_eq!(o.final_cost, o.cost.cost());
    assert_relative_eq!(o.final_cost, -4060.91964, epsilon = 1e-5);

    let c = SCB::new(k80, o.cost.info.clone()).build().unwrap();
    assert_relative_eq!(c.cost(), o.final_cost);
    assert_relative_eq!(c.cost(), o.cost.cost());
}

#[test]
fn k80_sim_data_from_nj() {
    // Check that optimisation on k80 data improves k80 likelihood when starting from an NJ tree
    let fldr = Path::new("./data/sim/K80");
    let info = PIB::new(fldr.join("K80.fasta")).build().unwrap();
    let k80 = SubstModel::<K80>::new(&[], &[4.0, 1.0]);
    let c = SCB::new(k80.clone(), info).build().unwrap();
    let unopt_logl = c.cost();
    let o = TopologyOptimiser::new(c).run().unwrap();

    assert!(o.final_cost >= unopt_logl);
    assert_eq!(o.initial_cost, unopt_logl);
    assert_eq!(o.final_cost, o.cost.cost());
    assert_relative_eq!(o.final_cost, -4060.91964, epsilon = 1e-5);

    let c = SCB::new(k80, o.cost.info.clone()).build().unwrap();
    assert_relative_eq!(c.cost(), o.final_cost);
    assert_relative_eq!(c.cost(), o.cost.cost());
}

#[test]
fn k80_sim_data_vs_phyml() {
    // Check that optimisation on k80 data under JC69 produces similar tree to PhyML with matching likelihood
    let fldr = Path::new("./data/sim/");
    let info = PIB::with_attrs(fldr.join("K80/K80.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let jc69 = SubstModel::<JC69>::new(&[], &[]);
    let c = SCB::new(jc69.clone(), info).build().unwrap();
    let unopt_logl = c.cost();
    let o = TopologyOptimiser::new(c).run().unwrap();

    assert!(o.final_cost >= unopt_logl);
    assert_eq!(o.initial_cost, unopt_logl);
    assert_eq!(o.final_cost, o.cost.cost());

    let c = SCB::new(jc69.clone(), o.cost.info.clone()).build().unwrap();
    assert_relative_eq!(c.cost(), o.final_cost);
    assert_relative_eq!(c.cost(), o.cost.cost());

    let phyml_info = PIB::with_attrs(
        fldr.join("K80/K80.fasta"),
        fldr.join("K80/phyml_tree.newick"),
    )
    .build()
    .unwrap();

    let phyml_c = SCB::new(jc69, phyml_info.clone()).build().unwrap();
    assert_relative_eq!(o.final_cost, phyml_c.cost(), epsilon = 1e-5);
    assert_relative_eq!(o.final_cost, -4038.721121221992, epsilon = 1e-5);

    let tree = o.cost.tree();
    assert_eq!(tree.robinson_foulds(&phyml_info.tree), 0);

    let taxa = ["Gorilla", "Orangutan", "Gibbon", "Human", "Chimpanzee"];
    for taxon in taxa.iter() {
        assert_relative_eq!(
            tree.node(&tree.idx(taxon)).blen,
            phyml_info.tree.node(&phyml_info.tree.idx(taxon)).blen,
            epsilon = 1e-5
        );
    }
    assert_relative_eq!(tree.height, phyml_info.tree.height, epsilon = 1e-4);
    assert_eq!(tree.robinson_foulds(&phyml_info.tree), 0);
}

#[test]
fn k80_sim_data_vs_phyml_wrong_start() {
    // Check that optimisation on k80 data under JC69 produces similar tree to PhyML with matching likelihood
    // when starting from a wrong tree
    let fldr = Path::new("./data/sim/");
    let info = PIB::with_attrs(fldr.join("K80/K80.fasta"), fldr.join("wrong_tree.newick"))
        .build()
        .unwrap();
    let jc69 = SubstModel::<JC69>::new(&[], &[]);
    let c = SCB::new(jc69.clone(), info).build().unwrap();
    let unopt_logl = c.cost();
    let o = TopologyOptimiser::new(c).run().unwrap();

    assert!(o.final_cost >= unopt_logl);
    assert_eq!(o.initial_cost, unopt_logl);
    assert_eq!(o.final_cost, o.cost.cost());

    let phyml_info = PIB::with_attrs(
        fldr.join("K80/K80.fasta"),
        fldr.join("K80/phyml_tree.newick"),
    )
    .build()
    .unwrap();
    let phyml_c = SCB::new(jc69.clone(), phyml_info.clone()).build().unwrap();

    assert_relative_eq!(o.final_cost, phyml_c.cost(), epsilon = 1e-5);
    assert_relative_eq!(o.final_cost, -4038.721121221992, epsilon = 1e-5);

    let tree = o.cost.tree();
    let taxa = ["Gorilla", "Orangutan", "Gibbon", "Human", "Chimpanzee"];
    for taxon in taxa.iter() {
        assert_relative_eq!(
            tree.by_id(taxon).blen,
            phyml_info.tree.by_id(taxon).blen,
            epsilon = 1e-5
        );
    }
    assert_relative_eq!(tree.height, phyml_info.tree.height, epsilon = 1e-4);
    assert_eq!(tree.robinson_foulds(&phyml_info.tree), 0);
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn wag_no_gaps_vs_phyml_nj_tree_start() {
    // Check that optimisation on protein data under WAG produces similar tree to PhyML with matching likelihoods
    // on sequences without gaps starting from an NJ tree
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("nogap_seqs.fasta");
    let tree_file = fldr.join("jati_wag_nogap_nj_start.newick");

    let wag = SubstModel::<WAG>::new(&[], &[]);
    let res = optimise_tree(&seq_file, &tree_file, wag.clone());
    assert!(res.final_cost >= res.initial_cost);
    let wag_tree = res.cost.tree();

    let phyml_res = PIB::with_attrs(seq_file.clone(), fldr.join("phyml_nogap.newick"))
        .build()
        .unwrap();
    let phyml_logl = SCB::new(wag, phyml_res.clone()).build().unwrap().cost();

    // Compare tree height and logl to the output of PhyML
    assert_relative_eq!(wag_tree.height, phyml_res.tree.height, epsilon = 1e-4);
    assert_eq!(wag_tree.robinson_foulds(&phyml_res.tree), 0);
    assert_relative_eq!(res.final_cost, phyml_logl, epsilon = 1e-5);
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn pip_vs_subst_dna_tree() {
    // Check that optimisation on k80 data under PIP and substitution model produces similar trees
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("K80/K80.fasta"), fldr.join("wrong_tree.newick"))
        .build()
        .unwrap();
    let k80 = SubstModel::<K80>::new(&[], &[4.0]);
    let k80_res = TopologyOptimiser::new(SCB::new(k80.clone(), info.clone()).build().unwrap())
        .run()
        .unwrap();

    let pip = PIPModel::<K80>::new(&[], &[0.5, 0.4, 4.0]);
    let pip_res = TopologyOptimiser::new(PIPCB::new(pip.clone(), info).build().unwrap())
        .run()
        .unwrap();

    // Tree topologies under PIP+K80 and K80 should match
    assert_eq!(pip_res.cost.tree().robinson_foulds(k80_res.cost.tree()), 0);

    // Check that likelihoods under substitution model are similar for both trees
    // but reoptimise branch lengths for PIP tree because they are not comparable
    let k80_pip_tree_res = BranchOptimiser::new(SCB::new(k80, pip_res.cost.info).build().unwrap())
        .run()
        .unwrap();
    assert_relative_eq!(
        k80_res.final_cost,
        k80_pip_tree_res.final_cost,
        epsilon = 1e-6
    );

    // Check that likelihoods under PIP are similar for both trees
    // but reoptimise branch lengths for substitution tree because they are not comparable
    let pip_k80_tree_res =
        BranchOptimiser::new(PIPCB::new(pip, k80_res.cost.info).build().unwrap())
            .run()
            .unwrap();
    assert_relative_eq!(
        pip_res.final_cost,
        pip_k80_tree_res.final_cost,
        epsilon = 1e-6
    );
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn wag_nogaps_pip_vs_subst_tree_nj_start() {
    // Check that optimisation on protein data under WAG produces similar trees for PIP and substitution model
    // on sequences without gaps
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("nogap_seqs.fasta");

    let pip = PIPModel::<WAG>::new(&[], &[50.0, 0.1]);
    let wag = SubstModel::<WAG>::new(&[], &[]);

    let pip_res = optimise_tree_pip(
        &seq_file,
        &fldr.join("jati_pip_nogap_pip_vs_wag.newick"),
        pip.clone(),
    );
    assert!(pip_res.final_cost >= pip_res.initial_cost);
    let pip_tree = pip_res.cost.tree();

    let wag_res = optimise_tree(
        &seq_file,
        &fldr.join("jati_wag_nogap_pip_vs_wag.newick"),
        wag.clone(),
    );
    assert!(wag_res.final_cost >= wag_res.initial_cost);
    let wag_tree = wag_res.cost.tree();

    // Compare tree created with a substitution model to the one with PIP
    assert_eq!(pip_tree.robinson_foulds(wag_tree), 0);

    // Check that likelihoods under same model are similar for both trees
    let pip_tree_reopt_wag_logl = BranchOptimiser::new(
        SCB::new(wag.clone(), pip_res.cost.info.clone())
            .build()
            .unwrap(),
    )
    .run()
    .unwrap()
    .final_cost;
    assert_relative_eq!(wag_res.final_cost, pip_tree_reopt_wag_logl, epsilon = 1e-5);

    // Check that the likelihoods under the same model are similar for both trees
    let wag_tree_reopt_pip_logl =
        BranchOptimiser::new(PIPCB::new(pip.clone(), wag_res.cost.info).build().unwrap())
            .run()
            .unwrap()
            .final_cost;
    assert_relative_eq!(pip_res.final_cost, wag_tree_reopt_pip_logl, epsilon = 1e-5);

    // Just a random check that reestimating branch lengths makes no difference
    let new_pip_cost = PIPCB::new(pip, pip_res.cost.info.clone()).build().unwrap();
    let reopt_res = BranchOptimiser::new(new_pip_cost.clone()).run().unwrap();
    assert_relative_eq!(new_pip_cost.cost(), reopt_res.final_cost, epsilon = 1e-5);
    assert_relative_eq!(
        pip_tree.height,
        reopt_res.cost.info.tree.height,
        epsilon = 1e-5
    );
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn pip_optimise_model_tree() {
    // Check that tree optimisation under PIP has a better likelihood when the model is also optimised
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("seqs.fasta");
    let start_info = PIB::new(seq_file.clone()).build().unwrap();

    let pip = PIPModel::<WAG>::new(&[], &[1.4, 0.5]);
    let res = optimise_tree_pip(
        &seq_file,
        &fldr.join("jati_pip_nj_start.newick"),
        pip.clone(),
    );
    assert!(res.final_cost >= res.initial_cost);

    // Optimise model parameters
    let o = ModelOptimiser::new(
        PIPCB::new(pip.clone(), start_info.clone()).build().unwrap(),
        Empirical,
    )
    .run()
    .unwrap();
    let pip_opt = o.cost.model;

    assert!(o.final_cost >= o.initial_cost);
    assert!(o.final_cost >= res.final_cost);

    let model_opt_res = optimise_tree_pip(
        &seq_file,
        &fldr.join("jati_pip_nj_start_model_opt.newick"),
        pip_opt.clone(),
    );

    assert!(model_opt_res.final_cost >= o.final_cost);
    assert!(model_opt_res.final_cost >= res.initial_cost);

    assert_eq!(
        model_opt_res.cost.tree().robinson_foulds(res.cost.tree()),
        0
    );

    assert!(model_opt_res.final_cost > res.final_cost);
    assert!(
        model_opt_res.final_cost
            > PIPCB::new(pip_opt, res.cost.info.clone())
                .build()
                .unwrap()
                .cost()
    );
    assert!(
        model_opt_res.final_cost
            > PIPCB::new(pip, model_opt_res.cost.info.clone())
                .build()
                .unwrap()
                .cost()
    );
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn wag_vs_phyml_empirical_freqs() {
    // Check that optimisation on protein data under WAG produces similar tree to PhyML with matching likelihoods
    // when using empirical frequencies
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("seqs.fasta");
    let tree_file = fldr.join("jati_wag_empirical.newick");
    let start_info = PIB::new(seq_file.clone()).build().unwrap();

    let wag = SubstModel::<WAG>::new(&[], &[]);
    let o = ModelOptimiser::new(
        SCB::new(wag.clone(), start_info).build().unwrap(),
        Empirical,
    )
    .run()
    .unwrap();

    assert!(o.final_cost >= o.initial_cost);
    let wag_opt = o.cost.model;

    let res = optimise_tree(&seq_file, &tree_file, wag_opt.clone());
    let tree = res.cost.tree();
    assert!(res.final_cost >= res.initial_cost);

    let phyml_res = PIB::with_attrs(seq_file.clone(), fldr.join("phyml_wag_empirical.newick"))
        .build()
        .unwrap();

    assert_relative_eq!(tree.height, phyml_res.tree.height, epsilon = 1e-4);
    assert_eq!(tree.robinson_foulds(&phyml_res.tree), 0);

    let wag_opt_phyml_logl = SCB::new(wag_opt, phyml_res).build().unwrap().cost();
    assert_relative_eq!(res.final_cost, wag_opt_phyml_logl, epsilon = 1e-5);
    assert_relative_eq!(res.final_cost, -5258.79254297163, epsilon = 1e-5);
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn pip_wag_vs_phyml_empirical_freqs() {
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("seqs.fasta");
    let tree_file = fldr.join("jati_pip_wag_empirical.newick");

    let start_info = PIB::new(seq_file.clone()).build().unwrap();
    let pip = PIPModel::<WAG>::new(&[], &[1.0, 2.0]);
    // Use empirical frequencies and optimise lambda and mu for PIP
    let o = ModelOptimiser::new(
        PIPCB::new(pip.clone(), start_info.clone()).build().unwrap(),
        Empirical,
    )
    .run()
    .unwrap();

    let pip_opt = o.cost.model;
    let res = optimise_tree_pip(&seq_file, &tree_file, pip_opt.clone());

    // Optimised tree should be better than the starting tree
    assert!(res.final_cost >= o.final_cost);

    let phyml_res = PIB::with_attrs(seq_file.clone(), fldr.join("phyml_wag_empirical.newick"))
        .build()
        .unwrap();
    // Reoptimise branch lengths for the phyml tree because they are not comparable with PIP
    let phyml_brlen_opt = BranchOptimiser::new(PIPCB::new(pip_opt, phyml_res).build().unwrap())
        .run()
        .unwrap();

    // Check that our tree is better than phyml
    assert!(res.final_cost > phyml_brlen_opt.final_cost);
    assert_relative_eq!(
        res.cost.tree().height,
        phyml_brlen_opt.cost.info.tree.height,
        epsilon = 1e-2
    );
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn wag_vs_phyml_fixed_freqs() {
    // Check that optimisation on protein data under WAG produces similar tree to PhyML with matching likelihoods
    // when using fixed frequencies
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("seqs.fasta");
    let tree_file = fldr.join("jati_wag_fixed.newick");
    let wag = SubstModel::<WAG>::new(&[], &[]);

    let res = optimise_tree(&seq_file, &tree_file, wag.clone());
    assert!(res.final_cost >= res.initial_cost);

    let phyml_res = PIB::with_attrs(seq_file.clone(), fldr.join("phyml_wag_fixed.newick"))
        .build()
        .unwrap();

    let tree = res.cost.tree();
    assert_relative_eq!(tree.height, phyml_res.tree.height, epsilon = 1e-4);
    assert_eq!(tree.robinson_foulds(&phyml_res.tree), 0);
    assert_relative_eq!(
        res.final_cost,
        SCB::new(wag, phyml_res).build().unwrap().cost(),
        epsilon = 1e-5
    );

    assert_relative_eq!(res.final_cost, -5295.08423, epsilon = 1e-3);
}

#[test]
fn basic_parsimony_tree_search() {
    let seqs = Sequences::new(vec![
        record!("A", b"GGA"),
        record!("B", b"GGG"),
        record!("C", b"ACA"),
        record!("D", b"ACG"),
    ]);
    let tree = tree!("((A:1.0,D:1.0):1.0,(C:1.0,B:1.0):1.0):0.0;");

    let info = PhyloInfo {
        msa: Alignment::from_aligned(seqs.clone(), &tree).unwrap(),
        tree,
    };
    let cost = BasicParsimonyCost::new(info).unwrap();
    assert_eq!(cost.cost(), -6.0);

    let res = TopologyOptimiser::new(cost).run().unwrap();
    assert_eq!(res.final_cost, -4.0);
}

#[test]
fn dollo_tree_search() {
    let tree = tree!("(((A:1.0,C:1.0)E:2.0,(C:1.0,B:1.0)F:2.0)G:3.0);");
    let msa = Alignment::from_aligned(
        Sequences::new(vec![
            record!("A", b"TTTTTTTTTTTCTATATATA-"),
            record!("B", b"TTTTTTTTTTTATATATAT-A"),
            record!("C", b"GGGGGGGGGGGTTATATATA-"),
            record!("D", b"GGGGGGGGGGGTTATATATA-"),
        ]),
        &tree,
    )
    .unwrap();
    let info = PhyloInfo { msa, tree };
    let k80 = SubstModel::<K80>::new(&[], &[4.0, 1.0]);
    let scoring = ModelScoringBuilder::new(k80)
        .times(vec![1.0])
        .build()
        .unwrap();

    let c = DolloParsimonyCost::with_scoring(info, scoring);
    let unopt_score = c.cost();
    let o = TopologyOptimiser::new(c).run().unwrap();

    assert!(o.final_cost >= unopt_score);
    assert_eq!(o.initial_cost, unopt_score);
    assert_eq!(o.final_cost, o.cost.cost());
}

#[test]
fn dollo_tree_search_sim_data_simple() {
    let fldr = Path::new("./data/sim/");
    let info = PIB::with_attrs(fldr.join("K80/K80.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let c = DolloParsimonyCost::new(info);
    let unopt_score = c.cost();

    let o = TopologyOptimiser::new(c).run().unwrap();
    assert!(o.final_cost >= unopt_score);
    assert_eq!(o.initial_cost, unopt_score);
    assert_eq!(o.final_cost, o.cost.cost());
}

#[test]
fn dollo_tree_search_sim_data_model() {
    let fldr = Path::new("./data/sim/");
    let info = PIB::with_attrs(fldr.join("K80/K80.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();

    let k80 = SubstModel::<K80>::new(&[], &[4.0, 1.0]);
    let scoring = ModelScoringBuilder::new(k80)
        .times(vec![1.0])
        .build()
        .unwrap();

    let c = DolloParsimonyCost::with_scoring(info, scoring);
    let unopt_score = c.cost();
    let o = TopologyOptimiser::new(c).run().unwrap();

    assert!(o.final_cost >= unopt_score);
    assert_eq!(o.initial_cost, unopt_score);
    assert_eq!(o.final_cost, o.cost.cost());
}
